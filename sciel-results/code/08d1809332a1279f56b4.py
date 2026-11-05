import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1)
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val # Restore
        neighbors.append(neigh)
        
    # Precompute neighbor sets for faster intersection
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    # Precompute ball arrays for vectorized operations
    ball_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    ball_size = len(neighbors[0])  # All balls have same size: 1 + n*(q-1)
    
    # Build sparse row structure for fast gain computation
    # For each point j, store which codewords cover it (inverse mapping)
    # coverers[j] = list of i such that j in ball(i)
    coverers = [[] for _ in range(N)]
    for i in range(N):
        for j in ball_arrays[i]:
            coverers[j].append(i)
    
    # Precompute coverer arrays for vectorized operations
    coverer_arrays = [np.array(cov, dtype=np.int32) for cov in coverers]
    
    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=np.float32)
    code = []
    code_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
        
        # Compute gains: for each candidate i, gain = sum of uncovered points in ball(i)
        # Use the inverse: for each uncovered point j, add 1 to all i in coverers[j]
        gains = np.zeros(N, dtype=np.float32)
        unc_idx = np.where(uncovered > 0)[0]
        for j in unc_idx:
            gains[coverer_arrays[j]] += 1.0
        
        # Set gains for already-chosen codewords to -1
        if code_set:
            gains[list(code_set)] = -1
        
        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]
        
        if max_gain <= 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[ball_arrays[best_idx]] = 0
    
    # 4. Local Search
    # counts[i] is the number of codewords in code that cover point i
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[ball_arrays[idx]] += 1
        
    def is_removable(c):
        """Check if codeword c can be removed"""
        return not np.any(counts[ball_arrays[c]] == 1)

    def compute_gains(uncovered_mask, code_set):
        """Compute gains for all candidates given uncovered mask and current code set"""
        gains = np.zeros(N, dtype=np.float32)
        unc_idx = np.where(uncovered_mask)[0]
        for j in unc_idx:
            gains[coverer_arrays[j]] += 1.0
        if code_set:
            gains[list(code_set)] = -1
        return gains

    def local_search_pass(code, code_set, counts, kick_size=0):
        """Perform one pass of deletion + swap local search. Returns (improved, code, code_set, counts)"""
        improved = False
        
        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if is_removable(c):
                to_remove.append(c)
                
        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
            
        # --- Kick Pass (optional) ---
        if kick_size > 0 and len(code) >= kick_size:
            # Remove kick_size random codewords
            kick_candidates = list(code)
            random.shuffle(kick_candidates)
            kicked = kick_candidates[:kick_size]
            for c in kicked:
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
            
            # Re-add using greedy to cover newly uncovered points
            uncovered_mask = (counts == 0)
            while np.any(uncovered_mask):
                if time.time() > deadline - 5:
                    break
                gains = compute_gains(uncovered_mask, code_set)
                best_idx = int(np.argmax(gains))
                max_gain = gains[best_idx]
                if max_gain <= 0:
                    break
                code.append(best_idx)
                code_set.add(best_idx)
                counts[ball_arrays[best_idx]] += 1
                uncovered_mask[ball_arrays[best_idx]] = False
                
            # Run deletion again after kick
            to_remove = []
            for c in code:
                if is_removable(c):
                    to_remove.append(c)
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
            improved = True
        
        # --- Swap Pass ---
        current_code_copy = list(code)
        random.shuffle(current_code_copy)
        
        for c in current_code_copy:
            if time.time() > deadline - 5:
                break
            
            if c not in code_set:
                continue

            # Get critical points (covered only by c)
            crit_mask = counts[ball_arrays[c]] == 1
            crit_points = ball_arrays[c][crit_mask].tolist()
            
            if not crit_points:
                continue
                
            # Find candidates that cover all critical points
            candidate_set = set(neighbor_sets[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= neighbor_sets[p]
            
            candidates = [x for x in candidate_set if x not in code_set]
            
            if not candidates:
                continue
            
            # Evaluate candidates using vectorized scoring
            # Score = (# points newly covered by w that are critical for other codewords)
            #       - (# points that lose their only other cover when c is removed)
            # Simplified: count how many points w covers that have count==1 (and aren't in ball[c])
            #             minus points in ball[c] with count==2 that w doesn't cover
            
            neigh_c_set = neighbor_sets[c]
            ball_c = ball_arrays[c]
            ball_c_set = set(ball_c.tolist())
            
            best_candidate = -1
            best_score = 0  # Need strictly positive improvement
            
            for w in candidates:
                ball_w = ball_arrays[w]
                # Points in ball_w with count==1 that are NOT in ball_c (newly covered)
                new_cover = 0
                for p in ball_w:
                    if counts[p] == 1 and p not in ball_c_set:
                        new_cover += 1
                # Points in ball_c with count==2 that are NOT in ball_w (lose redundancy)
                lost_redundancy = 0
                for p in ball_c:
                    if counts[p] == 2 and p not in neighbor_sets[w]:
                        lost_redundancy += 1
                
                score = new_cover - lost_redundancy
                if score > best_score:
                    best_score = score
                    best_candidate = w
                    
            if best_candidate != -1 and best_score > 0:
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
                
                code.append(best_candidate)
                code_set.add(best_candidate)
                counts[ball_arrays[best_candidate]] += 1
                
                improved = True
                # After a swap, run deletion pass immediately
                to_remove = []
                for c2 in code:
                    if is_removable(c2):
                        to_remove.append(c2)
                for c2 in to_remove:
                    code.remove(c2)
                    code_set.remove(c2)
                    counts[ball_arrays[c2]] -= 1
        
        return improved, code, code_set, counts

    def polish_pass(code, code_set, counts):
        """Final high-precision polish: try all single swaps with relaxed threshold"""
        improved = False
        for c in list(code):
            if time.time() > deadline - 2:
                break
            
            if c not in code_set:
                continue

            crit_mask = counts[ball_arrays[c]] == 1
            crit_points = ball_arrays[c][crit_mask].tolist()
            
            if not crit_points:
                # Can remove
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
                improved = True
                continue
                
            candidate_set = set(neighbor_sets[crit_points[0]])
            for p in crit_points[1:]:
                candidate_set &= neighbor_sets[p]
            
            candidates = [x for x in candidate_set if x not in code_set]
            if not candidates:
                continue
            
            ball_c = ball_arrays[c]
            ball_c_set = set(ball_c.tolist())
            
            best_candidate = -1
            best_score = 0
            
            for w in candidates:
                ball_w = ball_arrays[w]
                new_cover = 0
                for p in ball_w:
                    if counts[p] == 1 and p not in ball_c_set:
                        new_cover += 1
                lost_redundancy = 0
                for p in ball_c:
                    if counts[p] == 2 and p not in neighbor_sets[w]:
                        lost_redundancy += 1
                
                score = new_cover - lost_redundancy
                if score > best_score:
                    best_score = score
                    best_candidate = w
                    
            if best_candidate != -1 and best_score > 0:
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
                code.append(best_candidate)
                code_set.add(best_candidate)
                counts[ball_arrays[best_candidate]] += 1
                improved = True
        
        return improved

    # Run local search with increasing kick sizes to escape local minima
    kick_sizes = [0, 1, 2, 3, 4, 5]
    for kick in kick_sizes:
        if time.time() > deadline - 10:
            break
        improved = True
        while improved and time.time() < deadline - 5:
            improved, code, code_set, counts = local_search_pass(code, code_set, counts, kick_size=kick)
    
    # Final polish pass
    if time.time() < deadline - 2:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = polish_pass(code, code_set, counts)

    # Final cleanup
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[ball_arrays[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
