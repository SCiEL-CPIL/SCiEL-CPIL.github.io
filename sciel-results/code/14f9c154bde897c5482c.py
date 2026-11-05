import numpy as np
import itertools
import time
import random
from collections import deque
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # For the specific case q=3, n=7, radius=1
    # But keep it general enough
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors for radius 1 (including self)
    neighbors = []
    neighbor_sets = []
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
                w_list[pos] = orig_val
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))
    
    # Convert to numpy arrays for fast operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Phase 1: Greedy Set Cover with heap
    def greedy_cover(random_seed=0):
        rng = random.Random(random_seed)
        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # Initialize heap with (gain, random_tie_breaker, index)
        # Gain is number of uncovered points covered
        # We need to update gains dynamically, which is hard with a simple heap.
        # Alternative: Recalculate gains for top candidates or use a bucket queue.
        # Given N=2187, recalculating gains for all candidates is O(N * avg_deg) ~ 2187 * 19 ~ 40k ops.
        # Doing this K times (K~175) is ~7M ops. This is fast enough in Python/NumPy if vectorized.
        
        while np.any(uncovered):
            if time.time() > deadline - 30:
                break
            
            # Compute gains: for each candidate, how many uncovered points it covers
            gains = np.zeros(N, dtype=np.int32)
            uncovered_indices = np.where(uncovered)[0]
            
            for idx in uncovered_indices:
                gains[neighbors[idx]] += 1
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Randomized selection among top candidates
            top_candidates = np.where(gains == max_gain)[0]
            best_idx = top_candidates[rng.randint(0, len(top_candidates)-1)]
            
            code.append(int(best_idx))
            code_set.add(int(best_idx))
            uncovered[neighbors[best_idx]] = False
        
        return code, code_set
    
    # Phase 2: Local search with 2-for-1 and 1-for-1
    def local_search(code, code_set, time_budget=60):
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors[idx]] += 1
        
        best_code = code[:]
        best_code_set = code_set.copy()
        best_size = len(code)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # --- Deletion Pass ---
            to_remove = []
            for c in code:
                # Check if all neighbors are covered by others
                # min count in neighbors[c] must be > 1
                if np.min(counts[neighbors[c]]) > 1:
                    to_remove.append(c)
            
            if to_remove:
                improved = True
                for c in to_remove:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors[c]] -= 1
            
            if len(code) < best_size:
                best_code = code[:]
                best_code_set = code_set.copy()
                best_size = len(code)
            
            # --- 2-for-1 Replacement Pass ---
            # Try to find two codewords c1, c2 that can be replaced by one c3
            # This reduces size by 1.
            # We sample pairs to keep it fast.
            K = len(code)
            if K >= 2:
                # Number of pairs to check
                num_pairs_to_check = min(500, K * (K - 1) // 2)
                checked_pairs = set()
                found_replacement = False
                
                for _ in range(num_pairs_to_check):
                    if time.time() > deadline - 5:
                        break
                    if found_replacement:
                        break
                    
                    i, j = random.sample(range(K), 2)
                    if (i, j) in checked_pairs or (j, i) in checked_pairs:
                        continue
                    checked_pairs.add((i, j))
                    checked_pairs.add((j, i))
                    
                    c1 = code[i]
                    c2 = code[j]
                    
                    # Determine points that would become uncovered if c1, c2 removed
                    # U = { x | counts[x] - (1 if x in N(c1) else 0) - (1 if x in N(c2) else 0) == 0 }
                    # We only care about x in N(c1) U N(c2)
                    union_neigh = set(neighbors[c1]).union(neighbors[c2])
                    U = []
                    for x in union_neigh:
                        drop = 0
                        if x in neighbor_sets[c1]: drop += 1
                        if x in neighbor_sets[c2]: drop += 1
                        if counts[x] - drop == 0:
                            U.append(x)
                    
                    if not U:
                        # Both are redundant together, just remove them
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbors[c1]] -= 1
                        counts[neighbors[c2]] -= 1
                        improved = True
                        found_replacement = True
                        break
                    
                    # Try to find c3 covering U
                    # c3 must be in N(p) for some p in U
                    # Pick first p in U
                    p = U[0]
                    cands = [x for x in neighbors[p] if x not in code_set]
                    
                    best_c3 = -1
                    for c3 in cands:
                        # Check if c3 covers all of U
                        valid = all(k in neighbor_sets[c3] for k in U)
                        if valid:
                            # Optional: Score c3 based on how much it helps elsewhere
                            # For now, just take the first valid one
                            best_c3 = c3
                            break
                    
                    if best_c3 != -1:
                        code.remove(c1)
                        code.remove(c2)
                        code_set.remove(c1)
                        code_set.remove(c2)
                        counts[neighbors[c1]] -= 1
                        counts[neighbors[c2]] -= 1
                        
                        code.append(best_c3)
                        code_set.add(best_c3)
                        counts[neighbors[best_c3]] += 1
                        
                        improved = True
                        found_replacement = True
                        break
            
            # --- 1-for-1 Swap Pass (Optimization, not size reduction, but helps escape local minima for size reduction) ---
            if not improved:
                random.shuffle(code)
                for c in code:
                    if time.time() > deadline - 5:
                        break
                    
                    # Get critical points: points covered only by c
                    crit_points = [i for i in neighbors[c] if counts[i] == 1]
                    
                    if not crit_points:
                        continue
                    
                    # Try to find a replacement that covers all critical points
                    found_swap = False
                    random.shuffle(crit_points)
                    
                    # Use the first critical point to generate candidates
                    p = crit_points[0]
                    cands = [x for x in neighbors[p] if x not in code_set]
                    
                    best_w = -1
                    best_score = -1
                    
                    for w in cands:
                        # Check if w covers all critical points
                        valid = all(k in neighbor_sets[w] for k in crit_points)
                        if not valid:
                            continue
                        
                        # Score: maximize coverage of points with count==2 (potential for future deletion)
                        score = sum(1 for i in neighbors[w] if counts[i] == 2)
                        
                        if score > best_score:
                            best_score = score
                            best_w = w
                            
                    if best_w != -1:
                        code.remove(c)
                        code_set.remove(c)
                        counts[neighbors[c]] -= 1
                        
                        code.append(best_w)
                        code_set.add(best_w)
                        counts[neighbors[best_w]] += 1
                        
                        improved = True
                        found_swap = True
                        break
            
            # Final cleanup after any changes
            if improved:
                # Run deletion pass again immediately
                to_remove = []
                for c in code:
                    if np.min(counts[neighbors[c]]) > 1:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code.remove(c)
                        code_set.remove(c)
                        counts[neighbors[c]] -= 1
            
            if len(code) < best_size:
                best_code = code[:]
                best_code_set = code_set.copy()
                best_size = len(code)
        
        # Final cleanup
        counts = np.zeros(N, dtype=np.int32)
        for idx in best_code:
            counts[neighbors[idx]] += 1
        
        changed = True
        while changed:
            changed = False
            for c in list(best_code):
                if np.min(counts[neighbors[c]]) > 1:
                    best_code.remove(c)
                    best_code_set.remove(c)
                    counts[neighbors[c]] -= 1
                    changed = True
        
        return best_code
    
    # Phase 3: Multi-start with different random seeds
    best_code = None
    best_size = float('inf')
    
    num_starts = 8
    for seed in range(num_starts):
        if time.time() > deadline - 40:
            break
        
        code, code_set = greedy_cover(random_seed=seed)
        
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)
        
        # Local search
        refined = local_search(code, code_set)
        
        if len(refined) < best_size:
            best_code = refined
            best_size = len(refined)
    
    # Final phase: More aggressive local search on the best solution
    if best_code is not None and time.time() < deadline - 10:
        best_code_set = set(best_code)
        final_code = local_search(best_code, best_code_set)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)
    
    if best_code is None:
        # Fallback: return empty or a trivial solution
        best_code = [0]
    
    return [list(words[i]) for i in best_code]
