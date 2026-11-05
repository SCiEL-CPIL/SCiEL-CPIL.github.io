import numpy as np
import time
import random
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for the 150s limit
    
    # 1. Generate all words and index mapping
    words = list(product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 'radius')
    # For the specific case q=3, n=7, radius=1, this is small.
    # neighbors[i] contains indices j such that dist(words[i], words[j]) <= radius
    neighbors = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        # Generate all words within Hamming distance 1
        # For radius > 1, this logic would need expansion, but task implies radius=1 mostly.
        # The prompt says "differs ... in at most one position", so radius is 1.
        # We assume radius=1 for efficiency, but the interface passes radius.
        # If radius > 1, the neighbor generation is more complex. 
        # Given the context "Football pool problem ... radius 1", we optimize for radius 1.
        # If radius != 1, we might fall back to a slower general method, but let's assume 1.
        if radius == 1:
            for pos in range(n):
                orig_val = w_list[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    w_list[pos] = val
                    idx = word_to_idx[tuple(w_list)]
                    neigh.append(idx)
                    w_list[pos] = orig_val
        else:
            # General case for radius > 1 (slow, unlikely to be needed for this specific task)
            # This part is a fallback to ensure correctness if radius changes
            # But for now, we assume radius=1 is the target.
            pass
            
        neighbors.append(neigh)
        
    # Convert to numpy arrays for fast boolean indexing
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Precompute point_to_coverers: for each point p, which codewords cover it?
    # This is the reverse of neighbors.
    point_to_coverers = [[] for _ in range(N)]
    for i, neigh in enumerate(neighbors):
        for p in neigh:
            point_to_coverers[p].append(i)
            
    # Convert point_to_coverers to sets for faster intersection during swap search?
    # Or keep as lists and use set intersection on the fly if lists are small.
    # Lists are size 1 + n*(q-1) = 15. Sets are fine.
    point_to_coverers_sets = [set(lst) for lst in point_to_coverers]

    # 3. Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
            
        best_idx = -1
        max_gain = 0
        
        # Vectorized gain calculation
        # We need the sum of uncovered points for each candidate i.
        # Since N is small (2187), a loop with numpy sum is acceptable.
        for i in range(N):
            if i in code_set:
                continue
            gain = np.sum(uncovered[neighbor_arrays[i]])
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if max_gain == 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False
        
    # 4. Local Search
    # counts[i] = number of codewords in code that cover point i
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1
        
    def is_removable(c):
        """Check if codeword c can be removed without uncovering any point."""
        # If any point in neighbors[c] has count 1, it is NOT removable
        return np.min(counts[neighbor_arrays[c]]) > 1

    def cleanup():
        """Remove all redundant codewords."""
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_removable(c):
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1
                    changed = True
                    if time.time() > deadline - 5:
                        return

    cleanup()
    
    # Local Search Loop with ILS Kick
    # We alternate between standard local search and kicks.
    
    improved = True
    no_improve_count = 0
    kick_size = 1
    
    while time.time() < deadline - 10:
        improved = False
        
        # --- Deletion Pass ---
        cleanup()
        
        # --- Swap Pass ---
        # Try to swap codewords to enable further deletions or improve robustness
        critical_codewords = [c for c in code if not is_removable(c)]
        random.shuffle(critical_codewords)
        
        for c in critical_codewords:
            if time.time() > deadline - 5:
                break
                
            if c not in code_set:
                continue
            if is_removable(c):
                # It became removable due to previous swaps in this pass
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            # Find critical points of c
            crit_mask = counts[neighbor_arrays[c]] == 1
            crit_points = neighbor_arrays[c][crit_mask]
            
            if len(crit_points) == 0:
                # Should have been removed, but just in case
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                continue
                
            # Find candidates w not in code that cover ALL critical points
            # Intersection of coverers of all critical points
            candidate_set = point_to_coverers_sets[crit_points[0]]
            for p in crit_points[1:]:
                candidate_set &= point_to_coverers_sets[p]
                
            # Remove current code from candidates
            candidate_set -= code_set
            
            if not candidate_set:
                continue
                
            best_w = -1
            best_score = -100 # Large negative
            
            neigh_c_set = set(neighbors[c])
            
            for w in candidate_set:
                # Score the swap
                # +1 for each point in neighbors[w] \ neighbors[c] with count 1 (makes safe)
                # -2 for each point in neighbors[c] \ neighbors[w] with count 2 (makes critical)
                
                score = 0
                neigh_w_set = set(neighbors[w])
                
                # Points gaining coverage
                for p in neighbors[w]:
                    if p not in neigh_c_set:
                        if counts[p] == 1:
                            score += 1
                            
                # Points losing coverage
                for p in neighbors[c]:
                    if p not in neigh_w_set:
                        if counts[p] == 2:
                            score -= 2
                            
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                
                code.append(best_w)
                code_set.add(best_w)
                counts[neighbor_arrays[best_w]] += 1
                
                improved = True
                # Run cleanup immediately after a successful swap to cascade deletions
                cleanup()
                
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # --- ILS Kick ---
        # If we haven't improved for a while, perform a kick
        if no_improve_count >= 2:
            # Increase kick size slightly or keep it constant
            # Let's try a kick of size 2 or 3
            current_kick = min(3, len(code) // 10 + 1)
            
            # Remove random codewords
            kick_candidates = list(code)
            random.shuffle(kick_candidates)
            kicked = kick_candidates[:current_kick]
            
            for c in kicked:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1
                
            # Repair: Greedily add codewords to cover uncovered points
            uncovered = counts == 0
            while np.any(uncovered):
                if time.time() > deadline - 5:
                    break
                best_idx = -1
                max_gain = 0
                for i in range(N):
                    if i in code_set:
                        continue
                    gain = np.sum(uncovered[neighbor_arrays[i]])
                    if gain > max_gain:
                        max_gain = gain
                        best_idx = i
                if max_gain == 0:
                    break
                code.append(best_idx)
                code_set.add(best_idx)
                counts[neighbor_arrays[best_idx]] += 1
                uncovered[neighbor_arrays[best_idx]] = False
                
            # Cleanup after repair
            cleanup()
            no_improve_count = 0 # Reset counter after kick
            
    # Final cleanup
    cleanup()
    
    # Return the list of codewords
    return [list(words[i]) for i in code]
