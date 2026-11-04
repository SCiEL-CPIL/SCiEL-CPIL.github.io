import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters (keep existing logic or simple greedy)
        # The existing fallback is okay, but let's make it slightly faster if possible.
        # For now, keep the existing fallback logic as it's not the target.
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0: break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit: break
            
        # Reduction
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(np.array(list(neigh_set), dtype=int))
        
    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        # Iterate through all words to find max coverage of uncovered
        for i in range(N):
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    # 2. Local Search
    # State: C (list of indices), counts (array of coverage counts)
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors[c]] += 1
        return counts

    def reduce_code(code, counts):
        """
        Reduces code by removing redundant words.
        Returns new_code and new_counts.
        """
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()
        
        while True:
            removed_any = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors[c]] -= 1
                    removed_any = True
                else:
                    next_code.append(c)
            current_code = next_code
            if not removed_any:
                break
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        C, counts = reduce_code(C, counts)
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        # Try to find a swap that enables further reduction or is neutral but changes state
        improved = False
        # We iterate through C. If we make a swap, we break and restart the reduction/swap loop.
        for c in C:
            if time.time() - start_time > time_limit:
                break
                
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                continue # Should have been removed
                
            # If too many unique points, unlikely to find a single replacement
            if len(unique_points) > 3:
                continue
                
            # Find candidates c' that cover all unique_points
            # Intersection of neighbors[u] for u in unique_points
            # Start with neighbors of first unique point
            if len(unique_points) == 1:
                candidates_set = set(neighbors[unique_points[0]].tolist())
            else:
                candidates_set = set(neighbors[unique_points[0]].tolist())
                for u in unique_points[1:]:
                    candidates_set &= set(neighbors[u].tolist())
                    if not candidates_set:
                        break
            
            if not candidates_set:
                continue
            
            # Filter candidates: must not be in C
            # We want to replace c with c'
            # c' must cover unique_points.
            # If c' is in C, it's already there. If we remove c, c' stays.
            # But if c' is in C and c' != c, then c' was covering its own points.
            # The condition for removing c is that its unique points are covered by others.
            # If c' is in C, and c' covers unique_points, then c can be removed without adding anything.
            # This is a pure removal, which reduce_code should have caught?
            # No, reduce_code checks if min(counts) >= 2.
            # If c' is in C, then counts[unique_points] >= 2 (one from c, one from c').
            # So reduce_code would have removed c.
            # Therefore, if we are in the swap phase, unique_points are covered ONLY by c.
            # So c' MUST be NOT in C.
            
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
                
            # Pick a candidate
            # Heuristic: Pick one that maximizes overlap with other codewords?
            # Or just pick one arbitrarily.
            # Let's pick one that has the highest number of neighbors in C?
            # This might create more redundancy.
            best_c_prime = None
            max_overlap = -1
            # To speed up, just pick the first one?
            # Let's try picking one that is "dense" in C.
            # But computing overlap for all candidates is expensive.
            # Let's just pick one randomly or the first one.
            # Randomness might help escape cycles.
            c_prime = next(iter(candidates_set))
            
            # Apply swap
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(c_prime)
            
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            # Check validity (should be valid)
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
                
            # Try to reduce
            new_C, new_counts = reduce_code(C, counts)
            
            if len(new_C) < len(old_C):
                # Improvement!
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    best_counts = counts.copy()
                improved = True
                break # Restart loop
                
            else:
                # No reduction.
                # Keep the swap?
                # If we keep it, we might escape a local minimum.
                # If we revert, we might cycle.
                # Let's keep it if it's a "neutral" move that changes the structure.
                # But to avoid infinite loops of neutral swaps, we can limit them.
                # For now, let's keep it.
                # C and counts are already updated.
                improved = True
                break # Restart loop to see if this new state allows reduction
                
    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
