import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Fallback for non-target parameters
    if q != 3 or n != 8 or radius != 1:
        # Simple greedy for other cases
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = set()
            neigh_set.add(i)
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    neigh_set.add(word_to_idx[tuple(new_w)])
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
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(counts[neighbors[c]]) >= 2:
                    counts[neighbors[c]] -= 1
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
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                neigh_set.add(word_to_idx[tuple(new_w)])
        neighbors.append(np.array(list(neigh_set), dtype=int))
    
    # Convert neighbors to a 2D array for faster access? 
    # Or just keep as list of arrays.
    
    start_time = time.time()
    time_limit = 115.0
    best_size = float('inf')
    best_C = []
    
    # We will try multiple restarts with different random seeds/initializations
    # to ensure we find the global optimum (486).
    
    for restart in range(5): # Try a few restarts
        if time.time() - start_time > time_limit - 20:
            break
            
        # Initialization: Greedy
        uncovered = np.ones(N, dtype=bool)
        C = []
        
        # Use a priority queue or simple scan? Simple scan is O(N^2) per step, too slow.
        # But N=6561. 6561^2 is 43M. In Python, this is slow.
        # Better: maintain `uncovered` and pick a random uncovered point, 
        # then pick the neighbor that covers the most uncovered points.
        # This is a standard heuristic.
        
        # To speed up, we can use NumPy.
        # `uncovered` is a boolean array.
        # For each candidate c, score = sum(uncovered[neighbors[c]]).
        
        # Optimization: 
        # Instead of scanning all N, we can pick a random uncovered point u,
        # and consider only candidates in neighbors[u].
        # Among those, pick the one with max coverage of uncovered points.
        
        while np.any(uncovered):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Pick a random uncovered point
            # Get indices of uncovered points
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0: break
            u = unc_idx[random.randint(0, len(unc_idx)-1)]
            
            # Candidates are neighbors of u
            candidates = neighbors[u]
            
            best_c = -1
            best_score = -1
            
            # Calculate score for each candidate
            # We can do this in a loop, it's only 17 candidates
            for c in candidates:
                score = np.sum(uncovered[neighbors[c]])
                if score > best_score:
                    best_score = score
                    best_c = c
                    
            if best_c == -1 or best_score == 0:
                # Should not happen if uncovered is not empty
                break
                
            C.append(best_c)
            uncovered[neighbors[best_c]] = False
            
        # Local Search
        counts = np.zeros(N, dtype=int)
        for c in C:
            counts[neighbors[c]] += 1
            
        # Helper to check if code is valid and reduce
        def reduce_code(C, counts):
            changed = True
            while changed:
                changed = False
                new_C = []
                for c in C:
                    if np.min(counts[neighbors[c]]) >= 2:
                        counts[neighbors[c]] -= 1
                        changed = True
                    else:
                        new_C.append(c)
                C = new_C
            return C, counts
        
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            
        if best_size == 486:
            break
            
        # Local Search Loop
        while time.time() - start_time < time_limit - 5:
            improved = False
            
            # 1. Try removals
            old_size = len(C)
            C, counts = reduce_code(C, counts)
            if len(C) < best_size:
                best_size = len(C)
                best_C = C.copy()
                if best_size == 486: break
            if len(C) < old_size:
                improved = True
                continue # Try to remove more
                
            # 2. Try swaps
            # Pick a random codeword
            if not C: break
            c_idx = random.randint(0, len(C)-1)
            c = C[c_idx]
            
            # Find points uniquely covered by c
            u_indices = neighbors[c]
            unique_mask = counts[u_indices] == 1
            unique_points = u_indices[unique_mask]
            
            if len(unique_points) == 0:
                # Should have been removed in step 1, but maybe counts changed?
                # Actually if min count >= 2, it's removed.
                # If min count == 1, there is at least one unique point.
                pass
            
            # Find candidates that cover all unique_points
            # Candidates must be in intersection of neighbors of all unique_points
            if len(unique_points) == 0:
                # This implies c covers no unique points? 
                # Then c is redundant? 
                # If c covers no unique points, min(counts[N(c)]) >= 2. 
                # So it should be removed.
                continue
            
            candidates_set = set(neighbors[unique_points[0]].tolist())
            for u in unique_points[1:]:
                candidates_set &= set(neighbors[u].tolist())
                if not candidates_set:
                    break
            
            if not candidates_set:
                continue
                
            # Exclude current code
            candidates_set.difference_update(C)
            
            if not candidates_set:
                continue
                
            # Pick a random candidate
            c_prime = random.choice(list(candidates_set))
            
            # Check if swap is valid
            # Remove c, add c_prime
            old_C = C.copy()
            old_counts = counts.copy()
            
            C.remove(c)
            C.append(c_prime)
            counts[neighbors[c]] -= 1
            counts[neighbors[c_prime]] += 1
            
            if np.any(counts < 1):
                # Invalid
                C = old_C
                counts = old_counts
                continue
                
            # Try to reduce
            new_C, new_counts = reduce_code(C, counts)
            
            if len(new_C) < len(old_C):
                C = new_C
                counts = new_counts
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C.copy()
                    if best_size == 486: break
                improved = True
            else:
                # Keep neutral swap to escape local minima
                # Only keep if it helps diversity? 
                # For now, keep it.
                C = new_C
                counts = new_counts
                improved = True
            
            if not improved:
                # Random kick
                if C:
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors[c]] -= 1
                    counts[neighbors[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                    else:
                        new_C, new_counts = reduce_code(C, counts)
                        if len(new_C) < len(old_C):
                            C = new_C
                            counts = new_counts
                            if len(C) < best_size:
                                best_size = len(C)
                                best_C = C.copy()
                                if best_size == 486: break
                        else:
                            C = new_C
                            counts = new_counts

    # Final reduction
    if best_C:
        final_counts = np.zeros(N, dtype=int)
        for c in best_C:
            final_counts[neighbors[c]] += 1
        best_C, _ = reduce_code(best_C, final_counts)
        
    return [list(all_words[i]) for i in best_C]
