import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for non-target parameters
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
    
    # Precompute neighbors as lists of ints for faster set operations if needed, 
    # but numpy arrays are better for vectorized counting.
    # Let's use lists for the "candidates" intersection logic, as set intersection on lists is fast for small sets.
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(list(neigh_set))
        
    # Convert to numpy arrays for fast coverage counting
    neighbors_np = [np.array(n_list, dtype=int) for n_list in neighbors]

    # 1. Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0
    
    # Greedy: pick word covering most uncovered
    while np.any(uncovered):
        # Vectorized calculation of coverage count for all words
        # This is expensive: 6561 * 17 elements. 
        # Instead of looping, we can try to be smart, but a loop is likely fine for 500 iterations.
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors_np[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        if time.time() - start_time > time_limit:
            break
            
    # 2. Local Search
    # State: C (list of codeword indices), counts (np.array of coverage counts)
    
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_np[c]] += 1
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
        
        # We use a queue or just iterate until stable
        # Since removal reduces size, we can just loop
        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                # Check if c has any unique points
                # Unique if count == 1
                if np.min(current_counts[neighbors_np[c]]) >= 2:
                    # Remove c
                    current_counts[neighbors_np[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()
    
    # Helper to find candidates that cover a set of points
    def find_candidates(points):
        if not points:
            return set(range(N)) # Any word works
        # Intersection of neighbors of all points in 'points'
        cand_set = set(neighbors[points[0]])
        for p in points[1:]:
            cand_set &= set(neighbors[p])
            if not cand_set:
                break
        return cand_set

    # Local Search Loop
    while time.time() - start_time < time_limit:
        # 1. Reduction Pass
        old_size = len(C)
        C, counts = reduce_code(C, counts)
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()
            
        if time.time() - start_time > time_limit:
            break
            
        # 2. Swap Pass
        if len(C) == old_size:
            # No reduction happened. Try to find a move.
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                # Get unique points for c
                u_mask = counts[neighbors_np[c]] == 1
                unique_points = neighbors_np[c][u_mask]
                
                if len(unique_points) == 0:
                    # Should have been removed, but maybe logic missed it?
                    # Force removal
                    C.remove(c)
                    counts[neighbors_np[c]] -= 1
                    improved = True
                    break
                
                # If too many unique points, unlikely to find a single replacement
                # unless we allow larger swaps, but we stick to 1-for-1.
                # Limit to 2 or 3 to keep search space small
                if len(unique_points) > 2:
                    continue
                
                # Find candidates c' that cover all unique_points
                candidates_set = find_candidates(unique_points.tolist())
                
                if not candidates_set:
                    continue
                
                # Filter candidates: must not be in C
                # To make this fast, we can check membership.
                # C is a list, so 'in C' is O(N). 
                # We can maintain a set of C for faster lookup?
                # Let's just filter.
                c_set = set(C)
                candidates_set -= c_set
                
                if not candidates_set:
                    continue
                    
                # Pick a candidate
                # Heuristic: Pick one that maximizes overlap with other codewords?
                # Or just pick one randomly to explore.
                c_prime = random.choice(list(candidates_set))
                
                # Apply swap
                old_C = C.copy()
                old_counts = counts.copy()
                
                # Remove c, add c_prime
                C.remove(c)
                C.append(c_prime)
                
                counts[neighbors_np[c]] -= 1
                counts[neighbors_np[c_prime]] += 1
                
                # Check validity (should be valid by construction of U_c)
                # But let's be safe
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
            
            if not improved:
                # No move found in this pass.
                # To escape local minima, we can try a "kick":
                # Randomly swap a word with a random neighbor, even if it doesn't enable reduction immediately.
                for _ in range(10):
                    if time.time() - start_time > time_limit: break
                    if not C: break
                    c_idx = random.randint(0, len(C)-1)
                    c = C[c_idx]
                    # Pick a random neighbor of c
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand)-1)]
                    if c_prime in C: continue
                    
                    old_C = C.copy()
                    old_counts = counts.copy()
                    
                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors_np[c]] -= 1
                    counts[neighbors_np[c_prime]] += 1
                    
                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue
                        
                    # Try reduce
                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = counts.copy()
                        break
                    else:
                        # Accept the kick to change state
                        C = new_C
                        counts = new_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))
    
    return [list(all_words[i]) for i in best_C]
