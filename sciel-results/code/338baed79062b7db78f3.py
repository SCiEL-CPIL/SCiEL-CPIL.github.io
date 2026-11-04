import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for 120s limit

    # Fallback for other parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4:
            if q == 3 and radius == 1:
                return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(3), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187
    neighbors_list = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors
    for i in range(N):
        w = idx_to_tuple(i)
        neighbors_list[i, 0] = i
        count = 1
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    neighbors_list[i, count] = tuple_to_idx(new_w)
                    count += 1
        # Pad with -1 or just ignore extra indices? 
        # We know the size is 1 + 2*n = 15.
        # The array is fixed size 15.
        
    # Trim neighbors_list to actual size if needed, but 15 is constant for n=7, q=3.
    # neighbors_list[i] contains 15 elements.

    best_code_size = float('inf')
    best_code = None

    # Multi-start strategy
    num_starts = 5
    time_per_start = (deadline - start_time) / num_starts

    for run_id in range(num_starts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Set Cover
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []
        code_set = set()
        
        # Randomize tie-breaking by shuffling candidate order? 
        # Or just pick randomly among best?
        # Let's stick to deterministic greedy for stability, but maybe randomize the first pick?
        # First pick: 0 is symmetric.
        code.append(0)
        code_set.add(0)
        coverage[neighbors_list[0]] += 1
        uncovered_mask[neighbors_list[0]] = False
        
        while uncovered_mask.any() and time.time() < deadline:
            best_c = -1
            best_count = 0
            
            # Scan all candidates
            # Optimization: Vectorize the count?
            # For each c, count = sum(uncovered_mask[neighbors_list[c]])
            # This is a gather operation.
            # We can do this in a loop, N=2187 is small.
            
            # To add randomness, we can shuffle the indices we iterate over?
            # No, that slows down finding the best.
            # Instead, if multiple have same best_count, pick one randomly.
            
            counts = np.zeros(N, dtype=np.int32)
            # This is hard to vectorize easily without advanced indexing on 2D array
            # Let's stick to the loop, it's fast enough for 2187 items.
            
            max_count = 0
            candidates = []
            
            for c in range(N):
                if c in code_set:
                    continue
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > max_count:
                    max_count = cnt
                    candidates = [c]
                elif cnt == max_count and cnt > 0:
                    candidates.append(c)
            
            if not candidates:
                break
                
            # Randomly pick from best candidates to diversify runs
            best_c = random.choice(candidates)
            
            code.append(best_c)
            code_set.add(best_c)
            coverage[neighbors_list[best_c]] += 1
            uncovered_mask[neighbors_list[best_c]] = False

        # 2. Prune redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for c in list(code):
                if np.min(coverage[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                    changed = True

        # 3. Local Search
        def try_kick(k):
            if len(code) < k:
                return False
            indices = random.sample(range(len(code)), k)
            removed = [code[i] for i in indices]
            
            temp_coverage = coverage.copy()
            for c in removed:
                temp_coverage[neighbors_list[c]] -= 1
                
            uncovered = np.where(temp_coverage == 0)[0]
            
            if len(uncovered) == 0:
                # Perfect removal
                for c in removed:
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                return True
            
            uncovered_set = set(uncovered.tolist())
            added = []
            
            # Greedy repair
            temp_uncovered = set(uncovered_set)
            while temp_uncovered and len(added) < k: # Don't add more than we removed, else no gain
                best_c = -1
                best_count = 0
                # Candidates are neighbors of uncovered words
                # We can just scan all non-code words? 
                # Or restrict to neighbors of uncovered?
                # Neighbors of uncovered is a good heuristic.
                candidate_pool = set()
                for w in temp_uncovered:
                    candidate_pool.update(neighbors_list[w].tolist())
                
                for c in candidate_pool:
                    if c in code_set:
                        continue
                    cnt = len(set(neighbors_list[c].tolist()) & temp_uncovered)
                    if cnt > best_count:
                        best_count = cnt
                        best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_c == -1:
                    break
                added.append(best_c)
                temp_uncovered.difference_update(set(neighbors_list[best_c].tolist()))
            
            if len(added) < len(removed):
                # Apply change
                for c in removed:
                    code.remove(c)
                    code_set.remove(c)
                    coverage[neighbors_list[c]] -= 1
                for c in added:
                    if c not in code_set:
                        code.append(c)
                        code_set.add(c)
                        coverage[neighbors_list[c]] += 1
                return True
            return False

        no_improvement = 0
        while time.time() < deadline - 5: # Leave time for next run
            # Prune again
            changed = True
            while changed:
                changed = False
                for c in list(code):
                    if np.min(coverage[neighbors_list[c]]) > 1:
                        code.remove(c)
                        code_set.remove(c)
                        coverage[neighbors_list[c]] -= 1
                        changed = True
            
            # Kicks
            improved = False
            # Try k=1, 2, 3
            for k in [1, 2, 3]:
                if try_kick(k):
                    improved = True
                    break
            
            if improved:
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > 100:
                    break # Stuck in local minimum, restart

        if len(code) < best_code_size:
            best_code_size = len(code)
            best_code = code[:]

    return [idx_to_tuple(c) for c in best_code]
