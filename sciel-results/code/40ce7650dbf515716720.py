import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build the direct-sum construction
    # Tetracode is a perfect 1-error-correcting code of length 4 over GF(3)
    # It has 9 words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))
            
    m = len(code) # 81 for q=3, n=6
    total_words = q ** n
    
    # Precompute word indices
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword as numpy arrays for vectorized ops
    covers = []
    cover_sets = []  # for O(1) membership tests
    for c in code:
        c_set = set()
        c_set.add(word_to_index(c))
        c_list = list(c)
        for pos in range(n):
            orig_val = c_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                c_list[pos] = new_val
                c_set.add(word_to_index(tuple(c_list)))
            c_list[pos] = orig_val
        covers.append(np.array(sorted(c_set), dtype=np.int32))
        cover_sets.append(c_set)
    
    # Precompute coverage_count as numpy array
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in range(m):
        coverage_count[covers[ci]] += 1

    best_code = code[:]
    best_count = m
    
    max_restarts = 20000  # Will break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
            
        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active = np.ones(m, dtype=bool)
        
        # Random order for removal attempts
        order = np.random.permutation(m)
        
        # Greedy removal pass
        for ci in order:
            if not active[ci]:
                continue
            
            # Check if ci is redundant: all words it covers have coverage > 1
            if np.all(curr_coverage[covers[ci]] > 1):
                active[ci] = False
                curr_coverage[covers[ci]] -= 1
        
        # Deterministic greedy set-cover: ensure validity by adding codewords
        # that cover the most uncovered words
        if time.time() < deadline - 5.0:
            uncovered = curr_coverage == 0
            if np.any(uncovered):
                # Precompute uncovered count for each codeword
                uncovered_counts = np.zeros(m, dtype=np.int32)
                for ci in range(m):
                    uncovered_counts[ci] = np.sum(uncovered[covers[ci]])
                
                while np.any(uncovered):
                    if time.time() > deadline - 5.0:
                        break
                    # Find codeword covering most uncovered words
                    best_ci = int(np.argmax(uncovered_counts))
                    if uncovered_counts[best_ci] == 0:
                        break
                    active[best_ci] = True
                    curr_coverage[covers[best_ci]] += 1
                    # Update uncovered
                    newly_covered = (curr_coverage == 1) & (uncovered)
                    uncovered[newly_covered] = False
                    # Recompute uncovered counts for active candidates
                    for ci in range(m):
                        if not active[ci]:
                            uncovered_counts[ci] = np.sum(uncovered[covers[ci]])
        
        # Add-then-remove heuristic: try adding a random inactive codeword,
        # then greedily remove redundant ones. This can escape local minima.
        inactive_indices = np.where(~active)[0]
        if len(inactive_indices) > 0:
            # Try several random additions
            num_tries = min(10, len(inactive_indices))
            for _ in range(num_tries):
                if time.time() > deadline - 3.0:
                    break
                ri = inactive_indices[random.randint(0, len(inactive_indices) - 1)]
                if active[ri]:
                    continue
                
                # Save state
                saved_coverage = curr_coverage.copy()
                saved_active = active.copy()
                
                # Add ri
                active[ri] = True
                curr_coverage[covers[ri]] += 1
                
                # Greedy removal pass
                order2 = np.random.permutation(m)
                for ci in order2:
                    if not active[ci]:
                        continue
                    if np.all(curr_coverage[covers[ci]] > 1):
                        active[ci] = False
                        curr_coverage[covers[ci]] -= 1
                
                if np.sum(active) < best_count:
                    best_count = int(np.sum(active))
                    best_code = [code[i] for i in range(m) if active[i]]
                
                # Restore state
                curr_coverage = saved_coverage
                active = saved_active
        
        # Swap-based local search: try to swap an active codeword with an inactive one
        # to enable further removals
        improved = True
        while improved:
            if time.time() > deadline - 3.0:
                break
            improved = False
            active_list = np.where(active)[0]
            inactive_list = np.where(~active)[0]
            
            if len(inactive_list) == 0:
                break
            
            # Randomly sample swaps
            num_swap_tries = min(20, len(active_list) * len(inactive_list))
            for _ in range(num_swap_tries):
                if time.time() > deadline - 3.0:
                    break
                ai = active_list[random.randint(0, len(active_list) - 1)]
                ri = inactive_list[random.randint(0, len(inactive_list) - 1)]
                
                # Check if swap is safe: words only covered by ai must be covered by ri
                ai_unique = covers[ai][curr_coverage[covers[ai]] == 1]
                if len(ai_unique) == 0:
                    # Safe swap
                    active[ai] = False
                    active[ri] = True
                    curr_coverage[covers[ai]] -= 1
                    curr_coverage[covers[ri]] += 1
                    improved = True
                else:
                    # Check if ri covers all of ai's unique words
                    ri_set = cover_sets[ri]
                    if all(wi in ri_set for wi in ai_unique):
                        active[ai] = False
                        active[ri] = True
                        curr_coverage[covers[ai]] -= 1
                        curr_coverage[covers[ri]] += 1
                        improved = True
            
            if not improved:
                # Try another round of greedy removal after swaps
                order3 = np.random.permutation(m)
                for ci in order3:
                    if not active[ci]:
                        continue
                    if np.all(curr_coverage[covers[ci]] > 1):
                        active[ci] = False
                        curr_coverage[covers[ci]] -= 1
                
                if np.sum(active) < best_count:
                    best_count = int(np.sum(active))
                    best_code = [code[i] for i in range(m) if active[i]]
                    improved = True
        
        if np.sum(active) < best_count:
            best_count = int(np.sum(active))
            best_code = [code[i] for i in range(m) if active[i]]

    # Convert to lists for output
    return [list(c) for c in best_code]
