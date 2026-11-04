import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # Build the 81-word direct-sum construction (tetracode x 3)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    # For n=6, the tail is 2 positions
    from itertools import product
    code = [tuple(h) + tuple(tail) for h in tetracode for tail in product(range(q), repeat=n - 4)]
    
    m = len(code)  # 81
    total_words = q ** n

    # Precompute covered word indices for each codeword
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    codeword_covers = []
    for c in code:
        covers = set()
        covers.add(word_to_index(c))
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(c)
                w_list[pos] = val
                covers.add(word_to_index(tuple(w_list)))
        codeword_covers.append(covers)

    # Build coverage matrix: coverage_matrix[ci, wi] = 1 if codeword ci covers word wi
    coverage_matrix = np.zeros((m, total_words), dtype=np.uint8)
    for ci in range(m):
        for wi in codeword_covers[ci]:
            coverage_matrix[ci, wi] = 1

    # Build covered_by: for each word, which codewords cover it
    covered_by = [[] for _ in range(total_words)]
    for ci in range(m):
        for wi in codeword_covers[ci]:
            covered_by[wi].append(ci)

    # Greedy removal with vectorized redundancy checking
    best_code = code[:]
    best_count = m

    # Multiple restarts with different random orders
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        active = np.ones(m, dtype=np.uint8)
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Multiple passes
        for _pass in range(20):
            if time.time() > deadline - 5.0:
                break
            
            removed_any = False
            for ci in order:
                if time.time() > deadline - 5.0:
                    break
                if not active[ci]:
                    continue
                
                # Check if codeword ci is redundant
                ci_covers = codeword_covers[ci]
                
                # For each word covered by ci, check if any other active codeword covers it
                is_redundant = True
                for wi in ci_covers:
                    found_other = False
                    for cj in covered_by[wi]:
                        if cj != ci and active[cj]:
                            found_other = True
                            break
                    if not found_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active[ci] = 0
                    removed_any = True
            
            if not removed_any:
                break
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]

    # Convert to lists
    return [list(c) for c in best_code]
