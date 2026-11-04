import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 19 for q=3, n=6

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

    # Precompute word indices
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword as sorted lists for faster iteration
    covers = []
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
        covers.append(sorted(c_set))
    
    # Precompute coverage_count as numpy array
    coverage_count = np.zeros(total_words, dtype=np.int32)
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1

    best_code = code[:]
    best_count = m

    # Helper: check if removing ci is safe (all words it covers have coverage > 1)
    def is_redundant(ci, curr_coverage):
        for wi in covers[ci]:
            if curr_coverage[wi] <= 1:
                return False
        return True

    # Helper: check if swapping (remove ai, add ri) is safe
    def is_safe_swap(ai, ri, curr_coverage):
        # Words only covered by ai must also be covered by ri
        for wi in covers[ai]:
            if curr_coverage[wi] == 1 and wi not in covers[ri]:
                return False
        return True

    # Precompute for each codeword, the set of codewords it covers (for fast swap checks)
    # Actually, let's precompute a boolean matrix: covers[ci][wi] is not easily indexable
    # Instead, let's precompute for each ci, a set of wi it covers (already have as list)
    # For fast "wi in covers[ri]" checks, convert to sets
    covers_set = [set(lst) for lst in covers]

    def is_safe_swap_fast(ai, ri, curr_coverage):
        for wi in covers[ai]:
            if curr_coverage[wi] == 1 and wi not in covers_set[ri]:
                return False
        return True

    max_restarts = 20000  # Will break based on time

    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break

        # Reset coverage counts
        curr_coverage = coverage_count.copy()
        active = [True] * m
        active_set = set(range(m))
        removed_set = set()

        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)

        # Greedy removal pass
        for ci in order:
            if not active[ci]:
                continue
            if is_redundant(ci, curr_coverage):
                active[ci] = False
                active_set.discard(ci)
                removed_set.add(ci)
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1

        # Swap-based local search
        improved = True
        while improved:
            improved = False
            removed_list = list(removed_set)
            random.shuffle(removed_list)
            active_list = list(active_set)
            random.shuffle(active_list)

            for ri in removed_list:
                if ri not in removed_set:
                    continue
                for ai in active_list:
                    if ai not in active_set:
                        continue
                    if is_safe_swap_fast(ai, ri, curr_coverage):
                        # Perform swap
                        active[ai] = False
                        active[ri] = True
                        active_set.discard(ai)
                        active_set.add(ri)
                        removed_set.discard(ri)
                        removed_set.add(ai)
                        for wi in covers[ai]:
                            curr_coverage[wi] -= 1
                        for wi in covers[ri]:
                            curr_coverage[wi] += 1
                        improved = True
                        break
                if improved:
                    break

        current_count = len(active_set)
        if current_count < best_count:
            best_count = current_count
            best_code = [code[i] for i in active_set]

    return [list(c) for c in best_code]
