import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    # Tetracode: perfect 1-error-correcting code of length 4 over GF(3) with 9 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Direct sum construction: 9 * 3^2 = 81 words for q=3, n=6
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tail)
    
    m = len(code)  # 81
    total_words = q ** n  # 729
    
    # Precompute coverage: for each codeword, which words it covers
    # A codeword covers itself and all words at distance <= radius
    # For q=3, n=6, radius=1: each covers 1 + 6*2 = 13 words
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    # Build covers: list of sets of word indices covered by each codeword
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
        covers.append(c_set)
    
    # Precompute coverage counts for the full 81-word code
    full_coverage = [0] * total_words
    for ci in range(m):
        for wi in covers[ci]:
            full_coverage[wi] += 1
    
    best_code = code[:]
    best_count = m
    
    # Helper: try to reduce a code by removing redundant codewords
    def reduce_code(code_indices, covers_list):
        """Remove redundant codewords from code_indices using covers_list."""
        if not code_indices:
            return code_indices
        
        # Build coverage count for this subset
        cover_count = [0] * total_words
        for ci in code_indices:
            for wi in covers_list[ci]:
                cover_count[wi] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for ci in code_indices:
                essential = False
                for wi in covers_list[ci]:
                    if cover_count[wi] == 1:
                        essential = True
                        break
                if not essential:
                    for wi in covers_list[ci]:
                        cover_count[wi] -= 1
                    changed = True
                else:
                    new_indices.append(ci)
            code_indices = new_indices
        return code_indices
    
    # Phase 1: Multiple random-order greedy reductions
    max_restarts = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Track active codewords
        active = [True] * m
        curr_coverage = full_coverage[:]
        
        # Single pass greedy removal in random order
        for ci in order:
            if not active[ci]:
                continue
            
            # Check if ci is redundant: every word it covers has coverage > 1
            is_redundant = True
            for wi in covers[ci]:
                if curr_coverage[wi] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                active[ci] = False
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
        
        current_code_indices = [i for i in range(m) if active[i]]
        current_code = [code[i] for i in current_code_indices]
        
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
        
        # Early stop if we hit the theoretical lower bound
        if best_count <= 71:
            break
    
    # Phase 2: Local search with swaps on the best solution
    if time.time() < deadline - 3.0:
        # Work with the best_code found
        m_best = len(best_code)
        
        # Build covers for best_code
        best_covers = []
        for c in best_code:
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
            best_covers.append(c_set)
        
        # Build coverage for best_code
        best_coverage = [0] * total_words
        for ci in range(m_best):
            for wi in best_covers[ci]:
                best_coverage[wi] += 1
        
        # Try removals and swaps
        swap_trials = 0
        max_swap_trials = 2000
        
        while swap_trials < max_swap_trials and time.time() < deadline - 2.0:
            swap_trials += 1
            
            # Try removing a random codeword
            ri = random.randint(0, m_best - 1)
            temp_coverage = best_coverage[:]
            for wi in best_covers[ri]:
                temp_coverage[wi] -= 1
            
            # Find words that are now uncovered (coverage 0)
            uncovered = [wi for wi in range(total_words) if temp_coverage[wi] == 0]
            
            if len(uncovered) == 0:
                # We can remove codeword ri without losing coverage
                new_code = [best_code[i] for i in range(m_best) if i != ri]
                new_covers = [best_covers[i] for i in range(m_best) if i != ri]
                if len(new_code) < best_count:
                    best_count = len(new_code)
                    best_code = new_code
                    best_covers = new_covers
                    m_best = len(best_code)
                    best_coverage = [0] * total_words
                    for ci in range(m_best):
                        for wi in best_covers[ci]:
                            best_coverage[wi] += 1
                continue
            
            # If there are uncovered words, try to find a replacement
            if len(uncovered) <= 8:
                uncovered_set = set(uncovered)
                # Try all codewords from the original 81-set
                for ci in range(m):
                    if code[ci] in best_code:
                        continue
                    # Check if code[ci] covers all uncovered words
                    if uncovered_set.issubset(covers[ci]):
                        # Perform swap: remove ri, add ci
                        new_code = [best_code[i] for i in range(m_best) if i != ri]
                        new_code.append(code[ci])
                        new_covers = [best_covers[i] for i in range(m_best) if i != ri]
                        new_covers.append(covers[ci])
                        
                        # Verify coverage is still complete
                        new_coverage = [0] * total_words
                        for cci in range(len(new_code)):
                            for wi in new_covers[cci]:
                                new_coverage[wi] += 1
                        
                        if all(new_coverage[wi] > 0 for wi in range(total_words)):
                            # Check if this enables further removals
                            reduced = reduce_code(list(range(len(new_code))), new_covers)
                            if len(reduced) < best_count:
                                final_code = [new_code[i] for i in reduced]
                                best_count = len(final_code)
                                best_code = final_code
                                # Rebuild covers and coverage for new best
                                best_covers = []
                                for c in best_code:
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
                                    best_covers.append(c_set)
                                m_best = len(best_code)
                                best_coverage = [0] * total_words
                                for ci in range(m_best):
                                    for wi in best_covers[ci]:
                                        best_coverage[wi] += 1
                                break
    
    return [list(c) for c in best_code]
