import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # This task is specifically for q=3, n=6, radius=1
    # We use the tetracode direct-sum construction as a starting point
    # Tetracode: perfect 1-error-correcting code of length 4 over GF(3) with 9 words
    
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    
    # Construct initial code: direct sum with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tail)
            
    m = len(code)  # 81 for q=3, n=6
    total_words = q ** n  # 729
    
    # Precompute coverage sets for each codeword
    # A codeword covers itself and all words at distance 1
    # For q=3, n=6: each codeword covers 1 + 6*2 = 13 words
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
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
    
    # Initial coverage count
    coverage_count = [0] * total_words
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1
    
    best_code = code[:]
    best_count = m
    
    # Greedy removal with multiple random restarts
    # We can also try swapping: remove one codeword and add a different one if it helps
    
    max_restarts = 5000  # We'll break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3.0:
            break
        
        # Reset state
        curr_coverage = coverage_count[:]
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Single pass greedy removal
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
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
        
        # If we've reached the known best, we could stop, but let's try to beat it
        # Best known is 73, lower bound is 71
        if best_count <= 73:
            # Continue searching for a better solution
            pass
    
    # Try a local search / swap improvement on the best solution found
    if time.time() < deadline - 5.0:
        # Work with the best_code found
        # Try to remove one codeword and add a different one from the original set
        # Or try to find a smaller subset by more aggressive search
        
        # Convert best_code to a set for fast lookup
        best_set = set(best_code)
        m_best = len(best_code)
        
        # Recompute coverage for best_code
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
        
        best_coverage = [0] * total_words
        for ci in range(m_best):
            for wi in best_covers[ci]:
                best_coverage[wi] += 1
        
        # Try removing one codeword from best_code and see if we can add a different one
        # to maintain coverage while keeping the same size or reducing it
        for trial in range(200):
            if time.time() > deadline - 2.0:
                break
            
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
                if len(new_code) < best_count:
                    best_count = len(new_code)
                    best_code = new_code
                    best_set = set(best_code)
                    # Recompute covers and coverage
                    best_covers = best_covers[:ri] + best_covers[ri+1:]
                    m_best = len(best_code)
                    best_coverage = [0] * total_words
                    for ci in range(m_best):
                        for wi in best_covers[ci]:
                            best_coverage[wi] += 1
                continue
            
            # If there are uncovered words, try to find a codeword from the original set
            # that covers all uncovered words
            if len(uncovered) <= 5:  # Only try if few words are uncovered
                uncovered_set = set(uncovered)
                for ci in range(m):
                    if code[ci] in best_set:
                        continue
                    # Check if code[ci] covers all uncovered words
                    c_set = covers[ci]
                    if uncovered_set.issubset(c_set):
                        # We can replace ri with ci
                        # Check if this doesn't create issues (it shouldn't since we're just swapping)
                        new_code = [best_code[i] for i in range(m_best) if i != ri]
                        new_code.append(code[ci])
                        # This maintains the same size, but might help future removals
                        # For now, we only care about reducing size, so skip same-size swaps
                        pass
    
    # Additional local search: try pairwise swaps to escape local minima
    if time.time() < deadline - 3.0:
        best_set = set(best_code)
        m_best = len(best_code)
        
        # Recompute coverage for current best_code
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
        
        best_coverage = [0] * total_words
        for ci in range(m_best):
            for wi in best_covers[ci]:
                best_coverage[wi] += 1
        
        # Try pairwise swaps: remove codeword i, add codeword j (not in best_set)
        # This can help escape local minima where no single removal works
        swap_attempts = 0
        max_swap_attempts = 5000
        while swap_attempts < max_swap_attempts:
            if time.time() > deadline - 2.0:
                break
            
            swap_attempts += 1
            
            # Pick a random codeword to remove
            ri = random.randint(0, m_best - 1)
            
            # Compute coverage after removing ri
            temp_coverage = best_coverage[:]
            for wi in best_covers[ri]:
                temp_coverage[wi] -= 1
            
            # Find words that become uncovered
            uncovered = [wi for wi in range(total_words) if temp_coverage[wi] == 0]
            
            if len(uncovered) == 0:
                # Can remove ri entirely - this should have been caught earlier, but just in case
                new_code = [best_code[i] for i in range(m_best) if i != ri]
                if len(new_code) < best_count:
                    best_count = len(new_code)
                    best_code = new_code
                    best_set = set(best_code)
                    best_covers = best_covers[:ri] + best_covers[ri+1:]
                    m_best = len(best_code)
                    best_coverage = [0] * total_words
                    for ci in range(m_best):
                        for wi in best_covers[ci]:
                            best_coverage[wi] += 1
                continue
            
            # If few words are uncovered, try to find a codeword that covers them all
            if len(uncovered) <= 8:
                uncovered_set = set(uncovered)
                for ci in range(m):
                    if code[ci] in best_set:
                        continue
                    c_set = covers[ci]
                    if uncovered_set.issubset(c_set):
                        # Perform the swap
                        new_code = [best_code[i] for i in range(m_best) if i != ri]
                        new_code.append(code[ci])
                        best_code = new_code
                        best_set = set(best_code)
                        best_covers = best_covers[:ri] + best_covers[ri+1:] + [c_set]
                        m_best = len(best_code)
                        best_coverage = [0] * total_words
                        for ci2 in range(m_best):
                            for wi2 in best_covers[ci2]:
                                best_coverage[wi2] += 1
                        break
    
    return [list(c) for c in best_code]
