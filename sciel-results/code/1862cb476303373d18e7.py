import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Build the 81-word direct-sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    code = [tuple(h) + tuple(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    total_words = q ** n
    m = len(code)  # 81
    
    # Precompute coverage: for each codeword, which words it covers
    # Use a 2D boolean array: coverage[ci][wi] = True if codeword ci covers word wi
    # But 81 * 729 = 59049, which is fine
    
    # Map word to index
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    # For each codeword, compute covered word indices
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
    
    # Build covered_by: for each word, which codewords cover it
    covered_by = [[] for _ in range(total_words)]
    for ci in range(m):
        for wi in codeword_covers[ci]:
            covered_by[wi].append(ci)
    
    # Greedy removal with randomization
    best_code = code[:]
    best_count = m
    
    # Multiple restarts with different random orders
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
        # Multiple passes
        for _pass in range(10):
            if time.time() > deadline - 10.0:
                break
            
            removed_any = False
            for ci in order:
                if time.time() > deadline - 10.0:
                    break
                if not active[ci]:
                    continue
                
                # Check if codeword ci is redundant
                is_redundant = True
                for wi in codeword_covers[ci]:
                    # Check if any other active codeword covers wi
                    found_other = False
                    for cj in covered_by[wi]:
                        if cj != ci and active[cj]:
                            found_other = True
                            break
                    if not found_other:
                        is_redundant = False
                        break
                
                if is_redundant:
                    active[ci] = False
                    removed_any = True
            
            if not removed_any:
                break
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
    
    # Convert to lists
    return [list(c) for c in best_code]
