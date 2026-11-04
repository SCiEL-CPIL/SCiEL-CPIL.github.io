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
    # Mapping from word tuple to integer index
    # Since q^n is small (729), we can use a dictionary or direct indexing
    # Direct indexing is faster: idx = w[0]*q^(n-1) + ...
    
    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute coverage sets for each codeword
    # covers[ci] = set of word indices covered by codeword ci
    # A codeword covers itself and all words at distance 1
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
        
    # Precompute covered_by for initial state? 
    # Actually, for the greedy removal with counters, we just need:
    # 1. covers[ci]
    # 2. coverage_count[wi] (initially number of codewords covering wi)
    
    coverage_count = [0] * total_words
    for ci in range(m):
        for wi in covers[ci]:
            coverage_count[wi] += 1

    best_code = code[:]
    best_count = m
    
    # Greedy removal strategy with randomization
    # We perform multiple restarts. In each restart, we try to remove redundant codewords.
    # A codeword is redundant if every word it covers is covered by at least one other active codeword.
    # i.e., coverage_count[wi] > 1 for all wi in covers[ci].
    
    max_restarts = 1000 # We will break based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Reset coverage counts
        curr_coverage = coverage_count[:]
        active = [True] * m
        
        # Random order for removal attempts
        # We want to check codewords in a random order to escape local minima
        order = list(range(m))
        random.shuffle(order)
        
        # We iterate through the order. If a codeword is redundant, we remove it.
        # Removing it updates curr_coverage.
        # This is a single pass. We might need multiple passes because removing one might make another redundant?
        # No, removing a codeword decreases coverage counts. It can only make other codewords *less* likely to be redundant (more critical).
        # So a single pass in a fixed order is a valid greedy step.
        # However, the *order* matters.
        
        # Optimization: 
        # Instead of checking all m codewords in each restart, we can be smarter.
        # But m=81 is small. 81 * 729 checks is ~60k ops. Very fast.
        # 1000 restarts * 60k = 60M ops. Should be well within 115s in Python if optimized.
        
        for ci in order:
            if not active[ci]:
                continue
            
            # Check if ci is redundant
            is_redundant = True
            for wi in covers[ci]:
                if curr_coverage[wi] <= 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove ci
                active[ci] = False
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
        
        current_code = [code[i] for i in range(m) if active[i]]
        if len(current_code) < best_count:
            best_count = len(current_code)
            best_code = current_code[:]
            
            # If we hit the theoretical lower bound or known best, we can stop early?
            # Best known is 73. Lower bound 71.
            if best_count <= 73:
                # We found a very good solution. 
                # We could try to find 72 or 71, but 73 is the standard "best known".
                # Let's continue a bit longer to see if we can beat 73, but prioritize speed.
                pass

    # Convert to lists for output
    return [list(c) for c in best_code]
