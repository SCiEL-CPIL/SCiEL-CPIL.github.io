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
    
    # Phase 1: Standard greedy removal to get a good initial solution
    max_restarts_phase1 = 200
    for restart in range(max_restarts_phase1):
        if time.time() > deadline - 60.0:
            break
            
        # Reset coverage counts
        curr_coverage = coverage_count[:]
        active = [True] * m
        
        # Random order for removal attempts
        order = list(range(m))
        random.shuffle(order)
        
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
            
    # Phase 2: Swap-based local search with simulated annealing
    # Start from the best solution found in Phase 1
    if best_count < m:
        # Initialize active set from best_code
        best_set = set(best_code)
        active = [c in best_set for c in code]
        
        # Rebuild coverage counts for current active set
        curr_coverage = [0] * total_words
        for ci in range(m):
            if active[ci]:
                for wi in covers[ci]:
                    curr_coverage[wi] += 1
        
        # Simulated annealing parameters
        T_start = 0.5
        T_end = 0.01
        cooling_rate = 0.995
        T = T_start
        
        # Precompute for faster swap evaluation
        # For a swap (remove ci, add cj), we need to check:
        # 1. All words covered by ci (but not cj) still have coverage > 1 after removal
        # 2. All words that were uncovered or uniquely covered by ci are now covered by cj
        
        while time.time() < deadline - 5.0:
            # Random swap: pick two active codewords to swap
            # Actually, we want to try removing one and adding another
            active_indices = [i for i in range(m) if active[i]]
            
            if len(active_indices) < 2:
                break
            
            # Pick a random active codeword to remove
            ci = random.choice(active_indices)
            
            # Pick a random inactive codeword to add
            inactive_indices = [i for i in range(m) if not active[i]]
            if not inactive_indices:
                break
            cj = random.choice(inactive_indices)
            
            # Evaluate the swap
            # After removing ci and adding cj:
            # - Words in covers[ci] \ covers[cj] lose one coverage
            # - Words in covers[cj] \ covers[ci] gain one coverage
            # - Words in covers[ci] ∩ covers[cj] unchanged
            
            # Check validity:
            # 1. For all wi in covers[ci] \ covers[cj], curr_coverage[wi] - 1 >= 1 (i.e., curr_coverage[wi] >= 2)
            # 2. No new uncovered words are created (adding cj only helps)
            
            valid_swap = True
            for wi in covers[ci]:
                if wi not in covers[cj]:
                    if curr_coverage[wi] <= 1:
                        valid_swap = False
                        break
            
            if valid_swap:
                # Perform the swap
                active[ci] = False
                active[cj] = True
                for wi in covers[ci]:
                    curr_coverage[wi] -= 1
                for wi in covers[cj]:
                    curr_coverage[wi] += 1
                
                # Accept the swap (we're doing hill-climbing with random restarts implicitly)
                # Actually, for simulated annealing, we should sometimes accept worse solutions
                # But since the objective is just to minimize count, and swaps keep count same,
                # we always accept valid swaps as they might lead to better removal opportunities
                
                # Try to remove more codewords after the swap
                removed_any = True
                while removed_any:
                    removed_any = False
                    for idx in range(m):
                        if not active[idx]:
                            continue
                        
                        is_redundant = True
                        for wi in covers[idx]:
                            if curr_coverage[wi] <= 1:
                                is_redundant = False
                                break
                        
                        if is_redundant:
                            active[idx] = False
                            for wi in covers[idx]:
                                curr_coverage[wi] -= 1
                            removed_any = True
                            break
                    
                    if not removed_any:
                        break
                
                # Check if we improved
                current_count = sum(active)
                if current_count < best_count:
                    best_count = current_count
                    best_code = [code[i] for i in range(m) if active[i]]
                    # Restart temperature if we found a new best
                    T = T_start
            else:
                # Invalid swap, skip
                pass
            
            # Cool down
            T *= cooling_rate
    
    # Convert to lists for output
    return [list(c) for c in best_code]
