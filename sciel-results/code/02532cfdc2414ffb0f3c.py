import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Leave a safety margin of 5 seconds for cleanup and return
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    # Precompute neighbors
    # For q=3, n=8, radius=1: n_neighbors = 1 + 8 * 2 = 17
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    # Compute powers for index calculation
    q_powers = [q ** i for i in range(n)]
    
    # Build neighbor table
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            # Get digit at pos (0 is least significant in standard math, 
            # but our index calculation uses q_powers[n-1-pos] which maps 
            # pos 0 to the most significant digit if we follow the 
            # standard left-to-right string interpretation.
            # Let's verify the index mapping in the original code:
            # digit = (w // q_powers[n - 1 - pos]) % q
            # This treats pos 0 as the most significant digit (index n-1 in power).
            # So pos 0 -> q^(n-1), pos n-1 -> q^0.
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Phase 1: Greedy Set Cover
    covered = np.zeros(total_words, dtype=bool)
    gain = np.full(total_words, n_neighbors, dtype=np.int32)
    codewords = []
    
    remaining = total_words
    
    while remaining > 0:
        if time.time() > deadline - 25.0:
            break
            
        best_word = int(np.argmax(gain))
        best_gain = gain[best_word]
        
        if best_gain == 0:
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = int(uncovered[0])
        
        codewords.append(best_word)
        
        # Update covered and gain
        for v in neighbors[best_word]:
            if not covered[v]:
                covered[v] = True
                remaining -= 1
                for w in neighbors[v]:
                    gain[w] -= 1

    # Phase 2: Local Search
    cw = codewords[:]
    best_cw = cw[:]
    best_size = len(cw)
    
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    in_code = np.zeros(total_words, dtype=bool)
    
    for ci in cw:
        in_code[ci] = True
        for u in neighbors[ci]:
            count[u] += 1
    
    # Local search loop
    # We alternate between removing redundant codewords and performing swaps.
    # We use a random shuffle to avoid getting stuck in local minima caused by order.
    
    while time.time() < deadline - 5.0:
        improved = False
        
        # Pass 1: Remove redundant codewords
        # A codeword c is redundant if all v in N(c) have count[v] > 1
        indices = list(range(len(cw)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(cw):
                continue
            ci = cw[i]
            
            # Check redundancy: min count in neighborhood > 1
            if np.min(count[neighbors[ci]]) > 1:
                # Remove ci
                for u in neighbors[ci]:
                    count[u] -= 1
                in_code[ci] = False
                cw.pop(i)
                improved = True
                
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
        
        # Pass 2: Swaps
        if not improved:
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Find critical neighbors (covered only by ci)
                # critical = [u for u in neighbors[ci] if count[u] == 1]
                critical_mask = count[neighbors[ci]] == 1
                critical = neighbors[ci][critical_mask]
                
                if len(critical) == 0:
                    # Should have been removed in Pass 1, but maybe counts changed?
                    # If no critical neighbors, it's redundant.
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    break # Restart pass after a change
                
                if len(critical) > 0:
                    # Find candidates c' that cover all critical neighbors
                    # c' must be in intersection of N(u) for all u in critical
                    
                    # Start with neighbors of the first critical word
                    first_crit = critical[0]
                    # Filter out ci and words already in code
                    # We can do this by checking in_code array
                    cand_list = neighbors[first_crit]
                    
                    # Optimization: If len(critical) == 1, any valid neighbor not in code is a candidate
                    if len(critical) == 1:
                        # Find valid candidates
                        valid_cands = []
                        for cand in cand_list:
                            if cand == ci or in_code[cand]:
                                continue
                            # Since it's in N(first_crit) and first_crit is the only critical one,
                            # and cand != ci, it must cover first_crit.
                            # Does it cover others? There are no other criticals.
                            # So it's a valid swap.
                            valid_cands.append(cand)
                        
                        if valid_cands:
                            cand = random.choice(valid_cands)
                            # Perform swap
                            for u in neighbors[ci]:
                                count[u] -= 1
                            for u in neighbors[cand]:
                                count[u] += 1
                            
                            in_code[ci] = False
                            in_code[cand] = True
                            cw[i] = cand
                            improved = True
                            break # Restart pass after a swap
                    else:
                        # Multiple critical neighbors
                        # Filter cand_list by checking if it's in N(u) for all other u in critical
                        valid_cands = []
                        # To speed up, we can pre-check against the second critical word
                        # Let's just iterate and check all
                        for cand in cand_list:
                            if cand == ci or in_code[cand]:
                                continue
                            
                            # Check if cand covers all other critical neighbors
                            valid = True
                            for u in critical[1:]:
                                # Check if u is in neighbors[cand]
                                # np.any(neighbors[cand] == u)
                                # This is O(n_neighbors)
                                if not np.any(neighbors[cand] == u):
                                    valid = False
                                    break
                            
                            if valid:
                                valid_cands.append(cand)
                        
                        if valid_cands:
                            cand = random.choice(valid_cands)
                            # Perform swap
                            for u in neighbors[ci]:
                                count[u] -= 1
                            for u in neighbors[cand]:
                                count[u] += 1
                            
                            in_code[ci] = False
                            in_code[cand] = True
                            cw[i] = cand
                            improved = True
                            break # Restart pass after a swap
    
    # Convert back to list of lists
    result = []
    for ci in best_cw:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result
