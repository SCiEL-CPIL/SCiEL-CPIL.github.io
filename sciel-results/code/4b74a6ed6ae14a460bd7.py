# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    # Precompute neighbors
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    # Compute powers for index calculation
    q_powers = [q ** i for i in range(n)]
    
    # Build neighbor table
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            # Get digit at pos
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                # Calculate new index
                # Replace digit at pos with v
                # w = ... + digit * q^(n-1-pos) + ...
                # new_w = w - digit * q^(n-1-pos) + v * q^(n-1-pos)
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Phase 1: Greedy Set Cover with dynamic gain updates
    covered = np.zeros(total_words, dtype=bool)
    gain = np.full(total_words, n_neighbors, dtype=np.int32)
    codewords = []
    
    remaining = total_words
    
    while remaining > 0:
        if time.time() > deadline - 20.0:
            break
            
        best_word = np.argmax(gain)
        best_gain = gain[best_word]
        
        if best_gain == 0:
            # Should not happen if remaining > 0
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = uncovered[0]
        
        codewords.append(best_word)
        
        # Update covered and gain
        for v in neighbors[best_word]:
            if not covered[v]:
                covered[v] = True
                remaining -= 1
                # Decrement gain for all words that cover v
                # These are the neighbors of v
                for w in neighbors[v]:
                    gain[w] -= 1

    # Phase 2: Local Search
    cw = codewords[:]
    best_cw = cw[:]
    best_size = len(cw)
    
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw:
        for u in neighbors[ci]:
            count[u] += 1
    
    # Helper to check if a word is in the current code
    # Using a set for O(1) lookup is faster for small n
    cw_set = set(cw)
    
    # Local search loop
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
            
            # Check redundancy
            if np.all(count[neighbors[ci]] > 1):
                # Remove ci
                for u in neighbors[ci]:
                    count[u] -= 1
                cw.pop(i)
                cw_set.remove(ci)
                improved = True
                
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
        
        # Pass 2: Swaps
        if not improved:
            # Try to swap codewords to enable further removals
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Find critical neighbors (covered only by ci)
                critical = []
                for u in neighbors[ci]:
                    if count[u] == 1:
                        critical.append(u)
                
                if not critical:
                    # Redundant, should have been caught in Pass 1
                    # But maybe counts changed?
                    continue
                
                # Find candidates c' that cover all critical neighbors
                # c' must be in intersection of N(u) for all u in critical
                # Since |critical| is small, we can just iterate through neighbors of the first critical word
                # and check if it covers the others
                
                candidates = []
                first_crit = critical[0]
                for cand in neighbors[first_crit]:
                    if cand == ci:
                        continue
                    # Check if cand covers all other critical words
                    valid = True
                    for u in critical[1:]:
                        # Check if u is in neighbors[cand]
                        # This is O(n_neighbors) per check
                        if not np.any(neighbors[cand] == u):
                            valid = False
                            break
                    if valid:
                        candidates.append(cand)
                
                if candidates:
                    # Pick a random candidate
                    cand = random.choice(candidates)
                    
                    # Perform swap
                    # Remove ci
                    for u in neighbors[ci]:
                        count[u] -= 1
                    # Add cand
                    for u in neighbors[cand]:
                        count[u] += 1
                    
                    cw[i] = cand
                    cw_set.remove(ci)
                    cw_set.add(cand)
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

# EVOLVE-BLOCK-END
