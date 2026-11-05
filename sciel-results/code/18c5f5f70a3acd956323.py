import itertools
import random

def construct(q, n, radius):
    # Handle general cases or fallbacks
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n
    # Map integer index to list of digits
    # We work with integer indices for speed
    # index = d0 * 3^6 + d1 * 3^5 + ... + d6
    
    # Precompute neighbors
    # neighbors[i] is a list of integer indices
    neighbors = [[] for _ in range(total_words)]
    
    # Powers of 3
    powers = [1]
    for i in range(n):
        powers.append(powers[-1] * 3)
        
    # Precompute digit values for each index? No, just compute on fly or precompute table.
    # Precomputing digits for 2187 words is cheap.
    digits = [[0]*n for _ in range(total_words)]
    for i in range(total_words):
        val = i
        for j in range(n):
            digits[i][j] = val % 3
            val //= 3

    for i in range(total_words):
        neighbors[i].append(i)
        d = digits[i]
        for pos in range(n):
            orig_val = d[pos]
            # The step to change digit at pos by +1 or +2 (mod 3)
            # delta = (new_val - orig_val) * 3^(n-1-pos)
            # Actually, simpler:
            # new_val 0,1,2
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Calculate new index
                # diff = (new_val - orig_val)
                # idx_diff = diff * 3^(n-1-pos)
                # But mod 3 wrap around means we can't just add if diff is negative or >1?
                # No, integer arithmetic handles it if we do:
                # new_idx = i + (new_val - orig_val) * powers[n-1-pos]
                # Wait, if orig_val=2, new_val=0, diff=-2.
                # i - 2 * 3^k.
                # Is this correct?
                # Example: 02 (base 3) = 2. pos 1 (3s place). orig 0. new 2.
                # i=2. powers[1]=3.
                # new_idx = 2 + (2-0)*3 = 8.
                # 8 in base 3 is 22. Correct.
                # Example: 20 (base 3) = 6. pos 1. orig 2. new 0.
                # i=6. powers[1]=3.
                # new_idx = 6 + (0-2)*3 = 0.
                # 0 in base 3 is 00. Correct.
                
                new_idx = i + (new_val - orig_val) * powers[n-1-pos]
                neighbors[i].append(new_idx)

    # Greedy Set Cover with Random Restarts
    best_code = None
    best_size = float('inf')
    
    # Run greedy a few times with random tie-breaking
    for _ in range(5):
        uncovered = [1] * total_words
        remaining = total_words
        code = []
        
        while remaining > 0:
            best_idx = -1
            max_gain = 0
            # Add small random noise to break ties to explore different local optima
            # Or just pick first max? Random restarts handle the diversity.
            for i in range(total_words):
                # If this word doesn't cover any uncovered, skip?
                # Gain calculation
                gain = 0
                nbs = neighbors[i]
                for nb in nbs:
                    if uncovered[nb]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if remaining > 0
                
            code.append(best_idx)
            
            # Update uncovered
            for nb in neighbors[best_idx]:
                if uncovered[nb]:
                    uncovered[nb] = 0
                    remaining -= 1
                    
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Pruning Phase
    # Use coverage counts
    cover_count = [0] * total_words
    for idx in best_code:
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    # Greedy pruning
    # We iterate through the code.
    # To make it faster, we can shuffle the code order?
    # Or just iterate.
    i = 0
    while i < len(best_code):
        idx = best_code[i]
        # Tentatively remove
        all_covered = True
        for nb in neighbors[idx]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                all_covered = False
                break
        
        if all_covered:
            # Keep it removed
            best_code.pop(i)
            # Do not increment i
        else:
            # Restore
            for nb in neighbors[idx]:
                cover_count[nb] += 1
            i += 1

    # Convert indices back to lists of digits
    result = [digits[idx][:] for idx in best_code]
    return result
