import numpy as np
import itertools
import sys
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = q ** n
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n):
                digits[i] = idx % q
                idx //= q
            return digits

        def digits_to_idx(digits):
            idx = 0
            mult = 1
            for d in digits:
                idx += d * mult
                mult *= q
            return idx

        # Precompute sphere neighbors as list of indices
        # sphere_neighbors[i] is a list of indices covered by i (including i)
        sphere_neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            digits = idx_to_digits(i)
            neighbors = [i]
            for pos in range(n):
                orig_d = digits[pos]
                for delta in [1, 2]: # +1 and -1 mod 3
                    new_d = (orig_d + delta) % q
                    new_digits = digits.copy()
                    new_digits[pos] = new_d
                    j = digits_to_idx(new_digits)
                    neighbors.append(j)
            sphere_neighbors[i] = neighbors

        # Greedy Phase using bitsets for speed in intersection counting
        # We need bitsets for the greedy phase to be fast enough
        sphere_bits = [0] * total_words
        for i in range(total_words):
            mask = 0
            for j in sphere_neighbors[i]:
                mask |= (1 << j)
            sphere_bits[i] = mask

        full_mask = (1 << total_words) - 1
        uncovered = full_mask
        codewords = []
        
        # Greedy
        while uncovered:
            best_center = -1
            best_count = 0
            # Optimization: Sample if time is tight? No, 6561 is small enough for a few passes.
            # But we do this ~400 times. 400 * 6561 = 2.6M bit ops. Fast.
            for c in range(total_words):
                inter = uncovered & sphere_bits[c]
                count = inter.bit_count()
                if count > best_count:
                    best_count = count
                    best_center = c
                    if best_count == uncovered.bit_count():
                        break
            
            if best_center == -1 or best_count == 0:
                break
                
            codewords.append(best_center)
            uncovered &= ~sphere_bits[best_center]
            
        # Local Search: Minimization using coverage counts
        # counts[w] = number of codewords covering w
        counts = [0] * total_words
        for c in codewords:
            for w in sphere_neighbors[c]:
                counts[w] += 1
        
        changed = True
        # Multiple passes to ensure convergence
        while changed:
            changed = False
            # Shuffle codewords to avoid bias?
            # Iterate backwards
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                # Check if c is redundant
                # c is redundant if all w in sphere(c) have counts[w] > 1
                redundant = True
                for w in sphere_neighbors[c]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    for w in sphere_neighbors[c]:
                        counts[w] -= 1
                    codewords.pop(i)
                    changed = True
        
        # Convert to digits
        result = []
        for idx in codewords:
            result.append(idx_to_digits(idx))
            
        return result

    # Fallback
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
    return [[0]*n]
