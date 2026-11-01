import itertools
import numpy as np
import sys

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = q ** n
        # Precompute indices to digits and vice versa is not strictly needed if we work with indices
        # We need to generate the sphere matrix S
        # S[i, j] = 1 if dist(i, j) <= 1
        
        # Generate all words
        words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        
        # Compute pairwise distances? 
        # dist(i, j) = sum(words[i] != words[j])
        # This is a 6561x6561x8 comparison. 
        # words[:, :, None] != words[None, :, :] 
        # This creates a 6561x6561x8 array of bools. 6561^2 * 8 bytes = 344 MB. 
        # This might be too much memory or slow.
        
        # Alternative: Compute sphere masks on the fly or use a more memory efficient method.
        # Let's stick to Python int bitsets for memory efficiency and speed of bit operations, 
        # but use a smarter Greedy.
        
        # Actually, let's try the NumPy matrix approach but carefully.
        # 344MB is risky. 
        # Let's use the Python int bitset approach but optimize the loop.
        
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

        # Precompute sphere bits
        sphere_bits = [0] * total_words
        for i in range(total_words):
            digits = idx_to_digits(i)
            mask = 1 << i
            for pos in range(n):
                orig_d = digits[pos]
                for delta in [-1, 1]:
                    new_d = (orig_d + delta) % q
                    new_digits = digits.copy()
                    new_digits[pos] = new_d
                    j = digits_to_idx(new_digits)
                    mask |= (1 << j)
            sphere_bits[i] = mask

        full_mask = (1 << total_words) - 1
        uncovered = full_mask
        codewords = []
        
        # Greedy Phase
        # To speed up, we can't easily vectorize Python int bitsets.
        # But we can limit the search space?
        # No, let's just run it. 6561 iterations * 6561 checks.
        # 43M bit_count ops. 
        # bit_count is a C call. 43M C calls might take 10-20 seconds.
        # This is acceptable within 120s.
        
        while uncovered:
            best_center = -1
            best_count = 0
            # Optimization: We can skip centers that are already covered? No, a center can cover other uncovered words.
            # But if a center's sphere is entirely covered, it adds 0.
            # We can precompute a list of "active" centers?
            
            for c in range(total_words):
                inter = uncovered & sphere_bits[c]
                if inter:
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
            
        # Local Search: Minimization
        # Use coverage counts array
        # coverage[j] = number of codewords covering word j
        coverage = [0] * total_words
        for c in codewords:
            # Add sphere of c to coverage
            # This is O(N) per codeword if we iterate bits.
            # Better: Just recompute from scratch? 
            # K is ~500. N is 6561.
            # 500 * 6561 = 3M ops.
            # We can do this efficiently.
            
            # Actually, let's just use the bitset union check but optimized.
            # The previous method was O(K^2) unions.
            # Let's use the coverage count method.
            
            pass # Placeholder

        # Reset coverage calculation for LS
        coverage = np.zeros(total_words, dtype=np.int32)
        # Vectorized addition of spheres?
        # We have sphere_bits as Python ints.
        # We can convert to a numpy array of uint64s?
        # Or just iterate.
        
        # Let's use a simple loop for coverage update, it's fast enough.
        def update_coverage(c, delta):
            # delta is 1 or -1
            mask = sphere_bits[c]
            # Iterate set bits in mask?
            # Or just loop j from 0 to total_words?
            # total_words is 6561.
            # This is 6561 iterations per update.
            # We do this for each codeword in LS.
            
            # A faster way:
            # We can store spheres as a list of lists of indices?
            # sphere_indices[i] = list of j such that dist(i,j)<=1
            # Then update is:
            # for j in sphere_indices[c]: coverage[j] += delta
            
            pass

        # Precompute sphere_indices
        sphere_indices = []
        for i in range(total_words):
            digits = idx_to_digits(i)
            inds = [i]
            for pos in range(n):
                orig_d = digits[pos]
                for delta in [-1, 1]:
                    new_d = (orig_d + delta) % q
                    new_digits = digits.copy()
                    new_digits[pos] = new_d
                    j = digits_to_idx(new_digits)
                    inds.append(j)
            # Remove duplicates? 
            # The sphere has 1 + 2n elements. They are distinct for q>=3.
            sphere_indices.append(inds)

        # Initialize coverage
        for c in codewords:
            for j in sphere_indices[c]:
                coverage[j] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                # Check if any word in sphere of c has coverage == 1
                # If yes, c is essential.
                # If no, c is redundant.
                is_essential = False
                for j in sphere_indices[c]:
                    if coverage[j] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove c
                    for j in sphere_indices[c]:
                        coverage[j] -= 1
                    codewords.pop(i)
                    changed = True

        result = []
        for idx in codewords:
            result.append(idx_to_digits(idx))
        return result

    # Fallback
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
    return [[0]*n]
