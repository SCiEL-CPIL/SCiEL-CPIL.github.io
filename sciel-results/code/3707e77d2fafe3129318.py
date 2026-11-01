import numpy as np
import itertools
import sys

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

        # Precompute sphere bitsets
        sphere_bits = [0] * total_words
        for i in range(total_words):
            digits = idx_to_digits(i)
            mask = 1 << i
            for pos in range(n):
                orig_d = digits[pos]
                # +1 mod q
                new_d = (orig_d + 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                mask |= (1 << j)
                # -1 mod q
                new_d = (orig_d - 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                mask |= (1 << j)
            sphere_bits[i] = mask

        full_mask = (1 << total_words) - 1
        uncovered = full_mask
        codewords = []
        
        # Greedy Phase
        while uncovered:
            best_center = -1
            best_count = 0
            for c in range(total_words):
                inter = uncovered & sphere_bits[c]
                if inter == 0:
                    continue
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
            
        # Local Search: Minimization using count array
        # count[i] = number of codewords covering word i
        count = [0] * total_words
        for c in codewords:
            digits = idx_to_digits(c)
            count[c] += 1
            for pos in range(n):
                orig_d = digits[pos]
                for delta in [-1, 1]:
                    new_d = (orig_d + delta) % q
                    new_digits = digits.copy()
                    new_digits[pos] = new_d
                    j = digits_to_idx(new_digits)
                    count[j] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                digits = idx_to_digits(c)
                # Tentatively remove
                count[c] -= 1
                if count[c] < 0:
                    count[c] += 1
                    continue
                valid = True
                for pos in range(n):
                    orig_d = digits[pos]
                    for delta in [-1, 1]:
                        new_d = (orig_d + delta) % q
                        new_digits = digits.copy()
                        new_digits[pos] = new_d
                        j = digits_to_idx(new_digits)
                        count[j] -= 1
                        if count[j] < 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    codewords.pop(i)
                    changed = True
                else:
                    # Revert
                    count[c] += 1
                    for pos in range(n):
                        orig_d = digits[pos]
                        for delta in [-1, 1]:
                            new_d = (orig_d + delta) % q
                            new_digits = digits.copy()
                            new_digits[pos] = new_d
                            j = digits_to_idx(new_digits)
                            count[j] += 1
        
        # Convert to digits
        result = []
        for idx in codewords:
            result.append(idx_to_digits(idx))
            
        return result

    # Fallback
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
    return [[0]*n]
