import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    total = q ** n
    
    def int_to_digits(x, n, q):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits
    
    def digits_to_int(digits, q):
        x = 0
        for d in digits:
            x = x * q + d
        return x
    
    # Precompute coverage sets
    cov = []
    for c in range(total):
        c_digits = int_to_digits(c, n, q)
        covered = {c}
        for pos in range(n):
            for val in range(q):
                if val == c_digits[pos]:
                    continue
                new_digits = c_digits[:]
                new_digits[pos] = val
                covered.add(digits_to_int(new_digits, q))
        cov.append(covered)
    
    # Greedy set cover
    uncovered = set(range(total))
    chosen = []
    chosen_set = set()
    
    while uncovered:
        if time.time() > deadline:
            break
        best_c = -1
        best_count = 0
        for c in range(total):
            if c in chosen_set:
                continue
            # Count how many uncovered words this codeword covers
            count = len(uncovered & cov[c])
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        chosen.append(best_c)
        chosen_set.add(best_c)
        uncovered -= cov[best_c]
    
    if uncovered:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    best_code = chosen[:]
    
    # Local search: repeatedly try to remove a codeword
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_code)):
            if time.time() > deadline:
                break
            c = best_code[i]
            # Compute coverage of all other codewords
            rem_cov = set()
            for j in range(len(best_code)):
                if j != i:
                    rem_cov |= cov[best_code[j]]
            if len(rem_cov) == total:
                best_code.pop(i)
                improved = True
                break
    
    return [int_to_digits(c, n, q) for c in best_code]
