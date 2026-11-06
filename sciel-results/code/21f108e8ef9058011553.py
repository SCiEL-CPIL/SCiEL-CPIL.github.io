"""Program: find A, B (3 distinct ints in [1,128]) maximizing |A+B| with pairwise-coprime distinct sums."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = np.ones(n + 1, dtype=bool)
    if n >= 0:
        is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:n + 1:i] = False
    return is_prime

def construct():
    t_start = time.time()
    DEADLINE = 78  # safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _sieve(MAX_SUM)

    # Known-good fallback (9 distinct primes, pairwise coprime)
    # A = [1, 7, 13], B = [4, 10, 16] gives sums: 5,11,17, 11,17,23, 17,23,29
    # Distinct: {5,11,17,23,29} — only 5 distinct, not 9. Let me find a real one.
    # Actually, let's just search quickly.
    
    # Precompute a quick pairwise-coprime checker for up to 9 numbers
    def score_of(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) > 1:
                    return 0
        return n

    # Best known seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)

    # Search for 9 distinct pairwise-coprime sums (ideally all primes)
    # Parameterize: A = {a, a+d1, a+2*d1}, B = {b, b+d2, b+2*d2}
    # If gcd(d1,d2)==1, the 9 offsets i*d1+j*d2 are distinct, so 9 distinct sums.
    # Check if all 9 sums are prime.
    
    max_d = 63  # 2*d <= 127
    
    # Precompute offsets for each (d1,d2) pair lazily
    # Iterate d1, d2 in order of increasing d1+d2 to find solutions faster
    pairs = []
    for d1 in range(1, max_d + 1):
        for d2 in range(1, max_d + 1):
            if math.gcd(d1, d2) == 1:
                pairs.append((d1 + d2, d1, d2))
    pairs.sort()  # sort by d1+d2 ascending

    for _, d1, d2 in pairs:
        if time.time() - t_start > DEADLINE:
            break
        
        a_max = 128 - 2 * d1
        b_max = 128 - 2 * d2
        if a_max < 1 or b_max < 1:
            continue
        
        offsets = np.array([i * d1 + j * d2 for i in range(3) for j in range(3)])
        
        # For each a, find if any b makes all 9 sums prime
        for a in range(1, a_max + 1):
            if time.time() - t_start > DEADLINE:
                break
            
            # Vectorized check over all b values
            b_vals = np.arange(1, b_max + 1)
            sums = a + b_vals[np.newaxis, :] + offsets[:, np.newaxis]
            
            # All sums must be in [2, 256] and prime
            valid = np.all((sums >= 2) & (sums <= MAX_SUM) & is_prime[sums], axis=0)
            
            if np.any(valid):
                b_idx = np.argmax(valid)
                b = int(b_vals[b_idx])
                A = [a, a + d1, a + 2 * d1]
                B = [b, b + d2, b + 2 * d2]
                # Verify
                s = score_of(A, B)
                if s >= 9:
                    return [A, B]
                elif s > best_score:
                    best_score = s
                    best_A = A
                    best_B = B
    
    # Fallback: random search if time permits
    if time.time() - t_start < DEADLINE - 3:
        import random
        rng = random.Random(42)
        for _ in range(100000):
            if time.time() - t_start > DEADLINE:
                break
            A = rng.sample(range(1, 129), 3)
            B = rng.sample(range(1, 129), 3)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
                if s >= 9:
                    return [A, B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
