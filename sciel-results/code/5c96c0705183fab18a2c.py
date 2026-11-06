"""
Program for the Schütte k=4 tournament task.
Strategy:
1. Iterate through primes p ≡ 3 mod 4 (7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79).
2. For each prime, generate the Paley tournament.
3. Efficiently check the k=4 property using bitmasks.
   Property: For every 4-subset S, there is a vertex v such that S is a subset of out[v].
4. Return the first valid tournament found. If none, return n=83 (Paley 83).
"""

import time
import itertools

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def make_orientation(n, squares):
        return [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]

    # Seed solution details (backup)
    seed_n = 83
    seed_squares = {x * x % seed_n for x in range(1, seed_n)}
    best_result = {'n': seed_n, 'orientation': make_orientation(seed_n, seed_squares)}

    # Candidate primes p ≡ 3 mod 4, increasing order.
    # We know 67 works. We check smaller ones to see if any work.
    candidates = [7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79]

    for p in candidates:
        if time.time() > deadline:
            break
        
        # Precompute squares
        squares_set = {x * x % p for x in range(1, p)}
        
        # Precompute out-neighborhoods as bitmasks
        out = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i != j:
                    diff = (j - i) % p
                    if diff in squares_set:
                        mask |= (1 << j)
            out[i] = mask
            
        # Precompute NOT out-neighborhoods (within the p-bit space)
        # We only care about bits 0..p-1
        full_mask = (1 << p) - 1
        not_out = [full_mask ^ out[i] for i in range(p)]
        
        # Check property:
        # For every 4-subset S, exists v such that S & not_out[v] == 0.
        # If for some S, (S & not_out[v]) != 0 for ALL v, then S is not dominated.
        
        valid = True
        # Optimization: If p is small, this is fast.
        # For p=67, C(67,4) ~ 700k. 700k * 67 checks ~ 47M ops. ~2-5s.
        
        for S in itertools.combinations(range(p), 4):
            s_mask = 0
            # Unroll loop for speed?
            s_mask = (1 << S[0]) | (1 << S[1]) | (1 << S[2]) | (1 << S[3])
            
            covered = False
            # Check if any v covers S
            for v in range(p):
                if (s_mask & not_out[v]) == 0:
                    covered = True
                    break
            
            if not covered:
                valid = False
                break

        if valid:
            best_result = {
                'n': p,
                'orientation': make_orientation(p, squares_set)
            }
            return best_result

    return best_result
