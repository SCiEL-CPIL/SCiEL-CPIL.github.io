import math
from functools import lru_cache

def construct():
    # Primes p == 1 (mod 4)
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113]
    
    log_limit = math.log(10**18)
    
    best_score = 0
    best_exponents = []
    
    def search(idx, remaining_log, current_exponents, current_score):
        nonlocal best_score, best_exponents
        
        # Prune: if current_score * (max possible multiplier) <= best_score, return
        # Max possible multiplier is roughly bounded, but we can just proceed for small depth
        if idx >= len(primes_1_4):
            if current_score > best_score:
                best_score = current_score
                best_exponents = current_exponents[:]
            return
        
        p = primes_1_4[idx]
        log_p = math.log(p)
        
        # Max exponent we can afford
        max_e = int(remaining_log / log_p)
        
        # Try exponents from max_e down to 0
        # Note: We should prefer higher exponents on smaller primes, 
        # but the search explores all.
        for e in range(max_e, -1, -1):
            new_log = remaining_log - e * log_p
            if new_log < 0:
                continue
            new_score = current_score * (e + 1)
            
            # Optimization: If even taking max exponents for remaining primes 
            # doesn't beat best_score, skip.
            # Upper bound on remaining score: 
            # This is hard to estimate tightly, so we skip this optimization 
            # as the search space is small (depth ~10, branching ~20).
            
            search(idx + 1, new_log, current_exponents + [e], new_score)
            
    search(0, log_limit, [], 1)
    
    # Reconstruct M
    M = 1
    for i, e in enumerate(best_exponents):
        if e > 0:
            M *= primes_1_4[i] ** e
            
    # The number of pairs (u, v) with u <= v, u^2 + v^2 = M
    # Total solutions (x, y) integer is 4 * prod(e_i + 1)
    # We want u, v > 0 and u <= v.
    # If M is not a square and not 2*square, all solutions have x != y, x,y != 0.
    # Number of pairs = Total / 8 = (4 * prod(e_i+1)) / 8 = prod(e_i+1) / 2.
    # We need to list them.
    
    # Function to find a, b such that p = a^2 + b^2
    def find_sum_of_squares_prime(p):
        a = 1
        while a * a < p:
            b2 = p - a * a
            b = int(math.isqrt(b2))
            if b * b == b2:
                return a, b
            a += 1
        return 1, 1 # Should not happen for p in list
        
    # Generate all representations of M as sum of two squares
    # We use the composition law.
    # Start with (1, 0)
    # For each prime power p^e, generate all representations of p^e.
    # Then combine.
    
    # Actually, a simpler way for small number of solutions:
    # We know the count is small.
    # We can iterate u from 1 to sqrt(M/2)?
    # sqrt(10^18) = 10^9. Too slow.
    
    # We must use the factorization-based generation.
    
    # Representations of p^e:
    # If p = a^2 + b^2, then p^e has e+1 representations (up to sign/order).
    # Specifically, (a + ib)^k * (a - ib)^(e-k) for k=0..e?
    # No, the Gaussian integer factorization is unique.
    # p = (a+ib)(a-ib).
    # p^e = (a+ib)^e (a-ib)^e.
    # Any divisor of p^e in Z[i] is (a+ib)^k (a-ib)^(e-k) for k=0..e.
    # Let Z_k = (a+ib)^k (a-ib)^(e-k).
    # Then Z_k * conjugate(Z_k) = p^e.
    # The real and imaginary parts of Z_k give a representation.
    # Note: Z_k and Z_{e-k} are conjugates (up to unit).
    # So we only need k=0..e.
    
    # For M = prod p_i^e_i:
    # Z = prod Z_{k_i}
    # The number of such Z is prod (e_i + 1).
    # Each Z gives a solution (Re(Z), Im(Z)).
    # We need to handle signs and order.
    
    reps = [(1, 0)]
    
    for i, e in enumerate(best_exponents):
        if e == 0:
            continue
        p = primes_1_4[i]
        a, b = find_sum_of_squares_prime(p)
        
        # Generate all Z_k for this prime power
        prime_powers = []
        # Calculate (a+ib)^k * (a-ib)^(e-k)
        # We can do this iteratively.
        # Start with Z = 1.
        # For k from 0 to e:
        #   Z_k = Z_{k-1} * (a-ib) / (a+ib) ? No.
        #   Z_k = (a+ib)^k (a-ib)^(e-k)
        #   Z_0 = (a-ib)^e
        #   Z_1 = (a+ib) (a-ib)^(e-1)
        #   ...
        
        # Let's just compute powers of (a+ib) and (a-ib).
        pos_powers = [(1, 0)]
        neg_powers = [(1, 0)]
        
        # (a+ib)^j
        for j in range(1, e + 1):
            ra, ia = pos_powers[-1]
            # (ra + i ia) * (a + i b)
            new_ra = ra * a - ia * b
            new_ia = ra * b + ia * a
            pos_powers.append((new_ra, new_ia))
            
            # (ra - i ia) * (a - i b) for negative powers?
            # Actually (a-ib)^j is conjugate of (a+ib)^j
            # So we can just conjugate pos_powers.
            
        # We need Z_k = (a+ib)^k (a-ib)^(e-k)
        # Let P_k = (a+ib)^k
        # Let N_m = (a-ib)^m
        # Z_k = P_k * N_{e-k}
        
        new_reps = []
        for k in range(e + 1):
            pk = pos_powers[k]
            # N_{e-k} is conjugate of pos_powers[e-k]
            ne_k = (pos_powers[e-k][0], -pos_powers[e-k][1])
            
            # Multiply pk * ne_k
            r1, i1 = pk
            r2, i2 = ne_k
            zr = r1 * r2 - i1 * i2
            zi = r1 * i2 + i1 * r2
            
            prime_powers.append((zr, zi))
            
        # Combine with existing reps
        temp_reps = []
        for r1, i1 in reps:
            for r2, i2 in prime_powers:
                # (r1 + i i1) * (r2 + i i2)
                new_r = r1 * r2 - i1 * i2
                new_i = r1 * i2 + i1 * r2
                temp_reps.append((new_r, new_i))
        reps = temp_reps
        
    # Now reps contains all Gaussian integers Z such that |Z|^2 = M.
    # Each (r, i) in reps corresponds to a solution r^2 + i^2 = M.
    # We need to filter for r, i > 0 and r <= i.
    
    pairs = []
    seen = set()
    for r, i in reps:
        # Handle signs
        # The set of solutions generated includes all combinations of signs?
        # No, the generation Z = prod (a+ib)^k (a-ib)^(e-k) generates specific Gaussian integers.
        # Multiplying by units 1, i, -1, -i generates all 4 rotations.
        # But our construction might not cover all 4 directly.
        # Actually, for each Z, -Z, iZ, -iZ are also solutions.
        # And conjugates.
        # To be safe, we can just check all 8 variations for each (r, i).
        
        candidates = [
            (r, i), (-r, i), (r, -i), (-r, -i),
            (i, r), (-i, r), (i, -r), (-i, -r)
        ]
        
        for cr, ci in candidates:
            if cr <= 0 or ci <= 0:
                continue
            if cr > ci:
                continue
            a_val = cr * cr
            b_val = ci * ci
            if a_val + b_val == M:
                pair = (a_val, b_val)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append([a_val, b_val])
                    
    # Sort pairs for consistency
    pairs.sort()
    
    return {'N': M, 'pairs': pairs}

# EVOLVE-BLOCK-END
