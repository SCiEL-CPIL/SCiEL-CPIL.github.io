import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def make_orientation(n, squares):
        return [0 if (j - i) % n in squares else 1 for i in range(n) for j in range(i + 1, n)]

    # Seed solution details (backup) - Paley 83
    seed_n = 83
    seed_squares = {x * x % seed_n for x in range(1, seed_n)}
    best_result = {'n': seed_n, 'orientation': make_orientation(seed_n, seed_squares)}

    # Candidate primes p ≡ 3 mod 4, increasing order.
    # We know 67 works. We check smaller ones to see if any work.
    # The theoretical lower bound is 31, so we check all primes ≡ 3 mod 4 in [31, 67].
    # Primes ≡ 3 mod 4: 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83
    candidates = [31, 43, 47, 59, 67]

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
        
        # Check property:
        # For every 4-subset S of vertices, exists v in 0..p-1 such that S is a subset of out[v].
        # i.e., (s_mask & ~out[v]) == 0 for some v.
        
        valid = True
        # Iterate over all 4-combinations
        # For p=31, C(31,4) = 31465. For p=67, C(67,4) = 736281.
        # The check inside is fast: bitwise AND.
        for combo in itertools.combinations(range(p), 4):
            s_mask = 0
            for v in combo:
                s_mask |= (1 << v)
            
            # Check if any vertex dominates all 4
            covered = False
            for v in range(p):
                if (s_mask & out[v]) == s_mask:
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
            # Since we iterate in increasing order, the first valid one is the smallest.
            # We can return immediately.
            return best_result
        
        if time.time() > deadline - 5.0:
            break

    return best_result
