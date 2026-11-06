import math
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def time_left():
        return deadline - time.time()

    # Primes 1 mod 4
    # We need primes such that their product with exponents stays within 10^18 / Q
    # Q will be a square of a primorial.
    # Let's fix Q to be small to allow large M, or optimize Q too?
    # The current program tries multiple Q. Let's stick to a good Q or a few.
    # Q = (2*3*5*7*11*13*17*19)^2 is large.
    # Let's try a few Q values.
    
    primes_all = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # Primes 1 mod 4 for M
    # Generate primes up to ~1000
    def get_primes_1mod4(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i in range(2, limit + 1) if sieve[i] and i % 4 == 1]

    p_1mod4 = get_primes_1mod4(2000)
    
    best_N = 0
    best_pairs = []
    
    # Try a few different Q bases
    # Q = (prod first k primes)^2
    # We want M to be large to have many representations, but M must be composed of primes 1 mod 4 (mostly)
    # Actually, if M has a prime factor p=3 mod 4 with odd exponent, r_2(M)=0.
    # So M must be of the form m1 * m2^2 where m1 is squarefree and all prime factors of m1 are 1 mod 4.
    # To maximize representations, we want M to have many divisors, specifically many divisors congruent to 1 mod 4.
    # The number of representations is related to the number of divisors.
    
    # Strategy:
    # 1. Pick a Q.
    # 2. Find M <= 10^18/Q such that M is "rich" in representations.
    #    M should be a product of primes 1 mod 4.
    #    Ideally M is a powerful number itself? No, M just needs to be an integer.
    #    Wait, a and b must be powerful.
    #    a = Q*x^2, b = Q*y^2.
    #    Q is powerful. x^2 is a square. So a is powerful.
    #    So we just need to find many pairs (x,y) such that x^2 + y^2 = M.
    #    This is equivalent to finding many representations of M as sum of two squares.
    
    # Let's optimize M to maximize r_2(M).
    # r_2(M) = 4 * prod (e_i + 1) if all p_i = 1 mod 4.
    # If M has p=3 mod 4, it must be squared.
    # To maximize r_2, we should use only primes 1 mod 4.
    
    # We will perform a random search for M.
    # For each candidate M, we estimate the number of pairs.
    # Then we pick the best M and enumerate pairs for it using NumPy.
    
    # To be safe, we'll try a few Q values.
    # Small Q allows larger M.
    # Q = 1? Then a=x^2, b=y^2. x^2+y^2=N.
    # If Q=1, N=M.
    # Let's try Q=1, Q=4, Q=36, etc.
    
    q_candidates = []
    # Q=1
    q_candidates.append(1)
    # Q = (2*3)^2 = 36
    q_candidates.append(36)
    # Q = (2*3*5)^2 = 900
    q_candidates.append(900)
    # Q = (2*3*5*7)^2 = 44100
    q_candidates.append(44100)
    
    # For each Q, find best M
    # We have ~140s total.
    # Let's allocate time per Q.
    
    for Q in q_candidates:
        if time_left() < 10:
            break
            
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Search for M
        # We want to maximize prod (e_i + 1) subject to prod p_i^e_i <= M_max
        # This is a variant of the knapsack problem or integer optimization.
        # Randomized hill climbing:
        
        best_M = 1
        best_score = 0
        
        # Use primes 1 mod 4
        # Start with a few small primes
        search_primes = p_1mod4[:50]
        
        # Initialize with greedy
        # Sort primes
        # Assign exponents
        
        # Let's do random restarts
        num_restarts = 100
        time_per_restart = 1.0 # seconds
        
        for r in range(num_restarts):
            if time_left() < 5:
                break
                
            # Random exponents
            # Start with all 0
            exps = [0] * len(search_primes)
            M = 1
            score = 1
            
            # Greedy fill
            # Randomize order of primes?
            # Or just random exponents
            
            # Let's try a random walk
            # Start with a random valid M
            # Generate random M:
            # Pick random primes and random exponents
            # Ensure product <= M_max
            
            # Simpler: Start with 1, multiply by random primes^random_exp
            # If it exceeds M_max, stop or backtrack?
            # Backtracking is hard.
            
            # Alternative:
            # Generate a random vector of exponents for the first K primes.
            # K=20.
            # Exponents from 0 to 10.
            # Check if product <= M_max.
            
            K = 20
            p_subset = search_primes[:K]
            
            # Randomize
            for _ in range(100): # inner iterations per restart
                if time_left() < 5:
                    break
                exps = [random.randint(0, 10) for _ in range(K)]
                M_val = 1
                score_val = 1
                valid = True
                for i, e in enumerate(exps):
                    if e > 0:
                        M_val *= p_subset[i] ** e
                        score_val *= (e + 1)
                        if M_val > M_max:
                            valid = False
                            break
                if valid and score_val > best_score:
                    best_score = score_val
                    best_M = M_val
            
            # Local search on best_M found so far?
            # Try to increase exponents of small primes if possible
            # Or swap exponents
            
        # Now we have a candidate best_M
        # But maybe a different M has more representations?
        # r_2(M) is proportional to score_val.
        # So maximizing score_val is correct for primes 1 mod 4.
        
        M = best_M
        N = Q * M
        
        if N < 2 or N > 10**18:
            continue
        
        # Enumerate pairs for this M
        # x^2 + y^2 = M
        # x <= y
        # x^2 <= M/2
        # x_max = isqrt(M//2)
        
        # Use NumPy to speed up
        # We can't check all x up to 10^9 with NumPy in one go (memory).
        # But we can chunk it.
        
        limit_x = int(math.isqrt(M // 2)) + 1
        
        # If limit_x is too large, we might not be able to enumerate all.
        # But r_2(M) is small compared to M.
        # The number of solutions is ~ O(M^epsilon).
        # So most x will not work.
        # Brute force x from 1 to limit_x is O(sqrt(M)).
        # If M ~ 10^15, sqrt(M) ~ 3*10^7.
        # Python loop: 3*10^7 iterations ~ 10-30s. Too slow.
        # NumPy vectorized:
        # We can process 10^6 x values at a time.
        # 3*10^7 / 10^6 = 30 chunks.
        # Each chunk: 10^6 operations. Very fast.
        
        pairs = []
        M_val = M
        Q_val = Q
        
        # Chunk size
        chunk_size = 10**6
        
        # We need to find x such that M - x^2 is a perfect square.
        # y^2 = M - x^2
        # y = isqrt(M - x^2)
        # Check y^2 == M - x^2
        
        # We only care about x <= y => x^2 <= M/2 => x <= sqrt(M/2)
        
        # Iterate in chunks
        for start_x in range(1, limit_x + 1, chunk_size):
            if time_left() < 2:
                break
                
            end_x = min(start_x + chunk_size, limit_x + 1)
            xs = np.arange(start_x, end_x, dtype=np.int64)
            
            # x^2
            x2 = xs * xs
            y2 = M_val - x2
            
            # y2 must be >= 0
            valid_mask = y2 >= 0
            
            if not np.any(valid_mask):
                continue
            
            y2_valid = y2[valid_mask]
            xs_valid = xs[valid_mask]
            
            # isqrt
            # numpy doesn't have isqrt for int64 that returns int64 exactly?
            # np.sqrt returns float.
            # For 10^18, float64 has 52 bits of precision.
            # 10^18 ~ 2^60. So float64 cannot represent all integers exactly.
            # We need to be careful.
            # y = isqrt(y2)
            # We can compute y_float = np.sqrt(y2_valid)
            # Then check (y_float-1)^2, y_float^2, (y_float+1)^2
            
            y_float = np.sqrt(y2_valid)
            y_int = y_float.astype(np.int64)
            
            # Check y_int^2 == y2_valid
            # Also (y_int-1)^2 and (y_int+1)^2 due to rounding
            
            # y_int^2
            y2_calc = y_int * y_int
            mask1 = (y2_calc == y2_valid)
            
            # (y_int-1)^2
            y_int_minus = y_int - 1
            y2_calc_minus = y_int_minus * y_int_minus
            mask2 = (y2_calc_minus == y2_valid)
            
            # (y_int+1)^2
            y_int_plus = y_int + 1
            y2_calc_plus = y_int_plus * y_int_plus
            mask3 = (y2_calc_plus == y2_valid)
            
            # Combine masks
            # We need to find which xs correspond to valid y
            
            # This is getting complex with masks.
            # Let's simplify:
            # For each x in the chunk, check if M-x^2 is a square.
            # We can do this in a loop if the chunk is small, but chunk is 10^6.
            # Looping 10^6 times in Python is ~0.1s.
            # 30 chunks * 0.1s = 3s. Acceptable.
            
            # Let's just loop over the valid ones?
            # No, we need to check all.
            
            # Alternative: Use a set of squares?
            # No, M is large.
            
            # Let's stick to the vectorized check but handle the rounding carefully.
            # Actually, for x^2 + y^2 = M, if M is fixed, there are very few solutions.
            # The probability that a random x works is ~ 0.
            # So we can just check the candidates from y_float.
            
            # Let's try a different approach for enumeration:
            # Iterate x, check if M-x^2 is a square.
            # To speed up, we can use the fact that y^2 = M - x^2.
            # y = isqrt(M - x^2).
            
            # Let's use the loop but with a time check.
            # If limit_x is too large, we might time out.
            # But we only do this for the BEST M.
            
            # Let's assume the NumPy approach is faster.
            # Let's refine the NumPy check.
            
            # We have xs_valid, y2_valid.
            # We want to find indices where y2_valid is a perfect square.
            
            # y = isqrt(y2)
            # We can compute y = np.sqrt(y2).astype(int)
            # Then check y*y == y2
            # This works if y2 is not too close to the boundary of float precision.
            # M <= 10^18.
            # x^2 <= M/2.
            # y^2 <= M.
            # sqrt(10^18) = 10^9.
            # 10^9 is exactly representable in float64.
            # But y2 is up to 10^18.
            # float64 has 15-16 decimal digits.
            # 10^18 has 19 digits.
            # So y2 might not be exactly representable.
            # However, y2 is an integer.
            # If y2 is a perfect square, say k^2.
            # If k <= 10^9, k^2 <= 10^18.
            # Is k^2 exactly representable?
            # Not always.
            
            # Safer:
            # y_float = np.sqrt(y2_valid)
            # y_int = y_float.astype(np.int64)
            # Check (y_int)^2, (y_int-1)^2, (y_int+1)^2
            
            # Let's do this in a loop over the chunk?
            # No, vectorize the checks.
            
            # y_int = np.floor(y_float + 0.5) # rounding
            # But sqrt is not linear.
            
            # Let's just use the loop for the final enumeration if NumPy is tricky.
            # The loop is O(sqrt(M)).
            # If M ~ 10^15, sqrt(M) ~ 3*10^7.
            # 3*10^7 iterations in Python is ~ 10s.
            # We have ~100s left after search.
            # So we can afford to loop.
            
            # Let's switch to a simple loop for reliability.
            # We'll break if time is low.
            
            # Actually, we can mix:
            # If limit_x < 10^7, use loop.
            # If limit_x >= 10^7, use NumPy chunks.
            
            pass # Placeholder
        
        # Let's implement the loop for enumeration
        # We'll do it outside the Q loop, for the best M found across all Qs?
        # Or for each Q?
        # Let's find the best (Q, M) pair first, then enumerate.
        
        # Revised Plan:
        # 1. Search for best M for each Q.
        # 2. Keep the (Q, M) with highest estimated r_2(M).
        # 3. Enumerate pairs for the best (Q, M).
        
        # We need to store the best candidate.
        
        # Let's restructure.
        
    # End of Q loop (restructured)
    
    # Let's implement the search for best (Q, M)
    best_candidate_Q = 1
    best_candidate_M = 1
    best_candidate_score = 0
    
    for Q in q_candidates:
        if time_left() < 10:
            break
            
        M_max = 10**18 // Q
        if M_max < 2:
            continue
        
        # Search for M
        best_M = 1
        best_score = 0
        
        search_primes = p_1mod4[:50]
        
        # Random search
        num_restarts = 50
        for r in range(num_restarts):
            if time_left() < 5:
                break
                
            K = 20
            p_subset = search_primes[:K]
            
            for _ in range(100):
                if time_left() < 5:
                    break
                exps = [random.randint(0, 10) for _ in range(K)]
                M_val = 1
                score_val = 1
                valid = True
                for i, e in enumerate(exps):
                    if e > 0:
                        M_val *= p_subset[i] ** e
                        score_val *= (e + 1)
                        if M_val > M_max:
                            valid = False
                            break
                if valid and score_val > best_score:
                    best_score = score_val
                    best_M = M_val
        
        if best_score > best_candidate_score:
            best_candidate_score = best_score
            best_candidate_M = best_M
            best_candidate_Q = Q
            
    # Now enumerate for best_candidate_Q, best_candidate_M
    Q = best_candidate_Q
    M = best_candidate_M
    N = Q * M
    
    if N < 2 or N > 10**18:
        # Fallback
        return {'N': 5, 'pairs': [[1, 4]]}
        
    limit_x = int(math.isqrt(M // 2)) + 1
    
    pairs = []
    M_val = M
    Q_val = Q
    isqrt = math.isqrt
    
    # Use a loop
    # To speed up, we can check x in steps? No, x must be integer.
    # We can skip x where M-x^2 is not a square mod something?
    # Mod 4: squares are 0, 1.
    # M - x^2 must be 0 or 1 mod 4.
    # If M is 0 mod 4, x^2 must be 0 or 1 mod 4.
    # If M is 1 mod 4, x^2 must be 0 or 1 mod 4.
    # This doesn't help much.
    
    # Let's just loop.
    # If limit_x is too large, we might not finish.
    # But we have ~100s.
    # 10^8 iterations ~ 10s in PyPy, ~30s in CPython.
    # If limit_x > 3*10^7, it might be tight.
    
    for x in range(1, limit_x):
        if time_left() < 2:
            break
            
        x2 = x * x
        y2 = M_val - x2
        y = isqrt(y2)
        if y * y == y2 and x <= y:
            a = Q_val * x2
            b = Q_val * y2
            pairs.append([a, b])
            
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    return {'N': N, 'pairs': pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
