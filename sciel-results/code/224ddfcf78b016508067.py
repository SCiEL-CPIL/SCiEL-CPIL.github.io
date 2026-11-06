import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin, evaluator kills at 90s. 35s buffer for safety.
    
    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001
    
    # 1. Sieve for smallest prime factor (SPF)
    spf = list(range(N_LIMIT + 1))
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Precompute powerful parts using SPF and DP
    pp = [1] * (N_LIMIT + 1)
    rem = [0] * (N_LIMIT + 1)
    exp = [0] * (N_LIMIT + 1)
    
    pp[1] = 1
    rem[1] = 1
    exp[1] = 0
    
    for n in range(2, N_LIMIT + 1):
        p = spf[n]
        n_prime = n // p
        
        if n_prime % p == 0:
            exp[n] = exp[n_prime] + 1
            rem[n] = rem[n_prime]
        else:
            exp[n] = 1
            rem[n] = n_prime
            
        if exp[n] >= 2:
            pp[n] = pp[rem[n]] * (p ** exp[n])
        else:
            pp[n] = pp[rem[n]]
            
        if time.time() > deadline:
            break
    
    # 3. Find best n via Brute Force
    best_score = -1
    best_n = 10000
    
    for n in range(10000, N_MAX + 1):
        score = pp[n] * pp[n + 1]
        if score > best_score:
            best_score = score
            best_n = n
    
    # 4. CRT-based Construction Search
    # Generate prime powers
    prime_powers = set()
    for p in range(2, N_LIMIT + 1):
        if spf[p] == p:  # p is prime
            power = p
            while power <= N_LIMIT:
                prime_powers.add(power)
                power *= p
                
    # Sort descending to prioritize large products
    pp_list = sorted([x for x in prime_powers if x <= N_MAX], reverse=True)
    
    # Optimization: Only consider A, B such that A*B > best_score
    for i in range(len(pp_list)):
        A = pp_list[i]
        
        for j in range(i + 1, len(pp_list)):
            B = pp_list[j]
            prod = A * B
            
            # Since B is decreasing, if prod <= best_score, no further B will work
            if prod <= best_score:
                break
                
            # Solve n = A*k, n+1 = B*m
            # A*k - B*m = -1
            # k*A = 1 mod B
            # k = A^{-1} mod B
            try:
                k_inv = pow(A, -1, B)
            except ValueError:
                continue  # Should be coprime
                
            k = k_inv
            n0 = A * k
            step = A * B
            
            # Find t such that 10000 <= n0 + t*step <= 1000000
            if n0 < 10000:
                t_min = (10000 - n0 + step - 1) // step
            else:
                t_min = 0
                
            t_max = (1000000 - n0) // step
            
            if t_min <= t_max:
                # Usually only one or few solutions in range
                for t in range(t_min, t_max + 1):
                    n_cand = n0 + t * step
                    if 10000 <= n_cand <= N_MAX:
                        # Verify score
                        score_cand = pp[n_cand] * pp[n_cand + 1]
                        if score_cand > best_score:
                            best_score = score_cand
                            best_n = n_cand
                            
        if time.time() > deadline:
            break
            
    # 5. Extended search: consider pairs of prime powers that may not be coprime
    # but where the actual powerful parts are larger due to additional factors
    # We look for n where pp[n] and pp[n+1] are both large
    # Strategy: for each large prime power A, find n = A*k where pp[n] is large
    # and pp[n+1] is also large
    
    # Build a list of (n, pp[n]) for n in range where pp[n] > threshold
    threshold = max(1, best_score // 1000)  # We want pp[n] * pp[n+1] > best_score
    # Actually, let's be more targeted: look for n where pp[n] is in the top values
    
    # Collect top powerful parts
    top_pp = []
    for n in range(10000, N_MAX + 1):
        if pp[n] > threshold:
            top_pp.append((pp[n], n))
    
    # Sort by powerful part descending
    top_pp.sort(reverse=True)
    
    # For each candidate n with high pp[n], check if pp[n+1] is also high
    # This is already covered by brute force, but let's also check n-1
    # Actually, the brute force already covers all consecutive pairs.
    
    # 6. Try to find better solutions by looking at the structure more carefully
    # For each large prime power A, we want n ≡ 0 (mod A) and n+1 to have a large powerful part
    # Let's check if there are n where both n and n+1 are divisible by large prime powers
    
    # Already done in step 4. Let's add a check for cases where A and B share a factor
    # but the actual powerful parts are still large
    
    # 7. Final verification: ensure best_n is valid
    if best_n < 10000 or best_n > N_MAX:
        best_n = 10000
        
    return best_n
