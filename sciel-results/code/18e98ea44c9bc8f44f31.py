import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Primes p = 1 mod 4
    primes_1_4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 877, 881, 929, 937, 941, 953, 977, 997]
    
    # Small powerful numbers K
    # We want K to be small to allow large M.
    # But K must be powerful.
    # 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 200, 216, 225, 243, 256, 289, 324, 343, 361, 400, 441, 484, 512, 529, 576, 625, 676, 729, 784, 841, 864, 900, 961, 1000
    powerful_Ks = [1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 200, 216, 225, 243, 256, 289, 324, 343, 361, 400, 441, 484, 512, 529, 576, 625, 676, 729, 784, 841, 864, 900, 961, 1000]
    
    best_score = 0
    best_N = 2
    best_K = 1
    best_M = 1
    best_exps = [0] * len(primes_1_4)
    
    for K in powerful_Ks:
        if time.time() > deadline - 10.0:
            break
            
        M_max = 10**18 // K
        
        # Greedy allocation of exponents
        exps = [0] * len(primes_1_4)
        current_log_M = 0.0
        log_M_max = math.log(M_max)
        score_prod = 1.0
        
        # We use a simple loop. Since the number of primes is small and exponents are small,
        # we can just iterate.
        # To be efficient, we pick the best candidate each time.
        while True:
            best_ratio = -1
            best_idx = -1
            for i, p in enumerate(primes_1_4):
                # Cost: log(p)
                # Benefit: log((exps[i]+2)/(exps[i]+1))
                # We only consider if we can afford the cost
                if current_log_M + math.log(p) > log_M_max:
                    continue
                ratio = math.log((exps[i] + 2) / (exps[i] + 1)) / math.log(p)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = i
            
            if best_idx == -1:
                break
            
            exps[best_idx] += 1
            current_log_M += math.log(primes_1_4[best_idx])
            score_prod *= (exps[best_idx] + 1) / exps[best_idx] # Wait, exps[best_idx] is now 1, so 2/1=2. Correct.
            
        # Calculate actual M and verify
        M = 1
        for i, p in enumerate(primes_1_4):
            if exps[i] > 0:
                M *= p ** exps[i]
        
        if M > M_max:
            continue
            
        # The score is the number of pairs.
        # r_2(M) = 4 * prod(e_i + 1)
        # Number of pairs approx r_2(M) / 8
        # We use the exact product as a proxy for sorting, then verify.
        # Actually, score_prod IS prod(e_i+1).
        if score_prod > best_score:
            best_score = score_prod
            best_N = K * M
            best_K = K
            best_M = M
            best_exps = exps[:]
            
    # Generate pairs for best_N
    pairs = []
    K = best_K
    M = best_M
    max_x = math.isqrt(M)
    
    for x in range(1, max_x + 1):
        if time.time() > deadline - 5.0:
            break
        rem = M - x * x
        if rem < 1:
            break
        y = math.isqrt(rem)
        if y * y == rem and y >= x:
            a = K * x * x
            b = K * y * y
            # a, b are powerful by construction
            # Check a <= b
            if a <= b:
                pairs.append([a, b])
    
    # If no pairs found (e.g., M=1, x=1, y=0 not allowed), fallback
    if not pairs:
        # Try to find at least one pair
        # If M is a sum of two squares, there should be some.
        # If M=1, x=1, y=0. y must be >= 1. So no pairs for M=1 with x,y>=1.
        # But 1 is powerful.
        # If N=2, a=1, b=1.
        # 1+1=2.
        # Is 1 powerful? Yes.
        # So if best_M=1, we should handle it.
        if best_M == 1:
            pairs = [[best_K, best_K]]
            if best_K + best_K > 10**18:
                pairs = [[1, 1]]
                best_N = 2
            else:
                best_N = 2 * best_K
    
    return {'N': best_N, 'pairs': pairs}
