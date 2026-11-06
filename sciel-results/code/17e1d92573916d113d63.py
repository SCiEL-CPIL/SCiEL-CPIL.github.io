"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    N = 1000000
    LO = 10000
    HI = N
    
    # powerful_part[i] = product of p^e for all primes p where p^e || i and e >= 2
    # We compute this via a sieve-like approach
    # For each prime p, for each power p^k (k>=2), multiply into all multiples of p^k
    
    # First, generate primes up to sqrt(N) using sieve
    sqrt_N = int(math.isqrt(N))
    is_prime = [True] * (sqrt_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(sqrt_N)) + 1):
        if is_prime[i]:
            for j in range(i*i, sqrt_N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, sqrt_N + 1) if is_prime[i]]
    
    # powerful_part array, initialized to 1
    pw = [1] * (N + 1)
    
    # For each prime p, find all powers p^k with k>=2 and p^k <= N
    # For each such power, multiply pw[j] by p^k for all j that are multiples of p^k
    # BUT we need to be careful: if p^k divides j, we want to include p^k in the powerful part
    # The powerful part of j is the product of p^e where e = v_p(j) and e >= 2
    # So for prime p, if v_p(j) = e >= 2, we include p^e
    # This means for each prime p, we find the highest power p^e dividing each j
    # and if e >= 2, we multiply pw[j] by p^e
    
    # Efficient approach: for each prime p, iterate through multiples
    # For each multiple m of p, determine v_p(m)
    # If v_p(m) >= 2, multiply pw[m] by p^v_p(m)
    
    # Even more efficient: for each prime p, for each power p^k (k>=2):
    #   For each multiple m of p^k:
    #     Check if p^(k+1) does NOT divide m (i.e., v_p(m) == k)
    #     If so, multiply pw[m] by p^k
    #   But we also need to handle the case where v_p(m) > k
    # 
    # Actually, the cleanest approach:
    # For each prime p:
    #   For each multiple m of p:
    #     Compute e = v_p(m) by repeated division
    #     If e >= 2, multiply pw[m] by p^e
    
    # But this is O(N log N) which might be slow for N=1e6
    # Let's use the sparse update strategy:
    # For each prime p, for each power p^k with k>=2 and p^k <= N:
    #   For each multiple m of p^k:
    #     pw[m] *= p  (we accumulate by multiplying by p for each level)
    # Wait, that's not quite right either.
    
    # Let me think again. The powerful part of n is prod_{p: v_p(n)>=2} p^{v_p(n)}
    # For a fixed prime p, the contribution to pw[m] is p^{v_p(m)} if v_p(m)>=2, else 1.
    # 
    # Strategy: for each prime p, for each power p^k where k>=2:
    #   Mark all multiples of p^k. For a multiple m of p^k:
    #     v_p(m) >= k
    #   The highest k such that p^k | m gives v_p(m) = k (or higher)
    #   
    # Better: for each prime p, iterate m = p, 2p, 3p, ...
    #   For each m, count how many times p divides m (v_p(m))
    #   If v_p(m) >= 2, multiply pw[m] by p^{v_p(m)}
    
    # This is O(N * number_of_primes / p) which is manageable
    # For p=2: N/2 iterations, p=3: N/3, etc. Total ~ N * sum(1/p) ~ N * log(log(N))
    # For N=1e6, this is about 1e6 * 3 = 3e6 operations, very fast.
    
    for p in primes:
        # For each multiple of p, compute v_p(m)
        # m = p, 2p, 3p, ..., N
        # For m = k*p, v_p(m) = v_p(k) + 1
        # So we can recursively determine it, or just do repeated division
        
        # Iterate through multiples of p
        for m in range(p, N + 1, p):
            # Count v_p(m)
            temp = m
            e = 0
            while temp % p == 0:
                e += 1
                temp //= p
            if e >= 2:
                pw[m] *= (p ** e)
    
    # Now find the pair (n, n+1) in [LO, HI] that maximizes pw[n] * pw[n+1]
    best_val = 0
    best_n = LO
    for n in range(LO, HI):
        val = pw[n] * pw[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            if time.time() > deadline:
                break
    
    return best_n

# EVOLVE-BLOCK-END
