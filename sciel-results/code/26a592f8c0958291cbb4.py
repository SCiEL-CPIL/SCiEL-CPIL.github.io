import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 1000000
    
    # 1. Compute largest prime factor for all numbers up to N
    # We use a sieve approach.
    # lpf[i] will store the largest prime factor of i.
    lpf = [0] * (N + 1)
    
    # Initialize lpf for primes? No, we just iterate.
    # If we iterate i from 2 to N, and if i is prime, we update its multiples.
    # Since we go in increasing order, the last prime factor assigned to a number
    # is its largest prime factor.
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Note: n+1 must be <= 1000000, so n goes up to 999999.
    # The task says n in [10000, 1000000].
    # If n=1000000, n+1=1000001. We haven't computed lpf for 1000001.
    # However, the range is inclusive.
    # Let's check the constraint: "Return integer n in [10000,1000000]".
    # The metric is "largest prime factor of n(n+1)".
    # If n=1000000, we need lpf(1000000) and lpf(1000001).
    # 1000001 = 101 * 9901. Largest prime factor is 9901.
    # 1000000 = 10^6 = 2^6 * 5^6. Largest prime factor is 5.
    # Max is 9901.
    # It is highly unlikely that n=1000000 is the optimal solution compared to smaller n
    # where both n and n+1 are very smooth.
    # To be safe and correct, we should handle n=1000000 separately or extend the sieve.
    # Extending the sieve to 1000001 is trivial.
    
    N_ext = 1000001
    # Re-run sieve for N_ext to be precise?
    # The previous loop went to N=1000000.
    # We can just compute lpf for 1000001 manually or extend the array.
    # Let's just re-run the sieve up to 1000001 to be clean.
    # It's fast enough.
    
    lpf = [0] * (N_ext + 1)
    for i in range(2, N_ext + 1):
        if lpf[i] == 0:
            for j in range(i, N_ext + 1, i):
                lpf[j] = i
                
    best_n = 10000
    best_val = float('inf')
    
    # Check all n in [10000, 1000000]
    # n+1 will be in [10001, 1000001], which is covered by lpf array.
    for n in range(10000, N + 1):
        if time.time() > deadline:
            break
        lpf_n = lpf[n]
        lpf_np1 = lpf[n + 1]
        current_val = lpf_n if lpf_n > lpf_np1 else lpf_np1
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
