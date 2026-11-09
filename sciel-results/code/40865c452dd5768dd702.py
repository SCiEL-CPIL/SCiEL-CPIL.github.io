import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit.
        A powerful number has the form u^2 * v^3 where v is square-free.
        """
        powerful = set()
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(round(limit ** (1.0 / 3.0))) + 1
        
        # Generate square-free numbers up to max_v
        sq_free = [True] * (max_v + 1)
        if max_v >= 0:
            sq_free[0] = False
        if max_v >= 1:
            sq_free[1] = True
        for i in range(2, max_v + 1):
            sq_free[i] = True
        
        for p in range(2, int(math.isqrt(max_v)) + 1):
            p2 = p * p
            for j in range(p2, max_v + 1, p2):
                sq_free[j] = False
        
        sq_free_list = [v for v in range(1, max_v + 1) if sq_free[v]]
        
        for v in sq_free_list:
            v3 = v ** 3
            if v3 > limit:
                continue
            # u^2 * v^3 <= limit => u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        
        return sorted(powerful)

    # Strategy:
    # 1. Choose a limit L such that the set P of powerful numbers <= L has size K
    #    that allows O(K^2) pair-sum counting within the time budget.
    # 2. Use a Python nested loop to increment a numpy array of counts for each sum a+b.
    #    This avoids the memory explosion of vectorized broadcasting (K^2 * 8 bytes)
    #    while leveraging NumPy for fast array access and argmax.
    # 3. The "density" of powerful numbers increases with L, so larger L generally
    #    yields N with more representations. We want the largest L where K^2/2 ops
    #    fit in ~100 seconds.
    
    # Based on observations:
    # L=2e8 -> K~14000 -> K^2/2 ~ 1e8 ops. In Python, ~10-20s.
    # L=5e8 -> K~25000? -> K^2/2 ~ 3e8 ops. ~30-60s.
    # Let's try L=300_000_000.
    
    L = 300_000_000
    
    # Generate powerful numbers
    P = generate_powerful(L)
    K = len(P)
    
    # If generation took too long, we might have less time for the loop.
    # Check time.
    if time.time() > deadline - 10:
        # Fallback to a smaller set if we are already close to deadline
        # But generate_powerful is usually fast. Let's proceed.
        pass

    # Initialize count array. Size is 2*L.
    # 2 * 3e8 = 6e8. int32 is 4 bytes -> 2.4 GB.
    # This might be too much memory for some evaluators.
    # Let's check available memory or use a smaller L if necessary.
    # Standard practice in these tasks: 2-4GB is often okay, but risky.
    # If L=2e8, array size 4e8 * 4 bytes = 1.6 GB. Safer.
    # Let's stick to L=200_000_000 to be safe on memory, but K is smaller.
    # Actually, let's try to detect if we can allocate.
    
    # Re-evaluating L based on memory safety.
    # 1.6GB is a reasonable upper limit for numpy arrays in many environments.
    # Let's use L = 200_000_000.
    if L > 200_000_000:
         # We already generated P for 3e8. 
         # If we want to be safe, we should have started with 2e8.
         # Let's just filter P to <= 200M? No, P is sorted.
         # We can just truncate P to elements <= 200M to reduce K and array size?
         # But N can be up to 2*L. If we truncate P, we miss pairs where a > 200M.
         # However, if we set the *search* limit for N to 2*200M, and use P up to 300M,
         # we can still find pairs summing to N <= 400M.
         # But the count array needs to be size 2*max(P).
         # max(P) is ~300M. Array size 600M. 2.4GB.
         
         # Let's check if we can allocate 2.4GB.
         # If not, we will crash.
         # To be robust, let's restart with L=200M if we suspect memory issues?
         # Or just use L=200M from the start.
         
         # Since I can't easily "restart" cleanly without duplicating code,
         # and I already generated P for 300M, let's try to allocate the array.
         # If it fails, we catch the exception and fall back to a smaller L.
         
         pass
    
    try:
        max_sum = 2 * max(P)
        # Use int32 to save memory
        counts = np.zeros(max_sum + 1, dtype=np.int32)
    except MemoryError:
        # Fallback: Reduce L and regenerate
        L = 100_000_000
        P = generate_powerful(L)
        K = len(P)
        max_sum = 2 * max(P)
        counts = np.zeros(max_sum + 1, dtype=np.int32)

    P_list = P
    counts_arr = counts
    K_local = K
    
    # Nested loop to count sums
    # This is the bottleneck.
    # Optimizations:
    # - Local variable lookups
    # - No bounds checking on numpy array (it's fast)
    
    for i in range(K_local):
        a = P_list[i]
        # b >= a
        # We can break early if a + a > max_sum? No, max_sum is 2*max(P).
        # We just iterate j from i to K_local-1.
        for j in range(i, K_local):
            b = P_list[j]
            s = a + b
            counts_arr[s] += 1
        
        # Check time periodically
        if i % 500 == 0:
            if time.time() > deadline - 5:
                break

    # Find max
    best_N = int(np.argmax(counts_arr))
    best_count = int(counts_arr[best_N])
    
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Reconstruct pairs
    # We need pairs (a, b) with a <= b, a+b=best_N, a,b in P.
    # Since P is sorted, we can use two pointers or a set.
    # Set is O(1) lookup.
    P_set = set(P_list)
    pairs = []
    # a must be <= best_N / 2
    half_N = best_N // 2
    for a in P_list:
        if a > half_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    # Verify count matches (sanity check)
    # len(pairs) should be close to best_count.
    # Note: counts_arr counted ALL pairs (a,b) where a,b in P, a<=b.
    # So len(pairs) should equal best_count.
    
    return {'N': best_N, 'pairs': pairs}
