import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1
    DEADLINE = 85  # Safety margin before 90s kill

    # Arrays for linear sieve
    # spf[i] = smallest prime factor of i
    # exp[i] = exponent of spf[i] in i
    # powerful[i] = powerful part of i
    spf = np.zeros(N_LIMIT + 1, dtype=np.int64)
    exp = np.zeros(N_LIMIT + 1, dtype=np.int64)
    powerful = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    primes = []
    
    powerful[1] = 1
    spf[1] = 1
    exp[1] = 0
    
    for i in range(2, N_LIMIT + 1):
        if time.time() - start_time > DEADLINE:
            break
            
        if spf[i] == 0:
            # i is prime
            spf[i] = i
            exp[i] = 1
            powerful[i] = 1
            primes.append(i)
            
        for p in primes:
            if p > spf[i] or i * p > N_LIMIT:
                break
                
            j = i * p
            spf[j] = p
            
            if p == spf[i]:
                # p divides i
                exp[j] = exp[i] + 1
                if exp[j] >= 2:
                    if exp[i] >= 2:
                        # Remove old p^exp[i], add new p^exp[j]
                        # powerful[i] = (p^exp[i]) * rest
                        # powerful[j] = (p^exp[j]) * rest
                        powerful[j] = (powerful[i] // (p ** exp[i])) * (p ** exp[j])
                    else:
                        # exp[i] < 2, so p was not in powerful[i]
                        # powerful[j] = p^exp[j] * powerful[i]
                        powerful[j] = powerful[i] * (p ** exp[j])
                else:
                    # exp[j] < 2, so p is not in powerful[j]
                    # powerful[j] = powerful[i]
                    powerful[j] = powerful[i]
            else:
                # p does not divide i
                exp[j] = 1
                # p^1 is not powerful, so powerful[j] = powerful[i]
                powerful[j] = powerful[i]

    # Check if we finished the sieve in time
    if time.time() - start_time > DEADLINE:
        # If we ran out of time, we might have incomplete data.
        # However, linear sieve is very fast for 1e6. 
        # If it didn't finish, we fall back to a simpler scan of what we have.
        pass

    # Find max product
    max_search = min(MAX_N, N_LIMIT - 1)
    best_n = MIN_N
    max_prod = 0
    
    if max_search > MIN_N:
        # We can vectorize this for speed, but a simple loop is also fine for 1e6
        # Let's use numpy for the product calculation to be safe and fast
        # Ensure we only use computed values
        # Since the sieve is O(N) and N=1e6, it should finish in < 0.1s
        # So powerful array should be fully populated.
        
        # products = powerful[MIN_N:max_search + 1] * powerful[MIN_N + 1:max_search + 2]
        # max_idx = np.argmax(products)
        # best_n = MIN_N + max_idx
        
        # To be absolutely safe against any overflow or type issues (though int64 is large enough),
        # and to avoid creating a huge temporary array if not needed (though 1e6 is small),
        # we can just do it in numpy.
        
        p_n = powerful[MIN_N:max_search + 1]
        p_np1 = powerful[MIN_N + 1:max_search + 2]
        
        # Check for any zeros or uncomputed values (shouldn't happen)
        # If any are 0 or 1 (uncomputed might be 1), it's fine, product will be low.
        
        products = p_n * p_np1
        max_idx = np.argmax(products)
        best_n = MIN_N + int(max_idx)
        
    return int(best_n)
