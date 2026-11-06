import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    N = 1000001  # we need n and n+1, so up to 1000001
    
    # Sieve for smallest prime factor
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    def powerful_part(n):
        if n <= 1:
            return 1
        pp = 1
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count >= 2:
                pp *= p ** count
        return pp
    
    # Precompute powerful parts for all n in [1, N]
    pp_arr = [0] * (N + 1)
    pp_arr[0] = 0
    pp_arr[1] = 1
    for i in range(2, N + 1):
        pp_arr[i] = powerful_part(i)
    
    # Find n in [10000, 1000000] that maximizes pp_arr[n] * pp_arr[n+1]
    best_n = 10000
    best_val = pp_arr[10000] * pp_arr[10001]
    
    for n in range(10000, 1000001):
        val = pp_arr[n] * pp_arr[n+1]
        if val > best_val:
            best_val = val
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
