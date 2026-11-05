def check_smooth_run(p, primes, k_max=200):
    # primes must contain all primes <= p
    start = p * p + 1
    end = p * p + k_max
    rem = [0] * k_max
    for i in range(k_max):
        rem[i] = start + i
        
    for q in primes:
        if q > p:
            break
        # Find first multiple of q in [start, end]
        # m = ceil(start / q) * q
        m = ((start + q - 1) // q) * q
        if m < start:
            m += q
        for val in range(m, end + 1, q):
            i = val - start
            while rem[i] % q == 0:
                rem[i] //= q
                
    # Check how many are 1
    k = 0
    for i in range(k_max):
        if rem[i] == 1:
            k += 1
        else:
            break
    return k
