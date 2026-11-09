import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 1000000
    MAX_VAL = N + 1
    
    # Linear Sieve for Smallest Prime Factor
    spf = [0] * (MAX_VAL + 1)
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > MAX_VAL:
                break
            spf[i * p] = p
            
    # Compute powerful part
    powerful_part = [1] * (MAX_VAL + 1)
    p_power = [1] * (MAX_VAL + 1)
    
    for i in range(2, MAX_VAL + 1):
        p = spf[i]
        m = i // p
        
        if spf[m] == p:
            p_power[i] = p_power[m] * p
        else:
            p_power[i] = p
            
        rem = i // p_power[i]
        
        if p_power[i] > p:
            powerful_part[i] = p_power[i] * powerful_part[rem]
        else:
            powerful_part[i] = powerful_part[rem]
            
        if time.time() > deadline:
            break

    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
            
        if time.time() > deadline:
            break
            
    return best_n
