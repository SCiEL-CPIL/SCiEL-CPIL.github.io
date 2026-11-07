import array
import bisect

def construct():
    N = 10_000_000
    
    # Linear Sieve for sigma
    sigma = array.array('I', [0]) * (N + 1)
    lp = array.array('I', [0]) * (N + 1)
    pwr = array.array('I', [0]) * (N + 1)
    sig_pwr = array.array('I', [0]) * (N + 1)
    
    primes = []
    
    sigma[1] = 1
    lp[1] = 1
    pwr[1] = 1
    sig_pwr[1] = 1
    
    for i in range(2, N + 1):
        if lp[i] == 0:
            lp[i] = i
            pwr[i] = i
            sig_pwr[i] = 1 + i
            sigma[i] = 1 + i
            primes.append(i)
            
        for p in primes:
            if p > lp[i] or i * p > N:
                break
            lp[i * p] = p
            if p == lp[i]:
                pwr[i * p] = pwr[i] * p
                sig_pwr[i * p] = sig_pwr[i] + pwr[i * p]
                sigma[i * p] = sig_pwr[i * p] * sigma[i // pwr[i]]
            else:
                pwr[i * p] = p
                sig_pwr[i * p] = 1 + p
                sigma[i * p] = sig_pwr[i * p] * sigma[i]
                
    # Create combined keys for sorting
    # key = sigma * (N + 1) + n
    keys = array.array('Q', [0]) * N
    for n in range(1, N + 1):
        keys[n-1] = sigma[n] * (N + 1) + n
        
    keys.sort()
    
    best_a = 31
    best_b = 21
    best_err = abs(31/21 - 1.5)
    
    i = 0
    while i < N:
        current_sigma = keys[i] // (N + 1)
        group = []
        while i < N and keys[i] // (N + 1) == current_sigma:
            group.append(keys[i] % (N + 1))
            i += 1
            
        if len(group) < 2:
            continue
            
        group.sort()
        
        for b in group:
            target = 1.5 * b
            idx = bisect.bisect_left(group, target)
            
            if idx < len(group):
                a = group[idx]
                if a != b and 2 * a != 3 * b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
            if idx > 0:
                a = group[idx - 1]
                if a != b and 2 * a != 3 * b:
                    err = abs(a / b - 1.5)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        
    return [best_a, best_b]
