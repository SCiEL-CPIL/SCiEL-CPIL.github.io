import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe
    
    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def phi(x):
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result

    # 2. Search for a run of length 200 near 10^12
    # We want s + k - 1 <= 10^12. Let's try s = 10^12 - 199.
    # If that fails, we can try smaller s.
    # Given the low collision probability, s = 10^12 - 199 is very likely to work.
    
    max_k = 200
    target_end = 10**12
    
    # Try a few candidates near the top
    # It is extremely likely that the first candidate works.
    # To be robust, we can try a few.
    
    # Let's try s = 10^12 - 199
    s = target_end - max_k + 1
    
    # Verify s >= 1
    if s < 1:
        s = 1
        max_k = min(max_k, target_end) # Not applicable here as 10^12 > 200

    # Check if phi(s)...phi(s+199) are distinct
    # We can do this for a few s values if the first one fails.
    
    best_s = 1
    best_k = 1
    
    # We will try to find a run of length 200.
    # If we find one, we return it.
    # If we don't find one in time, we return the best found.
    
    # Since collision is rare, we expect to find k=200 immediately.
    # Let's just check s = 10^12 - 199.
    
    val = s
    seen = set()
    k = 0
    while val <= target_end and k < max_k:
        if time.time() > deadline:
            break
        pv = phi(val)
        if pv in seen:
            break
        seen.add(pv)
        k += 1
        val += 1
        
    if k == max_k:
        return {'start': s, 'length': k}
        
    # If the first one failed (unlikely), try a few more random spots or step down.
    # But stepping down from 10^12 is not "random", it's just adjacent.
    # Adjacent ranges will have similar collision properties.
    # Let's try a few other starts.
    
    # Actually, if k < 200, we can try to extend or find a better spot.
    # But for the purpose of maximizing k, finding ANY spot with k=200 is optimal.
    
    # Let's try 5 more candidates spaced out.
    candidates = [10**12 - 200 - i*1000 for i in range(5)]
    for s_try in candidates:
        if s_try < 1: continue
        if time.time() > deadline: break
        seen = set()
        k_try = 0
        val = s_try
        while val <= target_end and k_try < 200:
            pv = phi(val)
            if pv in seen:
                break
            seen.add(pv)
            k_try += 1
            val += 1
        if k_try > best_k:
            best_k = k_try
            best_s = s_try
        if best_k == 200:
            break
            
    return {'start': best_s, 'length': best_k}
