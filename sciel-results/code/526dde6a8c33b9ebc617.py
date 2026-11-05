import time

def construct():
    N = 123457
    start_time = time.time()
    
    # We will try B=100 first, then B=200 if needed.
    # Actually, let's just go with B=200 to be safe.
    B = 200
    
    # Generate primes up to B
    primes = []
    is_prime = [True] * (B + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, B + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, B + 1, i):
                is_prime[j] = False
            
    # Generate B-smooth numbers <= N
    current_list = [1]
    for p in primes:
        new_entries = []
        for x in current_list:
            y = x
            while y <= N:
                new_entries.append(y)
                y *= p
        current_list.extend(new_entries)
        
    smooth_set = set(current_list)
    
    def largest_prime_factor(n, primes):
        if n == 1:
            return 1
        lpf = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                lpf = p
                while n % p == 0:
                    n //= p
        if n > 1:
            # If n > 1, it must be a prime factor larger than B?
            # No, because we only generated numbers with factors <= B.
            # So if n > 1 after dividing by all p <= B, it implies n is prime and > B?
            # But wait, if n is in smooth_set, all its factors are <= B.
            # So n should become 1.
            # Unless n is a prime > B? No, smooth_set only contains numbers with factors <= B.
            # So this branch is likely dead code for numbers in smooth_set, 
            # unless n itself is a prime > B, which is impossible for smooth numbers.
            # However, if we check a number NOT in smooth_set, we wouldn't call this.
            # We only call this for a and b which are in smooth_set.
            # So n should always reduce to 1.
            lpf = n
        return lpf
        
    best_score = float('inf')
    best_pair = None
    
    for a in smooth_set:
        if a >= N:
            continue
        b = N - a
        if b in smooth_set:
            pa = largest_prime_factor(a, primes)
            pb = largest_prime_factor(b, primes)
            score = max(pa, pb)
            if score < best_score:
                best_score = score
                best_pair = (a, b)
                
    if best_pair is not None:
        return list(best_pair)
    else:
        return [1, 123456]
