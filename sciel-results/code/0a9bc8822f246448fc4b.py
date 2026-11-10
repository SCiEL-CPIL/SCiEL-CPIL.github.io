import math

def construct():
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def compute_phi_range(start, length):
        rem = [start + i for i in range(length)]
        phi = [start + i for i in range(length)]
        
        max_val = start + length - 1
        sqrt_max = int(math.isqrt(max_val))
        
        for p in primes:
            if p > sqrt_max:
                break
            
            first_j = (-start) % p
            if first_j < length:
                for j in range(first_j, length, p):
                    while rem[j] % p == 0:
                        rem[j] //= p
                        phi[j] -= phi[j] // p
        
        for j in range(length):
            if rem[j] > 1:
                phi[j] -= phi[j] // rem[j]
                
        return phi

    max_k = 200
    base_s = 10**11
    
    for offset in range(0, 1000):
        s = base_s + offset * 1000
        if s + max_k - 1 > 10**12:
            break
            
        phis = compute_phi_range(s, max_k)
        
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
            
    return {'start': 1, 'length': 1}
