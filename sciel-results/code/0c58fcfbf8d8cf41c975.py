import math

def construct():
    N = 10_000_000
    limit = int(math.isqrt(N)) + 1
    is_prime = [True] * (limit + 1)
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    def fast_sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factor_sum = 1
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    factor_sum += power
                result *= factor_sum
        if temp > 1:
            result *= (1 + temp)
        return result

    best_error = float('inf')
    best_a = None
    best_b = None
    
    for b in range(N, 0, -1):
        if 1.0 / (2.0 * b) > best_error:
            break
            
        start_k = 1 if b % 2 == 1 else 2
        max_k = 2 * N - 3 * b
        if max_k < start_k:
            continue
            
        max_k_to_check = int(2 * b * best_error)
        if max_k_to_check < start_k:
            continue
            
        upper_k = min(max_k, max_k_to_check)
        
        for k in range(start_k, upper_k + 1, 2):
            a1 = (3 * b + k) // 2
            if a1 <= N:
                if fast_sigma(a1) == fast_sigma(b):
                    err = k / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a1
                        best_b = b
                    if k == 1:
                        return [int(a1), int(b)]
                        
            a2 = (3 * b - k) // 2
            if 1 <= a2 <= N:
                if fast_sigma(a2) == fast_sigma(b):
                    err = k / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a2
                        best_b = b
                    if k == 1:
                        return [int(a2), int(b)]

    if best_a is not None:
        return [int(best_a), int(best_b)]
    else:
        return [11, 6]
