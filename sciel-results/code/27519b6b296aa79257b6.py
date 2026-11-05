import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    smooth = {1}
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Generate new smooth numbers involving p
        snapshot = list(smooth)
        new_numbers = set()
        
        for x in snapshot:
            y = x * p
            while y <= N:
                if y not in smooth:
                    new_numbers.add(y)
                y *= p
        
        if not new_numbers:
            continue
        
        # Add new numbers to smooth
        smooth.update(new_numbers)
        
        # Check if any newly added number has its complement in smooth
        for a in new_numbers:
            if time.time() > deadline:
                break
            b = N - a
            if b >= 1 and b in smooth:
                return [a, b]
    
    # Fallback: return 1, N-1 (valid but may not be optimal)
    return [1, N-1]
