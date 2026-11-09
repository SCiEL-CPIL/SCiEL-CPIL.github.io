import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin
    
    N = 123457
    
    def find_smooth_pair(primes, limit):
        """
        Generates all numbers composed only of the given primes up to limit.
        Checks if any pair sums to N.
        Returns [a, N-a] if found, else None.
        """
        smooth_set = set()
        
        # Recursive generation is cleaner for variable number of primes
        def generate(current, idx):
            if current > limit:
                return
            if idx == len(primes):
                smooth_set.add(current)
                return
            # Multiply by prime[idx] as many times as possible
            p = primes[idx]
            next_val = current
            while next_val <= limit:
                generate(next_val, idx + 1)
                next_val *= p
        
        generate(1, 0)
        
        # Check for pair
        for a in smooth_set:
            b = N - a
            if b > 0 and b in smooth_set:
                return [a, b]
        return None

    # Check 2-smooth (primes: [2])
    # Max LPF would be 2
    res = find_smooth_pair([2], N)
    if res:
        return res

    # Check 3-smooth (primes: [2, 3])
    # Max LPF would be 3
    res = find_smooth_pair([2, 3], N)
    if res:
        return res

    # Check 5-smooth (primes: [2, 3, 5])
    # Max LPF would be 5
    res = find_smooth_pair([2, 3, 5], N)
    if res:
        return res

    # Check 7-smooth (primes: [2, 3, 5, 7])
    # Max LPF would be 7
    res = find_smooth_pair([2, 3, 5, 7], N)
    if res:
        return res

    # Check 11-smooth (primes: [2, 3, 5, 7, 11])
    # Max LPF would be 11
    res = find_smooth_pair([2, 3, 5, 7, 11], N)
    if res:
        return res
    
    # Fallback: Full LPF search
    # This handles cases where the optimal max LPF > 11
    
    if time.time() > deadline:
        # Emergency fallback, though unlikely to be reached
        return [1, N - 1]
    
    lpf = [1] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_a = 1
    best_val = max(lpf[1], lpf[N - 1])
    
    # Only search up to N//2 since (a, N-a) and (N-a, a) give same max
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            # The next possible LPF after 11 is 13
            if best_val == 13:
                break
    
    return [best_a, N - best_a]
