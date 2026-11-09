import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 123457
    
    # Strategy: Generate smooth numbers (numbers with only small prime factors)
    # and check if both a and N-a are smooth. The smallest smoothness bound k
    # for which a pair exists gives the optimal answer.
    
    # For each k (prime bound), generate all k-smooth numbers up to N.
    # Then check if there's a pair a, N-a both in the set.
    
    # Primes to consider as bounds: 2, 3, 5, 7, 11, 13, 17, 19, ...
    # We'll try increasing bounds until we find a valid pair.
    
    # First, let's try to be smart: generate 5-smooth, 7-smooth, 11-smooth, etc.
    
    def generate_smooth_numbers(prime_list, limit):
        """Generate all numbers <= limit whose prime factors are all in prime_list."""
        if not prime_list:
            return {1}
        
        # Use a set-based generation
        smooth = set()
        
        def gen(idx, current):
            if current > limit:
                return
            smooth.add(current)
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            val = current
            while val <= limit:
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        return smooth
    
    # Try increasing prime bounds
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for bound_idx in range(len(primes)):
        if time.time() > deadline:
            break
        
        prime_list = primes[:bound_idx + 1]
        max_prime = prime_list[-1]
        
        smooth_set = generate_smooth_numbers(prime_list, N)
        
        # Check if there's a pair a, N-a both in smooth_set
        # We want to minimize max(P(a), P(b)) which is at most max_prime for smooth numbers
        # So any valid pair gives value <= max_prime
        # But we need the actual P(a) and P(b), which could be less than max_prime
        
        # Find all valid pairs
        best_val = float('inf')
        best_a = None
        
        for a in smooth_set:
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                # Both are smooth with primes up to max_prime
                # The actual largest prime factor could be less
                # But for now, the upper bound is max_prime
                # We'll compute exact values later if needed
                # For the purpose of finding the minimum, any pair with bound max_prime
                # has value <= max_prime
                # We want the minimum, so we should check the actual LPF
                pass
        
        # Actually, we need to be more careful. Let's compute LPF for smooth numbers.
        # But first, let's just find if ANY pair exists with this bound.
        # If it does, the answer is at most max_prime.
        # We want the minimum, so we should find the pair that minimizes the actual max LPF.
        
        # Let's collect all valid pairs and compute their actual LPF values.
        valid_pairs = []
        for a in smooth_set:
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                valid_pairs.append((a, b))
        
        if valid_pairs:
            # Compute LPF for each number in the pairs
            # Since these are smooth numbers, we can factor them easily
            def lpf_smooth(n, prime_list):
                """Compute largest prime factor of n, assuming n is smooth with primes in prime_list."""
                if n == 1:
                    return 1
                max_pf = 1
                for p in prime_list:
                    if n % p == 0:
                        max_pf = max(max_pf, p)
                        while n % p == 0:
                            n //= p
                if n > 1:
                    # This shouldn't happen if n is truly smooth
                    max_pf = max(max_pf, n)
                return max_pf
            
            for a, b in valid_pairs:
                pa = lpf_smooth(a, prime_list)
                pb = lpf_smooth(b, prime_list)
                val = max(pa, pb)
                if val < best_val:
                    best_val = val
                    best_a = a
            
            if best_a is not None:
                return [best_a, N - best_a]
    
    # Fallback: use the full sieve approach
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        n = i
        max_pf = 1
        while n > 1:
            p = spf[n]
            max_pf = max(max_pf, p)
            while n % p == 0:
                n //= p
        lpf[i] = max_pf
    
    best_a = 1
    best_val = max(lpf[1], lpf[N - 1])
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    return [best_a, N - best_a]
