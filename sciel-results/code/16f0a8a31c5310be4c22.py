import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    N = 10_000_000
    target_ratio = 1.5
    
    # Step 1: Generate prime powers and their sigma values
    # We need primes up to N
    # Sieve of Eratosthenes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Generate prime powers p^alpha <= N
    # sigma(p^alpha) = (p^(alpha+1) - 1) / (p - 1)
    prime_powers = []
    for p in primes:
        val = p
        sig_val = 1 + p
        while val <= N:
            prime_powers.append((val, sig_val))
            val *= p
            if val > N:
                break
            sig_val = sig_val * p + 1 # sigma(p^k) = p * sigma(p^(k-1)) + 1? No.
            # sigma(p^k) = 1 + p + ... + p^k = (p^(k+1)-1)/(p-1)
            # It is better to compute directly or iteratively:
            # s_k = s_{k-1} * p + 1 is WRONG.
            # s_k = 1 + p + ... + p^k.
            # s_{k-1} = 1 + ... + p^{k-1}.
            # s_k = s_{k-1} + p^k.
            # Let's just compute (p**(alpha+1)-1)//(p-1)
            pass
            
    # Re-do prime powers correctly
    prime_powers = []
    for p in primes:
        val = p
        alpha = 1
        while val <= N:
            sig_val = (p**(alpha+1) - 1) // (p - 1)
            prime_powers.append((val, sig_val))
            val *= p
            alpha += 1
            if val > N:
                break
                
    # Group by sigma value
    # Only keep groups with at least 2 distinct values
    groups = {}
    for val, sig in prime_powers:
        if sig in groups:
            groups[sig].append(val)
        else:
            groups[sig] = [val]
            
    base_pairs = []
    for sig, vals in groups.items():
        if len(vals) >= 2:
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    a, b = vals[i], vals[j]
                    if a != b:
                        base_pairs.append((a, b))
                        
    if time.time() - start_time > deadline - 10:
        # Fallback to simple pair if search takes too long
        return [11, 6]

    # Step 2: Search for combinations of base pairs
    # We want to minimize | (prod A_i) / (prod B_i) - 1.5 |
    # Let R = prod (A_i / B_i). We want R close to 1.5.
    # Logarithmic space: sum log(A_i/B_i) close to log(1.5).
    
    # Sort base pairs by ratio
    base_pairs.sort(key=lambda x: x[0] / x[1])
    
    # Use a greedy or DP approach?
    # Given the time limit, a simple local search or greedy accumulation might work.
    # Or, just check single pairs first.
    
    best_num = 4
    best_den = 12
    best_a = 11
    best_b = 6
    
    # Check single pairs
    for a, b in base_pairs:
        if a > N or b > N:
            continue
        num = abs(2 * a - 3 * b)
        den = 2 * b
        if num == 0:
            continue
        if num * best_den < best_num * den:
            best_num = num
            best_den = den
            best_a = a
            best_b = b
            
    # Try combining two pairs
    # This is O(M^2) which might be too slow if M is large.
    # M is number of base pairs.
    # Let's estimate M.
    # Number of primes ~ 664k.
    # Most primes have only p^1. sigma(p) = p+1.
    # Collisions in sigma(p) = p+1 are rare for primes (p1+1 = p2+1 => p1=p2).
    # Collisions happen between prime powers of different primes.
    # e.g., sigma(2^2)=7, sigma(6)=12 (not prime power).
    # sigma(2^1)=3. sigma(1^2)? No.
    # sigma(3^1)=4.
    # sigma(5^1)=6.
    # sigma(7^1)=8.
    # sigma(11^1)=12. sigma(6)=12 (6 is not prime power).
    # So base pairs from prime powers are sparse.
    
    # Let's just iterate through all pairs of base pairs if M is small enough.
    M = len(base_pairs)
    if M < 1000:
        for i in range(M):
            a1, b1 = base_pairs[i]
            for j in range(M):
                a2, b2 = base_pairs[j]
                # Check coprime condition for multiplicativity
                # gcd(a1, a2) == 1 and gcd(b1, b2) == 1 and gcd(a1, b2)==1 and gcd(a2, b1)==1
                # Actually, we just need gcd(a1, b1)=1? No, sigma is multiplicative for coprime args.
                # If we take a = a1 * a2, b = b1 * b2.
                # sigma(a) = sigma(a1) sigma(a2) if gcd(a1, a2)=1.
                # sigma(b) = sigma(b1) sigma(b2) if gcd(b1, b2)=1.
                # We know sigma(a1)=sigma(b1) and sigma(a2)=sigma(b2).
                # So sigma(a) = sigma(a1)sigma(a2) = sigma(b1)sigma(b2) = sigma(b).
                # This holds if gcd(a1, a2)=1 and gcd(b1, b2)=1.
                # We also need a, b <= N.
                
                if a1 * a2 > N or b1 * b2 > N:
                    continue
                if gcd(a1, a2) != 1 or gcd(b1, b2) != 1:
                    continue
                
                a = a1 * a2
                b = b1 * b2
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    continue
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b

    return [int(best_a), int(best_b)]
