import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin

    # We need primes up to sqrt(10^12) = 10^6
    prime_limit = 10**6
    
    # Sieve of Eratosthenes for primes up to 10^6
    is_prime = bytearray(b'\x01') * (prime_limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(prime_limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:prime_limit+1:i] = bytearray((prime_limit - i*i) // i + 1)
    primes = [i for i, v in enumerate(is_prime) if v]

    # Function to compute phi for a range [s, s+k-1] efficiently
    # We factorize each number by dividing out small primes.
    # Since k is small (<= 200), and we have primes up to 10^6,
    # for each n in the range, we can just trial divide by primes up to sqrt(n).
    # But doing this for 200 numbers is fast enough.
    
    def get_phis(s, k):
        phis = []
        seen = set()
        for i in range(k):
            n = s + i
            phi_val = n
            temp = n
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    phi_val -= phi_val // p
                    while temp % p == 0:
                        temp //= p
            if temp > 1:
                phi_val -= phi_val // temp
            
            if phi_val in seen:
                return None, i  # Collision found at index i
            seen.add(phi_val)
            phis.append(phi_val)
        return phis, -1

    # Strategy:
    # 1. Try s = 10^12 - 199, k = 200
    # 2. If fail, try s = 10^12 - 200, k = 199 (or just try nearby starts with k=200)
    # Actually, let's try a few candidates near 10^12 with k=200 first.
    # Then if all fail, try k=199, 198, etc.
    
    # But we should maximize k. So we check k=200 first.
    
    # Candidate 1: s = 10^12 - 199
    s = 10**12 - 199
    if time.time() < deadline:
        _, idx = get_phis(s, 200)
        if idx == -1:
            return {'start': s, 'length': 200}

    # Try other starts for k=200
    # The probability of collision is low, but let's check a few more.
    # We can search backwards from 10^12 - 199.
    
    # Optimization: We can check multiple starts in parallel? No, Python is single-threaded for CPU bound.
    # But we can just iterate.
    
    # Let's try 100 more starts for k=200
    for offset in range(1, 100):
        if time.time() > deadline - 5:
            break
        s = 10**12 - 199 - offset
        if s < 1:
            break
        _, idx = get_phis(s, 200)
        if idx == -1:
            return {'start': s, 'length': 200}
        
        # If we found a collision early, it might be faster to try a different start
        # rather than continuing with this one (but we already stopped at collision).
        
    # If k=200 fails, try k=199
    # Start from 10^12 - 198
    for k in range(199, 195, -1):
        if time.time() > deadline - 2:
            break
        # Try a few starts for this k
        # The best start is likely still near the top
        max_offset = 50
        for offset in range(max_offset):
            if time.time() > deadline - 2:
                break
            s = 10**12 - (k - 1) - offset
            if s < 1:
                break
            _, idx = get_phis(s, k)
            if idx == -1:
                return {'start': s, 'length': k}

    # Fallback: A known valid run
    # phi(1)=1, phi(2)=1 (collision)
    # phi(2)=1, phi(3)=2 (distinct)
    # So start=2, length=2
    # Or start=1, length=1
    # Let's return a safe valid run.
    # Check start=1, k=1: phi(1)=1. Valid.
    # Check start=2, k=2: phi(2)=1, phi(3)=2. Valid.
    # Check start=3, k=3: phi(3)=2, phi(4)=2 (collision).
    # Check start=4, k=3: phi(4)=2, phi(5)=4, phi(6)=2 (collision).
    # Check start=5, k=3: phi(5)=4, phi(6)=2, phi(7)=6. Distinct.
    # So start=5, length=3 is valid.
    
    # Let's try to find a slightly longer one quickly if time permits.
    # But for now, return a valid one.
    # Actually, we can just return the best we found if any, but we only return on success.
    # So if we get here, we haven't found one.
    
    # Let's do a quick search for a small valid run to be sure.
    for s in range(1, 100):
        for k in range(1, 10):
            if s + k - 1 > 10**12:
                break
            _, idx = get_phis(s, k)
            if idx == -1:
                return {'start': s, 'length': k}
                
    # Absolute fallback (should not reach here if logic is correct)
    return {'start': 1, 'length': 1}
