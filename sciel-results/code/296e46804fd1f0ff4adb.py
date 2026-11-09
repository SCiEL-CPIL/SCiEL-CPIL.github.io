import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    
    def phi(x):
        result = x
        n = x
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
        if n > 1:
            result -= result // n
        return result

    def compute_phi_segment(start, length):
        """Compute phi for consecutive integers [start, start+length-1] using segmented approach."""
        # Initialize: phi[i] = start+i, remaining[i] = start+i
        phis = [0] * length
        remaining = [0] * length
        for i in range(length):
            phis[i] = start + i
            remaining[i] = start + i
        
        # For each prime p, find multiples in [start, start+length-1]
        for p in primes:
            if p > length and p * p > start + length - 1:
                # If p > length, there's at most one multiple. But we need p*p <= max value
                # Actually, if p > length, we can have at most one multiple of p in the range
                # Check if there's a multiple
                if p * p > start + length - 1:
                    # No multiple of p^2, so if there's a multiple of p, it appears once
                    # Find first multiple >= start
                    if start % p == 0:
                        idx = 0
                    else:
                        idx = p - (start % p)
                    if idx < length:
                        # There's exactly one multiple at position idx
                        rem = remaining[idx]
                        if rem % p == 0:
                            phis[idx] -= phis[idx] // p
                            while rem % p == 0:
                                rem //= p
                            remaining[idx] = rem
                    continue
            
            # Find first multiple of p in [start, start+length-1]
            if start % p == 0:
                first = start
            else:
                first = start + (p - start % p)
            
            if first > start + length - 1:
                continue
            
            # Process all multiples of p
            for j in range(first, start + length, p):
                idx = j - start
                rem = remaining[idx]
                if rem % p == 0:
                    phis[idx] -= phis[idx] // p
                    while rem % p == 0:
                        rem //= p
                    remaining[idx] = rem
        
        # Handle remaining prime factors > sqrt(max)
        for i in range(length):
            rem = remaining[i]
            if rem > 1:
                phis[i] -= phis[i] // rem
        
        return phis

    best_k = 1
    best_s = 1
    
    # 2. Search in high range
    low = 10**11
    high = 10**12 - 200
    
    random.seed(42)
    
    attempts = 0
    # Try random starting points
    while time.time() < deadline:
        attempts += 1
        s = random.randint(low, high)
        
        # Compute phi for a segment of length 200
        phis = compute_phi_segment(s, 200)
        
        # Check for distinct values
        seen = set()
        k = 0
        valid = True
        for i in range(200):
            val = phis[i]
            if val in seen:
                valid = False
                if k > best_k:
                    best_k = k
                    best_s = s
                break
            seen.add(val)
            k += 1
        
        if valid:
            return {'start': s, 'length': 200}
        
        # If we're getting close to deadline, try a more targeted search
        # for longer runs by scanning sequentially from promising areas
        if attempts % 50 == 0 and time.time() > deadline - 60:
            # Do a sequential scan from a random point
            scan_start = random.randint(low, high - 1000)
            phis = compute_phi_segment(scan_start, 1000)
            # Find longest run of distinct values
            seen = set()
            current_k = 0
            current_start = scan_start
            for i in range(1000):
                val = phis[i]
                if val in seen:
                    if current_k > best_k:
                        best_k = current_k
                        best_s = current_start
                    seen = set()
                    current_k = 0
                    current_start = scan_start + i + 1
                else:
                    if current_k == 0:
                        current_start = scan_start + i
                    current_k += 1
                seen.add(val)
            if current_k > best_k:
                best_k = current_k
                best_s = current_start

    return {'start': best_s, 'length': best_k}
