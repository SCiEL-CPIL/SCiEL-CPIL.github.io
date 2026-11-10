import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Step 2: Fast phi function using precomputed primes
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
    
    # Step 3: Batch phi computation for a range [start, start+length-1]
    def phi_batch(start, length):
        """Compute phi for all numbers in [start, start+length-1]"""
        results = [0] * length
        
        # For each prime, mark multiples in the range
        for p in primes:
            if p * p > start + length - 1:
                # Check if p itself is in range
                if start <= p <= start + length - 1:
                    idx = p - start
                    results[idx] = 0  # Will handle later
                    break
                break
            
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            for multiple in range(first, start + length, p):
                idx = multiple - start
                if results[idx] == 0:
                    results[idx] = multiple  # Initialize with the number
                # Apply phi factor: multiply by (1 - 1/p)
                # We'll do this at the end to avoid floating point issues
                # Instead, track which primes divide each number
                # Actually, let's just compute phi directly for each number
                # But that's what we want to avoid...
        
        # Simpler approach: compute phi for each number using the prime list
        # But use early termination
        for i in range(length):
            n = start + i
            result = n
            y = n
            for p in primes:
                if p * p > y:
                    break
                if y % p == 0:
                    result -= result // p
                    while y % p == 0:
                        y //= p
            if y > 1:
                result -= result // y
            results[i] = result
        
        return results
    
    # Step 4: Search for long runs of consecutive integers with distinct totients
    best_start = 1
    best_len = 1
    
    # Search strategy:
    # 1. Try structured high ranges where numbers are large
    # 2. Use random sampling in high ranges
    # 3. Focus on ranges where we can quickly verify distinctness
    
    random.seed(42)
    
    # Generate candidate starting points
    candidates = []
    
    # Structured ranges in high territory
    for base in [10**11, 2*10**11, 3*10**11, 5*10**11, 7*10**11, 9*10**11]:
        for offset in range(0, 10**6, 5000):
            candidates.append(base + offset)
    
    # Random high ranges
    for _ in range(100):
        s = random.randint(10**10, 10**12 - 200)
        candidates.append(s)
    
    # Sort and deduplicate
    candidates = sorted(set(candidates))
    
    # Process candidates
    for start in candidates:
        if time.time() > deadline:
            break
        
        if start + 200 > 10**12:
            continue
        
        # Quick check: compute phi for up to 200 consecutive numbers
        # and check for distinctness
        seen = set()
        k = 0
        for i in range(200):
            n = start + i
            if n > 10**12:
                break
            v = phi(n)
            if v in seen:
                break
            seen.add(v)
            k += 1
        
        if k > best_len:
            best_len = k
            best_start = start
    
    # If we have time, do a more thorough search around the best found so far
    remaining_time = deadline - time.time()
    if remaining_time > 30 and best_len > 1:
        # Try to extend or find better runs near the current best
        search_window = min(10**5, int(remaining_time * 1000))
        for offset in range(-search_window, search_window + 1, 100):
            new_start = best_start + offset
            if new_start < 1 or new_start + 200 > 10**12:
                continue
            if time.time() > deadline - 5:
                break
            
            seen = set()
            k = 0
            for i in range(200):
                n = new_start + i
                if n > 10**12:
                    break
                v = phi(n)
                if v in seen:
                    break
                seen.add(v)
                k += 1
            
            if k > best_len:
                best_len = k
                best_start = new_start
    
    return {'start': best_start, 'length': best_len}
