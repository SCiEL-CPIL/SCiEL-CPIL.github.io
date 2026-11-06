"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_L(residues, primes, max_L=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    # For each x in 1..max_L, count how many congruences x satisfies
    # x satisfies congruence i if x % primes[i] == residues[i]
    # We need count >= 2 for all x in 1..L
    
    # Build a counter array
    # For each prime p with residue r, the numbers that satisfy x ≡ r (mod p) are r, r+p, r+2p, ...
    # But we only care about x in 1..max_L
    # Note: if r == 0, the numbers are p, 2p, 3p, ... (not 0, since we start from 1)
    
    count = [0] * (max_L + 1)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        for x in range(start, max_L + 1, p):
            count[x] += 1
    
    # Find the largest L such that count[x] >= 2 for all x in 1..L
    L = 0
    for x in range(1, max_L + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _compute_L_fast(residues, primes, max_L=100000):
    """Faster computation using numpy-like approach with list operations."""
    count = [0] * (max_L + 1)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        # Increment count for all x ≡ r (mod p) in [1, max_L]
        for x in range(start, max_L + 1, p):
            count[x] += 1
    
    L = 0
    for x in range(1, max_L + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    
    # Initial seed: the i-th prime covers the integer ceil(i/2)
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    current_L = _compute_L(residues, primes, max_L=100000)
    best_L = current_L
    best_residues = list(residues)
    
    # Local search: try swapping residues to extend the doubly-covered prefix
    random.seed(42)
    
    # Focus on the bottleneck: find the first x where coverage < 2
    # Then try to change residues of primes that don't cover x to cover x
    # while not breaking coverage of other points
    
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 5.0:
            break
        
        iteration += 1
        
        # Find the current bottleneck
        count = [0] * (best_L + 1)
        for i, (r, p) in enumerate(zip(best_residues, primes)):
            if r == 0:
                start = p
            else:
                start = r
            if start > best_L:
                continue
            for x in range(start, best_L + 1, p):
                count[x] += 1
        
        # Find first x in 1..best_L where count[x] < 2, or best_L+1 if all covered
        bottleneck = best_L + 1
        for x in range(1, best_L + 1):
            if count[x] < 2:
                bottleneck = x
                break
        
        # If we can extend beyond best_L, try to improve
        # Strategy: pick a random prime and try changing its residue
        # to cover the bottleneck point while maintaining coverage elsewhere
        
        # Try a few random modifications
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = best_residues[idx]
            
            # New residue: try to make this prime cover the bottleneck
            # x ≡ r (mod p) means r = x % p
            new_r = bottleneck % p
            
            if new_r == old_r:
                continue
            
            # Test the modification
            test_residues = list(best_residues)
            test_residues[idx] = new_r
            
            # Compute new L (limit to a reasonable range for speed)
            # We only need to check up to best_L + some margin
            test_max_L = min(best_L + 200, 100000)
            test_count = [0] * (test_max_L + 1)
            for i, (r, pp) in enumerate(zip(test_residues, primes)):
                if r == 0:
                    start = pp
                else:
                    start = r
                if start > test_max_L:
                    continue
                for x in range(start, test_max_L + 1, pp):
                    test_count[x] += 1
            
            test_L = 0
            for x in range(1, test_max_L + 1):
                if test_count[x] >= 2:
                    test_L = x
                else:
                    break
            
            if test_L > best_L:
                best_L = test_L
                best_residues = test_residues
                break
        
        # Occasionally do a full recompute to verify
        if iteration % 100 == 0:
            verified_L = _compute_L(best_residues, primes, max_L=100000)
            if verified_L < best_L:
                best_L = verified_L
    
    return best_residues

# EVOLVE-BLOCK-END
