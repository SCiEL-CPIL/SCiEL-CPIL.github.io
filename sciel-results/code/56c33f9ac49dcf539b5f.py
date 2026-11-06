"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

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
    max_L = min(max_L, 100000)
    count = np.zeros(max_L + 1, dtype=np.int16)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        idx = np.arange(start - 1, max_L, p)
        count[idx] += 1
    # Find the largest L such that count[x] >= 2 for all x in 1..L
    valid = count[1:] >= 2
    if not np.any(valid):
        return 0
    first_false = np.argmax(~valid)
    if valid[first_false]:
        return max_L
    return first_false

def _compute_L_range(residues, primes, start, end):
    """Compute coverage count for range [start, end] and return first x where count < 2."""
    length = end - start + 1
    count = np.zeros(length, dtype=np.int16)
    for i, (r, p) in enumerate(zip(residues, primes)):
        if r == 0:
            first = p
        else:
            first = r
        # Find first x >= start such that x ≡ r (mod p)
        if first < start:
            k = (start - first + p - 1) // p
            first += k * p
        if first > end:
            continue
        idx = (np.arange(first, end + 1, p) - start)
        count[idx] += 1
    # Find first index where count < 2
    valid = count >= 2
    if np.all(valid):
        return end + 1
    first_false = np.argmax(~valid)
    return start + first_false

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    primes_arr = np.array(primes)
    
    # Initial seed: try to cover small integers densely
    # Strategy: assign residues so that small numbers get multiple hits
    residues = [0] * n_primes
    
    # Greedy initialization: for each prime, choose residue to cover a number
    # that currently has the lowest coverage
    # Start with all zeros, then improve
    
    # Better initial: use the fact that we want to cover 1..L
    # Assign residues based on a pattern that maximizes early coverage
    for i, p in enumerate(primes):
        # Choose residue to cover a small number
        # Try to cover numbers that are multiples of small primes
        residues[i] = 1 % p  # Cover 1
    
    current_L = _compute_L(residues, primes, max_L=100000)
    best_L = current_L
    best_residues = list(residues)
    
    # Phase 1: Greedy improvement - for each prime, try all residues and pick best
    random.seed(42)
    
    # First, do a few passes of greedy assignment
    for pass_num in range(3):
        if time.time() > deadline - 20.0:
            break
        improved = False
        for idx in range(n_primes):
            if time.time() > deadline - 20.0:
                break
            p = primes[idx]
            best_r = best_residues[idx]
            best_local_L = best_L
            
            # Try a subset of residues (not all, for speed)
            # Focus on residues that cover numbers near the current bottleneck
            candidates = set()
            candidates.add(best_r)
            # Add residues that cover numbers in [1, best_L + 100]
            for x in range(1, min(best_L + 50, 500)):
                candidates.add(x % p)
            # Add some random candidates
            for _ in range(10):
                candidates.add(random.randint(0, p - 1))
            
            for r in candidates:
                if r == best_r:
                    continue
                test_residues = list(best_residues)
                test_residues[idx] = r
                test_L = _compute_L(test_residues, primes, max_L=min(best_L + 200, 100000))
                if test_L > best_local_L:
                    best_local_L = test_L
                    best_r = r
            
            if best_r != best_residues[idx]:
                best_residues[idx] = best_r
                best_L = best_local_L
                improved = True
        
        if not improved:
            break
    
    # Phase 2: Local search targeting the bottleneck
    max_iterations = 20000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline - 10.0:
            break
        
        iteration += 1
        
        # Find the current bottleneck: first x where coverage < 2
        # Check up to best_L + 1000
        check_end = min(best_L + 1000, 100000)
        bottleneck = _compute_L_range(best_residues, primes, 1, check_end)
        
        # If bottleneck > best_L, we've extended
        if bottleneck > best_L:
            best_L = bottleneck
            # Recompute to verify
            verified_L = _compute_L(best_residues, primes, max_L=100000)
            if verified_L < best_L:
                best_L = verified_L
            continue
        
        # Try to fix the bottleneck by changing a residue
        # The bottleneck is at position `bottleneck`
        # We need to find a prime whose residue can be changed to cover `bottleneck`
        # without breaking coverage of other points
        
        # Strategy: pick a prime that doesn't currently cover the bottleneck
        # and change its residue to cover it
        
        # First, find which primes cover the bottleneck
        covers_bottleneck = []
        for i, (r, p) in enumerate(zip(best_residues, primes)):
            if bottleneck % p == r:
                covers_bottleneck.append(i)
        
        # We need at least one more prime to cover the bottleneck
        # Try changing a prime that doesn't cover it
        
        # Pick a random prime that doesn't cover the bottleneck
        non_covers = [i for i in range(n_primes) if i not in covers_bottleneck]
        
        if non_covers:
            # Try a few random modifications
            for _ in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                idx = random.choice(non_covers)
                p = primes[idx]
                old_r = best_residues[idx]
                new_r = bottleneck % p
                
                if new_r == old_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[idx] = new_r
                
                # Quick check: does this improve the bottleneck?
                test_check_end = min(best_L + 200, 100000)
                test_bottleneck = _compute_L_range(test_residues, primes, 1, test_check_end)
                
                if test_bottleneck > best_L:
                    best_residues = test_residues
                    best_L = test_bottleneck
                    # Verify
                    verified_L = _compute_L(best_residues, primes, max_L=100000)
                    if verified_L < best_L:
                        best_L = verified_L
                    break
        
        # Occasionally try a different strategy: change multiple residues
        if iteration % 50 == 0:
            if time.time() > deadline - 10.0:
                break
            
            # Try changing 2 residues simultaneously
            for _ in range(2):
                idx1 = random.randint(0, n_primes - 1)
                idx2 = random.randint(0, n_primes - 1)
                if idx1 == idx2:
                    continue
                
                p1 = primes[idx1]
                p2 = primes[idx2]
                
                # Choose new residues to cover the bottleneck
                new_r1 = bottleneck % p1
                new_r2 = bottleneck % p2
                
                test_residues = list(best_residues)
                test_residues[idx1] = new_r1
                test_residues[idx2] = new_r2
                
                test_check_end = min(best_L + 200, 100000)
                test_bottleneck = _compute_L_range(test_residues, primes, 1, test_check_end)
                
                if test_bottleneck > best_L:
                    best_residues = test_residues
                    best_L = test_bottleneck
                    verified_L = _compute_L(best_residues, primes, max_L=100000)
                    if verified_L < best_L:
                        best_L = verified_L
                    break
    
    # Final verification
    final_L = _compute_L(best_residues, primes, max_L=100000)
    if final_L < best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
