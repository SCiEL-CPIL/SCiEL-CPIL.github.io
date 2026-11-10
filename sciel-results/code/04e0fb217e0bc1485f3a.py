import time
import itertools
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    # Primes up to 40
    primes = []
    for i in range(2, P['n'] + 1):
        is_p = True
        for d in range(2, isqrt(i) + 1):
            if i % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    # Map primes to indices
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Generate vectors for all numbers 2..40
    candidates = []
    for x in range(2, P['n'] + 1):
        vec = [0] * num_primes
        temp = x
        for p, idx in prime_to_idx.items():
            while temp % p == 0:
                vec[idx] += 1
                temp //= p
        # If temp > 1, it's a prime > 40? No, x <= 40.
        # All prime factors are in primes list.
        candidates.append((x, tuple(vec)))
    
    # Helper to check if adding vec to current sums is valid
    # current_sums is a set of tuples
    def can_add(vec, current_sums):
        # Check if any s in current_sums has s - vec in current_sums
        # This corresponds to s1 + vec = s2
        # We iterate through current_sums
        for s in current_sums:
            # Calculate s - vec
            diff = tuple(s[i] - vec[i] for i in range(num_primes))
            if diff in current_sums:
                return False
        return True

    # Greedy initialization
    # Sort candidates: Primes first, then composites by complexity
    def complexity(c):
        x, vec = c
        # Count non-zero elements
        non_zero = sum(1 for v in vec if v > 0)
        return (0 if x in primes else 1, non_zero, x)
    
    sorted_candidates = sorted(candidates, key=complexity)
    
    current_set = []
    current_sums = {tuple([0] * num_primes)}
    
    for x, vec in sorted_candidates:
        if can_add(vec, current_sums):
            current_set.append(x)
            # Update sums
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + vec[i] for i in range(num_primes))
                new_sums.add(new_s)
            current_sums.update(new_sums)
        
        if time.time() > deadline:
            break

    best_set = list(current_set)
    best_size = len(best_set)
    
    # Local Search / Improvement
    # Try to add any remaining candidates
    # Try to swap
    
    # Re-evaluate remaining candidates
    remaining = [c for c in candidates if c[0] not in best_set]
    
    # Try adding one by one
    for x, vec in remaining:
        if can_add(vec, current_sums):
            current_set.append(x)
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + vec[i] for i in range(num_primes))
                new_sums.add(new_s)
            current_sums.update(new_sums)
            best_set = list(current_set)
            best_size = len(best_set)
        
        if time.time() > deadline:
            return best_set

    # Try swaps: remove one, add two?
    # This is more complex. Given time, maybe just return the greedy result.
    # But we can try a simple hill climbing:
    # Try removing a prime and adding a composite?
    
    # Let's try a more robust local search:
    # 1. Try to add any remaining. (Done above)
    # 2. Try swapping: for each i in current_set, for each j in remaining:
    #    Remove i, add j. If size increases (impossible) or same size but better?
    #    We just want max size.
    #    If we remove 1 and add 1, size stays same.
    #    If we can remove 1 and add 2, size increases.
    
    # Let's try to find a set of size > best_size.
    # We can try to remove a "heavy" vector and add two "light" ones.
    
    # For simplicity and safety, return the current best if time is running out.
    if time.time() > deadline - 5:
        return best_set

    # Try to improve by swapping
    # We'll iterate a few times
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding any remaining
        for x, vec in remaining:
            if x not in current_set:
                if can_add(vec, current_sums):
                    current_set.append(x)
                    new_sums = set()
                    for s in current_sums:
                        new_s = tuple(s[i] + vec[i] for i in range(num_primes))
                        new_sums.add(new_s)
                    current_sums.update(new_sums)
                    best_set = list(current_set)
                    best_size = len(best_set)
                    improved = True
                    # Update remaining
                    remaining = [c for c in candidates if c[0] not in current_set]
        
        if time.time() > deadline - 5:
            break
        
        # Try 1-for-2 swaps
        # Remove one from current, add two from remaining
        # This is O(N^3) potentially.
        # N ~ 18, Remaining ~ 20.
        # 18 * 20 * 20 = 7200 checks.
        # Each check is O(2^18) ~ 260k.
        # 7200 * 260k is too slow.
        
        # Instead, we can just return best_set.
        # The greedy with primes first usually gets a good set.
        break

    return best_set

# EVOLVE-BLOCK-END
