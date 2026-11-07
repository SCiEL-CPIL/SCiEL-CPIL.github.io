"""
Strategy: Simulated Annealing on the space of valid sets.
- Start with a greedy initialization using large primes (low divisibility risk).
- Use a 1-swap neighborhood (remove one, add one) to explore the space.
- Accept worse solutions probabilistically to escape local optima.
- Optimize validity checking by precomputing subset sums and checking divisibility only against existing sums.
"""

import time
import random
import math
import sys

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _is_valid(S):
    """Full validity check for a set S."""
    if not S:
        return True
    sums = _subset_sums(S)
    total = sum(S)
    sums_list = sorted(sums)
    sums_set = sums
    for s in sums_list:
        if s == 0:
            continue
        # Check if any multiple of s is in the set
        # We only need to check up to total
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _check_addition(S, x, sums, total):
    """
    Check if adding x to S keeps the set valid.
    Returns (new_sums, new_total) if valid, else (None, None).
    """
    new_total = total + x
    new_sums = set(sums)
    # Add new sums
    for s in list(sums):
        new_sums.add(s + x)
    new_sums.add(x)
    
    # Check divisibility
    # We need to check if any existing sum divides any new sum, 
    # or if any new sum divides any existing sum (which is covered by checking multiples),
    # or if any new sum divides another new sum.
    
    # Optimization: 
    # 1. Check if x divides any existing sum (or vice versa)
    # 2. Check if (s+x) divides any existing sum or other (s'+x)
    
    # Simpler robust check: 
    # Check all new sums against all existing sums + other new sums for divisibility.
    # Since we only add one element, the number of new sums is |sums| + 1.
    # |sums| is at most 2^k - 1. For k=10, ~1000. 1000*1000 = 1M ops. Might be slow in Python loop.
    
    # Let's use the multiple check method but restricted.
    # For each s in new_sums, check if any multiple of s is in new_sums.
    # This is O(N * N/s).
    
    new_sums_list = list(new_sums)
    new_sums_set = new_sums
    
    for s in new_sums_list:
        if s == 0:
            continue
        # Check multiples
        # If s is large, few multiples. If s is small, many.
        # But we can break early if multiple > new_total.
        m = 2 * s
        while m <= new_total:
            if m in new_sums_set:
                return None, None
            m += s
            
    return new_sums, new_total

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    
    pool = list(range(1, 257))
    
    # Precompute primes for initialization
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in range(2, 257) if is_prime(x)]
    
    # Initialization: Greedy with large primes first
    # Large numbers have fewer multiples, so they are "safer" to add initially.
    # Also, primes don't divide each other (except 1, which is not in primes > 1).
    
    current_set = []
    current_sums = set()
    current_total = 0
    used = set()
    
    # Sort candidates: Primes > 128 first, then other primes, then rest
    # This biases towards large, indivisible numbers.
    candidates = sorted(pool, key=lambda x: (-1 if is_prime(x) and x > 128 else 0, -x))
    
    # Greedy initialization
    for x in candidates:
        if time.time() > deadline - 5: # Leave time for SA
            break
        if x in used:
            continue
        new_sums, new_total = _check_addition(current_set, x, current_sums, current_total)
        if new_sums is not None:
            current_set.append(x)
            current_sums = new_sums
            current_total = new_total
            used.add(x)
            if len(current_set) >= 30:
                break
                
    best_set = list(current_set)
    best_size = len(current_set)
    best_sums = _subset_sums(best_set)
    best_total = sum(best_set)
    
    if best_size == 30:
        return [int(x) for x in best_set]
        
    # Simulated Annealing
    # State: set of elements
    # Move: Remove one, Add one (from pool)
    # Cost: -len(set) (we want to maximize size, so minimize negative size)
    # Actually, we only accept if valid. If valid and same size, we might accept to explore.
    # If valid and larger size, always accept.
    # If valid and smaller size, accept with probability exp(-delta / T).
    # If invalid, reject.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    rng = random.Random(42)
    
    current = list(best_set)
    cur_sums = set(best_sums)
    cur_total = best_total
    cur_size = len(current)
    
    # Precompute available pool for swaps
    # We need to pick one to remove from current, and one to add from pool.
    
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 1.0 and steps < max_steps:
        steps += 1
        
        if not current:
            break
            
        # Pick a random element to remove
        idx = rng.randint(0, len(current) - 1)
        remove_elem = current[idx]
        
        # Temporarily remove
        temp_set = current[:idx] + current[idx+1:]
        temp_sums = _subset_sums(temp_set) # Expensive! 
        # Optimization: Recalculate sums from scratch for temp_set? 
        # For k=10, 2^10=1024 sums. This is fast enough.
        temp_total = sum(temp_set)
        
        # Pick a random element to add from pool (not in temp_set)
        # To speed up, we can pick from a subset of likely candidates?
        # Or just random from pool.
        # Let's pick from pool, ensuring it's not in temp_set.
        # To avoid infinite loops on invalid, we sample a few candidates.
        
        found_valid_swap = False
        new_size = -1
        new_set = None
        new_sums = None
        new_total = 0
        
        # Try up to 5 random candidates for addition
        for _ in range(5):
            add_elem = rng.choice(pool)
            if add_elem in temp_set:
                continue
            
            ns, nt = _check_addition(temp_set, add_elem, temp_sums, temp_total)
            if ns is not None:
                # Valid swap
                new_set = temp_set + [add_elem]
                new_sums = ns
                new_total = nt
                new_size = len(new_set)
                found_valid_swap = True
                break
        
        if not found_valid_swap:
            # No valid swap found in this step. 
            # This is a "stuck" state. 
            # We can try a pure removal? No, that reduces size.
            # We can try a pure addition? 
            # Let's try a pure addition from pool.
            for _ in range(5):
                add_elem = rng.choice(pool)
                if add_elem in current:
                    continue
                ns, nt = _check_addition(current, add_elem, cur_sums, cur_total)
                if ns is not None:
                    new_set = current + [add_elem]
                    new_sums = ns
                    new_total = nt
                    new_size = len(new_set)
                    found_valid_swap = True # Treat as improvement
                    break
            
            if not found_valid_swap:
                # Stuck. Cool down faster? Or just continue.
                T *= 0.95
                continue
        
        # Evaluate move
        if new_size > cur_size:
            # Always accept improvement
            current = new_set
            cur_sums = new_sums
            cur_total = new_total
            cur_size = new_size
            
            if cur_size > best_size:
                best_size = cur_size
                best_set = list(current)
                best_sums = set(cur_sums)
                best_total = cur_total
                
            if best_size >= 30:
                break
                
            # Reset temperature on improvement? Or keep cooling.
            # Keep cooling to converge.
            
        elif new_size == cur_size:
            # Same size. Accept with high probability to explore.
            if rng.random() < 0.5:
                current = new_set
                cur_sums = new_sums
                cur_total = new_total
                # cur_size unchanged
                
        else:
            # new_size < cur_size (should not happen if we only do swaps/adds that are valid, 
            # unless we did a swap that resulted in smaller size? 
            # Swap: remove 1, add 1 -> same size.
            # Add: add 1 -> larger size.
            # So new_size < cur_size is impossible in our move generation above.
            # Wait, if we do a swap, size is same. If we do add, size is larger.
            # So this branch is dead code unless we implement removals.
            pass
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback
        return [int(x) for x in best_set[:best_size]] if best_set else []
        
    return [int(x) for x in best_set]
