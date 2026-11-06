"""Rewritten program for few prime factors across pairwise sums."""
import time
import random
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START

def _prime_factors(n):
    """Return the set of distinct prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _count_distinct_primes(arr):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    prime_set = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            prime_set |= _prime_factors(arr[i] + arr[j])
    return len(prime_set)


def _try_replace(arr, primes, candidate_pool, current_count):
    """
    Try to replace one element of arr with one from candidate_pool to reduce prime count.
    Returns (new_arr, new_count) if improved, else (None, None).
    """
    n = len(arr)
    # For each position, try replacing with a candidate
    # We compute the effect: removing arr[i] eliminates its sums, adding candidate adds new sums
    # To be efficient, we precompute for each i the set of primes contributed by arr[i] with others
    # But that's expensive. Instead, we just try a limited set of candidates per position.
    
    # Shuffle candidates to randomize
    candidates = list(candidate_pool)
    random.shuffle(candidates)
    
    best_arr = None
    best_count = current_count
    
    # Try each position
    for i in range(n):
        old_val = arr[i]
        # Try a subset of candidates
        for cand in candidates[:200]:
            if cand == old_val:
                continue
            if cand in arr:
                continue
            new_arr = arr[:i] + [cand] + arr[i+1:]
            # Check distinctness
            if len(set(new_arr)) != n:
                continue
            # Count primes
            cnt = _count_distinct_primes(new_arr)
            if cnt < best_count:
                best_count = cnt
                best_arr = new_arr
                if best_count <= 3:
                    return best_arr, best_count
    return best_arr, best_count


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    # Initial setup
    random.seed(42)
    
    # Candidate pool: numbers 1..10000
    # Focus on small numbers and numbers with smooth sums
    # A good heuristic: numbers that are multiples of small primes tend to have sums with small prime factors
    # Let's build a candidate pool biased toward numbers with many small prime factors
    # Actually, we want the SUMS to have few distinct primes.
    # If we pick all numbers congruent to 0 mod 2, sums are even -> prime 2
    # If we pick all numbers congruent to 0 mod 3, sums are divisible by 3 -> prime 3
    # If we pick numbers that are 0 mod 6, sums are 0 mod 6 -> primes 2, 3
    # So a set of numbers all divisible by 6 would have sums divisible by 6, giving primes {2,3} at minimum
    # But sums could have other primes too.
    
    # Strategy: Start with a pool of numbers that are multiples of small numbers
    # Then do local search
    
    # Build initial candidate pool
    # Numbers divisible by 2, 3, 5, 6, 10, 15, 30, etc.
    # Let's use numbers 1..200 as initial pool for speed, then expand if needed
    
    # Phase 1: Greedy construction
    # Start with a small set and greedily add numbers that minimize new primes
    
    # Let's try multiple random starts
    
    best_result = None
    best_count = float('inf')
    
    # Candidate pool: focus on numbers <= 1000 with some structure
    # Numbers that are multiples of 6: 6, 12, 18, ...
    # Numbers that are multiples of 30: 30, 60, 90, ...
    # Let's create a diverse pool
    
    pool = set()
    # Add multiples of small numbers
    for m in [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]:
        for k in range(1, 10000 // m + 1):
            pool.add(m * k)
    # Also add some small numbers
    for i in range(1, 200):
        pool.add(i)
    
    pool = sorted(pool)
    pool_set = set(pool)
    
    # Try multiple random initial sets
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        # Create initial set
        if start_idx == 0:
            # Start with multiples of 6
            init_set = [6 * i for i in range(1, 13)]
        elif start_idx == 1:
            # Start with multiples of 30
            init_set = [30 * i for i in range(1, 13)]
        elif start_idx == 2:
            # Start with small numbers
            init_set = list(range(1, 13))
        elif start_idx == 3:
            # Start with even numbers
            init_set = [2 * i for i in range(1, 13)]
        elif start_idx == 4:
            # Start with multiples of 12
            init_set = [12 * i for i in range(1, 13)]
        elif start_idx == 5:
            # Start with multiples of 24
            init_set = [24 * i for i in range(1, 13)]
        else:
            # Random start from pool
            init_set = random.sample(pool, 12)
        
        # Ensure distinct and <= 10000
        init_set = sorted(set(x for x in init_set if 1 <= x <= 10000))
        if len(init_set) < 12:
            # Fill with random from pool
            remaining = [x for x in pool if x not in set(init_set)]
            random.shuffle(remaining)
            for x in remaining:
                if len(init_set) >= 12:
                    break
                init_set.append(x)
            init_set = sorted(init_set[:12])
        
        current_arr = list(init_set)
        current_count = _count_distinct_primes(current_arr)
        
        if current_count < best_count:
            best_count = current_count
            best_result = current_arr[:]
        
        # Local search: try to improve
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            new_arr, new_count = _try_replace(current_arr, None, pool, current_count)
            if new_arr is not None:
                current_arr = new_arr
                current_count = new_count
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_result = current_arr[:]
                if best_count <= 3:
                    break
        
        if best_count <= 3:
            break
    
    # Phase 2: If we haven't found a good solution, try more aggressive search
    if best_count > 4 and time.time() < deadline - 10.0:
        # Try to find a set where all pairwise sums are divisible by a common small prime
        # For example, all numbers even -> sums even -> prime 2
        # All numbers odd -> sums even -> prime 2
        # All numbers congruent to 0 mod 3 -> sums divisible by 3
        # All numbers congruent to 1 mod 3 -> sums congruent to 2 mod 3, not divisible by 3
        # So we need numbers all in the same residue class mod p for prime p to divide all sums
        
        # Try: all numbers even
        evens = [2 * i for i in range(1, 5000)]
        if len(evens) >= 12:
            test_set = evens[:12]
            cnt = _count_distinct_primes(test_set)
            if cnt < best_count:
                best_count = cnt
                best_result = test_set[:]
        
        # Try: all numbers divisible by 3
        threes = [3 * i for i in range(1, 3334)]
        if len(threes) >= 12:
            test_set = threes[:12]
            cnt = _count_distinct_primes(test_set)
            if cnt < best_count:
                best_count = cnt
                best_result = test_set[:]
        
        # Try: all numbers divisible by 6
        sixes = [6 * i for i in range(1, 1667)]
        if len(sixes) >= 12:
            test_set = sixes[:12]
            cnt = _count_distinct_primes(test_set)
            if cnt < best_count:
                best_count = cnt
                best_result = test_set[:]
        
        # Try: all numbers divisible by 30
        thirties = [30 * i for i in range(1, 334)]
        if len(thirties) >= 12:
            test_set = thirties[:12]
            cnt = _count_distinct_primes(test_set)
            if cnt < best_count:
                best_count = cnt
                best_result = test_set[:]
    
    # Final check: ensure validity
    if best_result is None:
        best_result = list(range(1, 13))
    
    # Verify distinctness and bounds
    best_result = sorted(set(x for x in best_result if 1 <= x <= 10000))
    if len(best_result) < 12:
        # Fill with valid numbers
        existing = set(best_result)
        for i in range(1, 10001):
            if len(best_result) >= 12:
                break
            if i not in existing:
                best_result.append(i)
        best_result = sorted(best_result[:12])
    
    return best_result

# EVOLVE-BLOCK-END
