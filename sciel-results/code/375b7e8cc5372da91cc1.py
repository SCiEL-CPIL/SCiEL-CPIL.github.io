import time
import random
from math import isqrt

def _factorize(n):
    """Return the set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _is_smooth(n, prime_set):
    """Check if n is divisible only by primes in prime_set."""
    if n <= 1:
        return True
    for p in prime_set:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Strategy: Select a small set of primes P, then find 12 distinct positive
    # integers <= 10000 such that all pairwise sums are P-smooth.
    # This directly minimizes the number of distinct prime factors.
    
    # Candidate prime sets to try (small sets of small primes)
    prime_sets_to_try = [
        # Size 2
        {2, 3}, {2, 5}, {2, 7}, {3, 5}, {2, 11}, {2, 13},
        # Size 3
        {2, 3, 5}, {2, 3, 7}, {2, 3, 11}, {2, 5, 7}, {2, 3, 13},
        {3, 5, 7}, {2, 5, 11}, {2, 7, 11}, {2, 3, 17},
        # Size 4
        {2, 3, 5, 7}, {2, 3, 5, 11}, {2, 3, 7, 11}, {2, 5, 7, 11},
        {2, 3, 5, 13}, {2, 3, 5, 17}, {3, 5, 7, 11},
        # Size 5
        {2, 3, 5, 7, 11}, {2, 3, 5, 7, 13}, {2, 3, 5, 11, 13},
    ]
    
    overall_best_set = None
    overall_best_score = float('inf')
    
    def find_smooth_numbers(P, time_budget):
        """Find 12 distinct positive integers <= 10000 whose pairwise sums are P-smooth."""
        # Generate all numbers <= 10000 that are P-smooth
        smooth_numbers = []
        for n in range(1, 10001):
            if _is_smooth(n, P):
                smooth_numbers.append(n)
        
        if len(smooth_numbers) < 12:
            return None
        
        # We need to find 12 numbers from smooth_numbers such that
        # all pairwise sums are also P-smooth
        # Use backtracking with pruning
        
        best_found = None
        best_count = 13  # We want exactly 12
        
        # Try different subsets of smooth_numbers
        # First, try to find a set where all pairwise sums are P-smooth
        
        def backtrack(start, current, depth):
            nonlocal best_found, best_count
            
            if time.time() > time_budget:
                return
            
            if depth == 12:
                # Verify all pairwise sums are P-smooth
                for i in range(12):
                    for j in range(i + 1, 12):
                        s = current[i] + current[j]
                        if not _is_smooth(s, P):
                            return
                if best_found is None or len(current) < best_count:
                    best_found = list(current)
                    best_count = len(current)
                return
            
            # Pruning: if we can't reach 12, stop
            if len(smooth_numbers) - start < 12 - depth:
                return
            
            # Try adding smooth_numbers[start]
            # Check if it's compatible with all current elements
            compatible = True
            for x in current:
                if not _is_smooth(x + smooth_numbers[start], P):
                    compatible = False
                    break
            
            if compatible:
                current.append(smooth_numbers[start])
                backtrack(start + 1, current, depth + 1)
                current.pop()
            
            # Try skipping smooth_numbers[start]
            backtrack(start + 1, current, depth)
        
        # Random restarts with different orderings
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > time_budget:
                break
            
            # Shuffle smooth_numbers for different search paths
            shuffled = list(smooth_numbers)
            random.shuffle(shuffled)
            
            # Limit the search space to improve speed
            # Focus on smaller numbers first (more likely to have smooth sums)
            shuffled.sort(key=lambda x: (x, random.random()))
            
            # Try a subset to keep search manageable
            search_space = shuffled[:min(200, len(shuffled))]
            
            # Backtrack
            current = []
            backtrack(0, current, 0)
            
            if best_found is not None:
                break
        
        return best_found
    
    # Try each prime set
    num_sets = len(prime_sets_to_try)
    time_per_set = (deadline - start_time) / num_sets
    
    for idx, P in enumerate(prime_sets_to_try):
        if time.time() > deadline - 2.0:
            break
        
        remaining = deadline - time.time() - 2.0
        if remaining < 0.5:
            break
        
        t_budget = min(time_per_set, remaining)
        
        result = find_smooth_numbers(P, t_budget)
        
        if result is not None and len(result) == 12:
            # This is optimal for this prime set
            score = len(P)
            if score < overall_best_score:
                overall_best_score = score
                overall_best_set = result
                # If we found a very small set, we can stop early
                if score <= 3:
                    break
    
    # Fallback: if no smooth set found, use greedy approach
    if overall_best_set is None:
        # Simple fallback: use multiples of a highly composite number
        # This ensures many shared factors
        base = 210  # 2 * 3 * 5 * 7
        result = [base * (i + 1) for i in range(12)]
        
        # Verify
        if len(set(result)) != 12 or any(x > 10000 for x in result):
            result = list(range(1, 13))
        
        overall_best_set = result
    
    # Final verification
    if len(set(overall_best_set)) != 12:
        return list(range(1, 13))
    if any(x > 10000 for x in overall_best_set):
        return list(range(1, 13))
    if any(x <= 0 for x in overall_best_set):
        return list(range(1, 13))
    
    return overall_best_set
