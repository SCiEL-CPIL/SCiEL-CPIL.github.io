"""Program to minimize distinct prime factors across pairwise sums."""
import random
import math
from itertools import combinations

def _sieve_smallest_prime(limit):
    """Return list spf where spf[i] is smallest prime factor of i."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _prime_factors(n, spf):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def _count_distinct_primes(arr, spf):
    """Count distinct primes dividing all pairwise sums of arr."""
    prime_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                prime_set.add(p)
                while n % p == 0:
                    n //= p
    return len(prime_set)

def _count_bad_primes(arr, spf, target_primes):
    """Count distinct primes NOT in target_primes dividing all pairwise sums of arr."""
    bad_set = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            s = arr[i] + arr[j]
            n = s
            while n > 1:
                p = spf[n]
                if p not in target_primes:
                    bad_set.add(p)
                while n % p == 0:
                    n //= p
    return len(bad_set)

def _generate_smooth_numbers(primes, limit, count):
    """Generate up to 'count' distinct numbers <= limit that are smooth w.r.t 'primes'."""
    nums = set()
    # Simple DFS/BFS to generate smooth numbers
    # Start with 1
    current = [1]
    while current and len(nums) < count:
        next_level = []
        for x in current:
            for p in primes:
                y = x * p
                if y <= limit and y not in nums:
                    nums.add(y)
                    next_level.append(y)
        current = next_level
    return list(nums)

def construct():
    n = 12
    max_val = 10000
    
    # Precompute smallest prime factor up to 20000 (max sum of two <= 10000)
    limit = 2 * max_val
    spf = _sieve_smallest_prime(limit)
    
    # Small primes list
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    
    best_arr = None
    best_score = float('inf')
    
    # Strategy: Try to find a solution using a small set of primes.
    # Iterate over possible sets of primes of size k=2, 3, 4...
    
    # We will try to minimize the number of "bad" primes for a specific target set.
    # If we reach 0 bad primes, the score is the size of the target set.
    
    target_sets = []
    # Size 2
    for combo in combinations(small_primes, 2):
        target_sets.append(set(combo))
    # Size 3
    for combo in combinations(small_primes, 3):
        target_sets.append(set(combo))
    # Size 4
    for combo in combinations(small_primes, 4):
        target_sets.append(set(combo))
        
    # Sort target sets by size, then by sum of primes (smaller is usually easier/better)
    target_sets.sort(key=lambda s: (len(s), sum(s)))
    
    random.seed(42)
    
    for target_primes in target_sets:
        # If we already have a score better than len(target_primes), we can skip larger sets
        # But we might find a better arrangement for a slightly larger set? No, score is just count.
        # If we found a solution with score 3, any set of size 4 is worse or equal.
        # So we can break if we find a solution with score K, and we are trying sets of size >= K.
        # However, we iterate by size. So if we find a solution for size 2, we return immediately.
        
        k = len(target_primes)
        if best_score <= k:
            break
            
        # Generate candidate numbers from smooth numbers for this target set
        # We need 12 distinct numbers.
        # If there aren't enough smooth numbers, we can't use this set exclusively?
        # Actually, the numbers themselves don't have to be smooth, only their sums.
        # But starting with smooth numbers is a good heuristic.
        
        smooth_nums = _generate_smooth_numbers(list(target_primes), max_val, 50)
        
        if len(smooth_nums) < n:
            # Not enough smooth numbers to start with, but we can still try random numbers
            # or just skip this set if it's too restrictive?
            # Let's just use random numbers in [1, max_val] as initial population if needed.
            # But a better initial guess is better.
            # If we can't get 12 smooth numbers, this target set might be hard.
            # Let's try anyway with random initialization.
            init_arr = random.sample(range(1, max_val + 1), n)
        else:
            # Pick 12 distinct from smooth_nums
            # We want to avoid 1 if possible? 1 is smooth.
            init_arr = random.sample(smooth_nums, n)
            
        # Local search to minimize bad primes
        current_arr = init_arr
        current_bad = _count_bad_primes(current_arr, spf, target_primes)
        
        # If we start with 0 bad primes, we found a solution!
        if current_bad == 0:
            best_score = k
            best_arr = current_arr
            break
            
        # Hill Climbing
        # We want to reduce current_bad.
        # If current_bad == 0, we are done.
        
        improved = True
        max_iter = 5000
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            for i in range(n):
                old_val = current_arr[i]
                # Try replacing current_arr[i] with a random value
                # Bias towards smooth numbers?
                # Let's try a mix: 50% random from smooth, 50% random from 1..max_val
                
                # Generate a few candidates
                candidates = []
                if smooth_nums:
                    # Pick random smooth numbers
                    for _ in range(5):
                        candidates.append(random.choice(smooth_nums))
                # Pick random numbers
                for _ in range(5):
                    candidates.append(random.randint(1, max_val))
                
                for new_val in candidates:
                    if new_val == old_val:
                        continue
                    if new_val in current_arr:
                        continue
                        
                    trial = list(current_arr)
                    trial[i] = new_val
                    
                    new_bad = _count_bad_primes(trial, spf, target_primes)
                    
                    if new_bad < current_bad:
                        current_bad = new_bad
                        current_arr = trial
                        improved = True
                        if current_bad == 0:
                            break
                    elif new_bad == current_bad:
                        # Accept with some probability to escape local minima?
                        # Or just accept if it's distinct?
                        # Let's stick to strict improvement for now to ensure convergence to 0 if possible.
                        # But if we get stuck, we might need random restarts.
                        pass
                        
                if current_bad == 0:
                    break
                    
        if current_bad == 0:
            best_score = k
            best_arr = current_arr
            break
        else:
            # If we didn't reach 0, check if this arrangement is better than our global best
            # The global score is the total number of distinct primes.
            # If we have 'bad' primes, the total primes is at least k + bad_count?
            # No, total primes = (primes in target set that appear) + (bad primes).
            # It's possible that not all target primes appear.
            # So we must calculate the actual score.
            actual_score = _count_distinct_primes(current_arr, spf)
            if actual_score < best_score:
                best_score = actual_score
                best_arr = current_arr

    # Fallback: if we didn't find a solution with a small prime set, 
    # run the original general local search on the best candidate found so far
    if best_arr is None:
        best_arr = list(range(1, n + 1))
        best_score = _count_distinct_primes(best_arr, spf)
    else:
        # Refine the best_arr found with a general local search to ensure it's optimal for its prime set
        # Or just run general local search from this start
        improved = True
        max_iterations = 5000
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            for i in range(n):
                old_val = best_arr[i]
                # Try neighbors
                for delta in range(-10, 11):
                    new_val = old_val + delta
                    if 1 <= new_val <= max_val and new_val != old_val and new_val not in best_arr:
                        trial = list(best_arr)
                        trial[i] = new_val
                        new_score = _count_distinct_primes(trial, spf)
                        if new_score < best_score:
                            best_score = new_score
                            best_arr = trial
                            improved = True
                            break
                if improved:
                    break

    return best_arr
