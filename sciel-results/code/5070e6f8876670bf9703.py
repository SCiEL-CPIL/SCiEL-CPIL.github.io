"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill

    def prime_factors(n):
        """Return set of prime factors of n."""
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

    def count_distinct_primes_in_sums(nums):
        """Count distinct primes across all pairwise sums."""
        primes = set()
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                s = nums[i] + nums[j]
                primes.update(prime_factors(s))
        return len(primes)

    def greedy_clique_with_prime_set(prime_set, max_val, target_size, deadline):
        """
        Greedily build a clique where all pairwise sums have prime factors within prime_set.
        Returns the clique if found, else None.
        """
        # First, identify all numbers whose potential sums with other candidates
        # would only use primes from prime_set
        # A number x is "compatible" with y if all prime factors of (x+y) are in prime_set
        
        # Strategy: iterate through numbers 1..max_val, maintain a candidate list
        # For each new number, check if it's compatible with all already-selected numbers
        
        selected = []
        blocking_primes = Counter()  # Track which primes block us
        
        for x in range(1, max_val + 1):
            if time.time() > deadline:
                return None
            
            compatible = True
            for y in selected:
                s = x + y
                factors = prime_factors(s)
                for p in factors:
                    if p not in prime_set:
                        compatible = False
                        blocking_primes[p] += 1
                        break
                if not compatible:
                    break
            
            if compatible:
                selected.append(x)
                if len(selected) >= target_size:
                    return sorted(selected)
        
        return None if len(selected) < target_size else sorted(selected)

    # Adaptive prime set expansion
    # Start with small prime sets, expand if stuck
    initial_prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    best_result = None
    best_score = float('inf')
    
    for base_primes in initial_prime_sets:
        if time.time() > deadline - 10:
            break
        
        # Try increasing max_val
        for max_val in [200, 500, 1000, 2000, 5000]:
            if time.time() > deadline - 5:
                break
            
            clique = greedy_clique_with_prime_set(set(base_primes), max_val, 12, deadline - 2)
            
            if clique is not None:
                score = count_distinct_primes_in_sums(clique)
                if score < best_score:
                    best_score = score
                    best_result = clique
                    if score == len(base_primes):
                        return best_result
    
    # If we haven't found a good result, try a different approach:
    # Local search starting from a smooth-number seed
    if time.time() < deadline - 15:
        # Generate 2,3,5-smooth numbers up to 10000
        smooth_nums = []
        for a in range(14):  # 2^13 = 8192
            for b in range(14):  # 3^13 > 10000
                for c in range(14):  # 5^13 > 10000
                    n = (2 ** a) * (3 ** b) * (5 ** c)
                    if 1 <= n <= 10000:
                        smooth_nums.append(n)
        smooth_nums = sorted(set(smooth_nums))
        
        if len(smooth_nums) >= 12:
            # Start with first 12, then try local search to minimize distinct primes
            current = sorted(smooth_nums[:12])
            current_score = count_distinct_primes_in_sums(current)
            
            if current_score < best_score:
                best_score = current_score
                best_result = current[:]
            
            # Simple local search: swap elements to reduce distinct primes
            for iteration in range(500):
                if time.time() > deadline - 3:
                    break
                
                improved = False
                for i in range(12):
                    for j in range(12):
                        if i == j:
                            continue
                        # Try swapping current[i] with a random smooth number
                        for new_val in smooth_nums:
                            if new_val in current:
                                continue
                            
                            trial = current[:]
                            trial[i] = new_val
                            trial.sort()
                            
                            # Quick check: does this reduce distinct primes?
                            trial_score = count_distinct_primes_in_sums(trial)
                            if trial_score < current_score:
                                current = trial
                                current_score = trial_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_result = current[:]
    
    if best_result is not None:
        return best_result
    
    # Final fallback
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# EVOLVE-BLOCK-END
