"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 12
    MAXV = 10000

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

    def generate_smooth_numbers(primes, limit):
        """Generate all numbers <= limit whose prime factors are in primes."""
        smooth = [1]
        for p in primes:
            new_smooth = []
            for x in smooth:
                v = x
                while v * p <= limit:
                    v *= p
                    new_smooth.append(v)
            smooth.extend(new_smooth)
        return sorted(set(smooth))

    def find_clique_smooth(prime_set, limit, target_size, deadline):
        """
        Find a clique of size target_size in the graph of smooth numbers
        where edges connect x, y if x+y is also smooth (using the same prime set).
        """
        # 1. Generate candidates: numbers that are smooth AND whose double is smooth?
        # Actually, we just need x+y to be smooth.
        # Candidates are smooth numbers.
        smooth_nums = generate_smooth_numbers(prime_set, limit)
        if len(smooth_nums) < target_size:
            return None
        
        # Precompute smooth set for O(1) lookup
        smooth_set = set(smooth_nums)
        
        # Candidates must be such that they can form sums with others.
        # We search for a clique in the graph G=(V, E) where V=smooth_nums
        # and (x,y) in E if x+y in smooth_set.
        
        # Backtracking search
        best_clique = []
        
        def backtrack(current, start_idx):
            nonlocal best_clique
            if time.time() > deadline:
                return False
            if len(current) == target_size:
                best_clique = list(current)
                return True
            
            # Heuristic: try to pick numbers that have high degree to remaining candidates?
            # For simplicity, iterate through smooth_nums from start_idx
            for i in range(start_idx, len(smooth_nums)):
                x = smooth_nums[i]
                # Check if x is compatible with all in current
                compatible = True
                for y in current:
                    if (x + y) not in smooth_set:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                current.append(x)
                if backtrack(current, i + 1):
                    return True
                current.pop()
                
                # Pruning: if we can't fill the rest, stop?
                # Simple bound: need target_size - len(current) more nodes.
                # If remaining nodes < needed, break.
                if len(smooth_nums) - (i + 1) < target_size - len(current):
                    break
                    
            return False

        backtrack([], 0)
        if best_clique:
            return best_clique
        return None

    # Strategy: Try increasing prime sets
    # Start with smallest prime sets to minimize the count of distinct primes.
    # If we find a clique for S, the score is |S|.
    
    prime_candidates = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]
    
    best_result = None
    best_score = float('inf')
    
    for p_set in prime_candidates:
        if time.time() > deadline - 5:
            break
            
        # Try to find a clique of size 12 using this prime set
        # We search up to MAXV
        clique = find_clique_smooth(p_set, MAXV, N, deadline - 2)
        
        if clique is not None:
            # Verify the score
            score = count_distinct_primes_in_sums(clique)
            if score < best_score:
                best_score = score
                best_result = clique
                # If we achieved the theoretical minimum for this set, we can stop
                # The theoretical minimum is len(p_set) if the clique exists.
                # But the actual score might be higher if some sums introduce other primes?
                # No, by definition of the clique search, all sums are smooth w.r.t p_set.
                # So the prime factors are a subset of p_set.
                # Thus score <= len(p_set).
                if score == len(p_set):
                    return clique
                # Even if score > len(p_set) (which shouldn't happen if logic is correct),
                # we continue to see if a larger set yields a smaller score?
                # No, larger set means larger minimum possible score.
                # So if we found a valid clique for S, the best we can do with S is |S|.
                # If we move to S' with |S'| > |S|, the best possible is |S'|.
                # So we should return the first valid clique found.
                return clique
    
    # If no clique found for small prime sets, fall back to the previous heuristic approach
    # which might find a set with fewer primes than the trivial range, even if not a perfect smooth clique.
    
    if time.time() < deadline - 10:
        # Generate 2,3,5-smooth numbers as a pool for local search
        smooth_pool = generate_smooth_numbers([2, 3, 5], MAXV)
        if len(smooth_pool) >= N:
            # Start with the first N smooth numbers
            current = smooth_pool[:N]
            current_score = count_distinct_primes_in_sums(current)
            
            if current_score < best_score:
                best_score = current_score
                best_result = current[:]
            
            # Simple local search: swap elements to reduce distinct primes
            # Limit iterations to ensure we finish
            for iteration in range(200):
                if time.time() > deadline - 3:
                    break
                
                improved = False
                # Try swapping each element with a random element from the pool
                for i in range(N):
                    if time.time() > deadline - 3:
                        break
                    # Pick a few random candidates from the pool
                    for _ in range(5):
                        j = int(time.time() * 1000) % len(smooth_pool) # Pseudo-random
                        new_val = smooth_pool[j]
                        if new_val in current and new_val != current[i]:
                            continue
                        if new_val == current[i]:
                            continue
                            
                        trial = current[:]
                        trial[i] = new_val
                        trial.sort()
                        
                        trial_score = count_distinct_primes_in_sums(trial)
                        if trial_score < current_score:
                            current = trial
                            current_score = trial_score
                            improved = True
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
