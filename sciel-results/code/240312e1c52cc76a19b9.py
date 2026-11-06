import time
import math
import random
from functools import reduce

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of prime factors
    of all pairwise sums is minimized.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s buffer
    
    def is_smooth(n, primes):
        """Check if n is smooth with respect to the given primes."""
        if n <= 0:
            return False
        for p in primes:
            while n % p == 0:
                n //= p
        return n == 1

    def find_clique(primes, target_size=12, max_val=100):
        """
        Find a set of 'target_size' distinct integers in [1, max_val]
        such that all pairwise sums are 'primes'-smooth.
        """
        # Generate smooth numbers up to 2*max_val
        limit = 2 * max_val
        smooth_set = set()
        for i in range(1, limit + 1):
            if is_smooth(i, primes):
                smooth_set.add(i)
        
        # Build adjacency list
        # adj[v] contains u such that u != v and u+v is smooth
        adj = {v: set() for v in range(1, max_val + 1)}
        for v in range(1, max_val + 1):
            for u in range(1, max_val + 1):
                if u != v and (u + v) in smooth_set:
                    adj[v].add(u)
        
        # Sort vertices by degree descending to help backtracking
        vertices = list(range(1, max_val + 1))
        vertices.sort(key=lambda v: len(adj[v]), reverse=True)
        
        def backtrack(clique, candidates, start_idx):
            if len(clique) == target_size:
                return clique
            if start_idx >= len(candidates):
                return None
            
            # Pruning: if not enough candidates left, return None
            if len(candidates) - start_idx < target_size - len(clique):
                return None
                
            for i in range(start_idx, len(candidates)):
                v = candidates[i]
                if time.time() > deadline:
                    return None
                
                # Check if v is connected to all existing clique members?
                # Actually, the candidates list is already filtered to be connected to all previous clique members
                # So we just need to filter the remaining candidates
                
                new_clique = clique + [v]
                # New candidates must be adjacent to v and have index > i
                new_candidates = [u for j in range(i + 1, len(candidates)) 
                                  if candidates[j] in adj[v]]
                
                result = backtrack(new_clique, new_candidates, 0)
                if result is not None:
                    return result
            
            return None
        
        return backtrack([], vertices, 0)

    # Try different prime sets
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    best_answer = None
    best_score = float('inf')
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
            
        # Try to find a clique
        result = find_clique(primes, 12, 100)
        
        if result is not None:
            # Scale by product of primes
            scale = reduce(lambda x, y: x * y, primes, 1)
            answer = [x * scale for x in result]
            
            # Verify
            if len(set(answer)) == 12 and all(1 <= x <= 10000 for x in answer):
                # Check distinct primes
                all_primes = set()
                for i in range(12):
                    for j in range(i + 1, 12):
                        s = answer[i] + answer[j]
                        # Factor s
                        temp = s
                        d = 2
                        while d * d <= temp:
                            while temp % d == 0:
                                all_primes.add(d)
                                temp //= d
                            d += 1
                        if temp > 1:
                            all_primes.add(temp)
                
                score = len(all_primes)
                if score < best_score:
                    best_score = score
                    best_answer = answer
                
                # If we found a solution with score equal to len(primes), it's optimal for this set
                # and likely globally optimal or close.
                if score == len(primes):
                    return answer

    # Fallback: Local Search
    if best_answer is None or time.time() < deadline - 5:
        # Precompute smallest prime factor for numbers up to 20000
        MAX_SUM = 20000
        spf = list(range(MAX_SUM + 1))
        for i in range(2, int(MAX_SUM**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, MAX_SUM + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def get_prime_factors(n):
            factors = set()
            while n > 1:
                p = spf[n]
                factors.add(p)
                while n % p == 0:
                    n //= p
            return factors
        
        def count_distinct_primes(nums):
            primes = set()
            for i in range(12):
                for j in range(i + 1, 12):
                    primes.update(get_prime_factors(nums[i] + nums[j]))
            return len(primes)
        
        # Initialize with a random set or the best found so far
        if best_answer:
            current_set = best_answer[:]
            best_score = count_distinct_primes(current_set)
        else:
            current_set = list(range(1, 13))
            best_score = count_distinct_primes(current_set)
        
        random.seed(42)
        no_improve = 0
        
        while time.time() < deadline - 2 and no_improve < 1000:
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Try to find a better value for this index
            # Sample from small numbers and large numbers
            candidates = [random.randint(1, 100) for _ in range(5)] + \
                         [random.randint(100, 10000) for _ in range(5)]
            
            improved = False
            for new_val in candidates:
                if new_val in current_set and new_val != old_val:
                    continue
                
                current_set[idx] = new_val
                score = count_distinct_primes(current_set)
                
                if score < best_score:
                    best_score = score
                    improved = True
                    break
                else:
                    current_set[idx] = old_val
            
            if improved:
                no_improve = 0
                if best_answer is None or best_score < len(best_answer) * 0: # just update
                    best_answer = current_set[:]
            else:
                no_improve += 1
                
                # If stuck for a while, try a random restart or larger jumps
                if no_improve % 100 == 0:
                    # Try a completely random set
                    current_set = [random.randint(1, 10000) for _ in range(12)]
                    # Ensure distinct
                    while len(set(current_set)) < 12:
                        current_set[random.randint(0, 11)] = random.randint(1, 10000)
                    score = count_distinct_primes(current_set)
                    if score < best_score:
                        best_score = score
                        best_answer = current_set[:]
                        no_improve = 0
                    else:
                        current_set = best_answer[:] if best_answer else list(range(1, 13))
        
        if best_answer is not None:
            return best_answer
    
    # Final fallback
    return list(range(1, 13))
