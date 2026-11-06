"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of prime factors
    of all pairwise sums is minimized.
    
    Key insight: If we find 12 distinct positive integers k_1..k_12 such that all
    pairwise sums k_i + k_j are 3-smooth (prime factors in {2,3}), then setting
    x_i = 6 * k_i gives us numbers where all sums have only prime factors {2,3}.
    This yields a score of 2.
    
    Strategy:
    1. Generate all 3-smooth numbers up to a reasonable limit.
    2. Build a graph where vertices are small positive integers and edges connect
       pairs whose sum is 3-smooth.
    3. Find a clique of size 12 in this graph.
    4. Scale by 6 to get the final answer.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s buffer before 90s kill
    
    def is_3_smooth(n):
        """Check if n is 3-smooth (all prime factors in {2,3})."""
        if n <= 0:
            return False
        while n % 2 == 0:
            n //= 2
        while n % 3 == 0:
            n //= 3
        return n == 1
    
    # Generate 3-smooth numbers up to 200 (since max sum of two numbers <= 100 is 200)
    smooth_3 = set()
    for i in range(1, 201):
        if is_3_smooth(i):
            smooth_3.add(i)
    
    # We need to find 12 distinct positive integers k_1..k_12 such that
    # all pairwise sums k_i + k_j (i<j) are in smooth_3.
    # Search for such a clique using backtracking with pruning.
    
    def find_clique_3_smooth(size, max_val=50):
        """Find a set of 'size' distinct integers in [1, max_val] with all pairwise sums 3-smooth."""
        
        # Build adjacency: adj[v] = set of u != v such that u+v is 3-smooth
        adj = {}
        for v in range(1, max_val + 1):
            adj[v] = set()
            for u in range(1, max_val + 1):
                if u != v and (u + v) in smooth_3:
                    adj[v].add(u)
        
        # Backtracking clique search with degree-based ordering
        vertices = list(range(1, max_val + 1))
        # Sort by degree (highest first) for better pruning
        vertices.sort(key=lambda v: len(adj[v]), reverse=True)
        
        def backtrack(clique, candidates, start_idx):
            if len(clique) == size:
                return clique
            if start_idx >= len(candidates):
                return None
            
            for i in range(start_idx, len(candidates)):
                v = candidates[i]
                if time.time() > deadline:
                    return None
                
                new_clique = clique + [v]
                # All remaining candidates must be adjacent to v
                new_candidates = [u for j in range(i+1, len(candidates)) 
                                  for u in [candidates[j]] if u in adj[v]]
                
                result = backtrack(new_clique, new_candidates, 0)
                if result is not None:
                    return result
            
            return None
        
        return backtrack([], vertices, 0)
    
    # Try to find a clique of size 12 in the 3-smooth sum graph
    result = None
    for max_val in [30, 40, 50, 60, 80]:
        if time.time() > deadline:
            break
        result = find_clique_3_smooth(12, max_val)
        if result is not None:
            break
    
    if result is not None:
        # Scale by 6: x_i = 6 * k_i
        answer = [x * 6 for x in result]
        # Verify: all distinct, positive, <= 10000
        if len(set(answer)) == 12 and all(1 <= x <= 10000 for x in answer):
            # Double-check that all pairwise sums are 3-smooth
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    s = answer[i] + answer[j]
                    if not is_3_smooth(s):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return answer
    
    # Fallback: Try 5-smooth approach (prime factors in {2,3,5})
    def is_5_smooth(n):
        """Check if n is 5-smooth (all prime factors in {2,3,5})."""
        if n <= 0:
            return False
        for p in [2, 3, 5]:
            while n % p == 0:
                n //= p
        return n == 1
    
    smooth_5 = set()
    for i in range(1, 201):
        if is_5_smooth(i):
            smooth_5.add(i)
    
    def find_clique_5_smooth(size, max_val=50):
        """Find a set of 'size' distinct integers in [1, max_val] with all pairwise sums 5-smooth."""
        adj = {}
        for v in range(1, max_val + 1):
            adj[v] = set()
            for u in range(1, max_val + 1):
                if u != v and (u + v) in smooth_5:
                    adj[v].add(u)
        
        vertices = list(range(1, max_val + 1))
        vertices.sort(key=lambda v: len(adj[v]), reverse=True)
        
        def backtrack(clique, candidates, start_idx):
            if len(clique) == size:
                return clique
            if start_idx >= len(candidates):
                return None
            
            for i in range(start_idx, len(candidates)):
                v = candidates[i]
                if time.time() > deadline:
                    return None
                
                new_clique = clique + [v]
                new_candidates = [u for j in range(i+1, len(candidates)) 
                                  for u in [candidates[j]] if u in adj[v]]
                
                result = backtrack(new_clique, new_candidates, 0)
                if result is not None:
                    return result
            
            return None
        
        return backtrack([], vertices, 0)
    
    result5 = None
    for max_val in [30, 40, 50, 60, 80]:
        if time.time() > deadline:
            break
        result5 = find_clique_5_smooth(12, max_val)
        if result5 is not None:
            break
    
    if result5 is not None:
        # Scale by 30: x_i = 30 * k_i
        answer = [x * 30 for x in result5]
        if len(set(answer)) == 12 and all(1 <= x <= 10000 for x in answer):
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    s = answer[i] + answer[j]
                    if not is_5_smooth(s):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return answer
    
    # Final fallback: Local search to minimize distinct prime factors
    def count_distinct_primes_of_sums(nums):
        """Count distinct prime factors across all pairwise sums of nums."""
        primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                d = 2
                temp = s
                while d * d <= temp:
                    while temp % d == 0:
                        primes.add(d)
                        temp //= d
                    d += 1
                if temp > 1:
                    primes.add(temp)
        return len(primes)
    
    # Start with a reasonable initial set
    random.seed(42)
    best_set = list(range(1, 13))
    best_score = count_distinct_primes_of_sums(best_set)
    
    no_improve_count = 0
    while time.time() < deadline - 5 and no_improve_count < 500:
        current_set = best_set[:]
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        improvement_found = False
        
        for _ in range(5):
            if time.time() > deadline:
                break
            new_val = random.randint(1, 10000)
            if new_val in current_set:
                continue
            current_set[idx] = new_val
            score = count_distinct_primes_of_sums(current_set)
            if score < best_score:
                best_set = current_set[:]
                best_score = score
                improvement_found = True
                no_improve_count = 0
                break
            else:
                current_set[idx] = old_val
        
        if not improvement_found:
            no_improve_count += 1
    
    return best_set
# EVOLVE-BLOCK-END
