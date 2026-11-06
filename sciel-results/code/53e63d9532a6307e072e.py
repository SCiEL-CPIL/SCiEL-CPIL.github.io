import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    n = 12
    max_val = 10000
    
    # List of prime sets to try, ordered by size
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
    ]
    
    best_solution = None
    best_score = float('inf')
    
    def get_smooth_numbers(max_sum, primes):
        """Generate all numbers <= max_sum with prime factors in primes."""
        smooth = set()
        primes_set = set(primes)
        
        def gen(current, max_p_idx):
            if current > max_sum:
                return
            if current > 1:
                smooth.add(current)
            for i in range(max_p_idx, len(primes)):
                p = primes[i]
                if current * p > max_sum:
                    break
                gen(current * p, i)
        
        gen(1, 0)
        return smooth

    def find_clique(n, max_val, primes):
        """Find a clique of size n in the graph where edge (u,v) exists if u+v is smooth wrt primes."""
        if time.time() > deadline:
            return None
            
        max_sum = 2 * max_val
        smooth_sums = get_smooth_numbers(max_sum, primes)
        
        # Build adjacency list
        # neighbors[u] is a set of v such that u+v is in smooth_sums
        neighbors = defaultdict(set)
        
        # Optimization: Only iterate u, and for each u, check all k in smooth_sums
        # v = k - u
        # This is |smooth_sums| * max_val iterations
        # If |smooth_sums| is small, this is fine.
        
        for u in range(1, max_val + 1):
            for k in smooth_sums:
                v = k - u
                if 1 <= v <= max_val and v != u:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
        
        # Sort candidates by degree (number of compatible partners)
        # Higher degree means more likely to be part of a clique
        candidates = sorted(range(1, max_val + 1), key=lambda x: len(neighbors[x]), reverse=True)
        
        # Limit candidates to top 500 to keep search fast
        # If a clique exists, it's likely in the high-degree region
        top_candidates = candidates[:500]
        
        # Map original value to index in top_candidates for fast lookup?
        # Or just use a set for fast membership.
        top_set = set(top_candidates)
        
        # DFS to find clique
        # We will pick nodes from top_candidates
        # To ensure distinctness and order, we can just iterate
        # But standard clique search uses recursion
        
        found_clique = []
        
        def dfs(current_clique, available):
            if time.time() > deadline:
                return False
            if len(current_clique) == n:
                found_clique.extend(current_clique)
                return True
                
            # Heuristic: Pick the next node from available
            # To speed up, we can try nodes in order of degree within available
            # But 'available' shrinks as we go.
            
            # Simple approach: iterate through available
            # If available is empty, fail
            if not available:
                return False
            
            # Try each candidate in available
            # We can optimize by checking if the candidate is connected to all in current_clique
            # But 'available' is already filtered to be connected to all in current_clique
            
            for i, node in enumerate(available):
                # Add node
                new_clique = current_clique + [node]
                
                # Filter available: keep only those connected to node
                # And remove node itself
                new_available = [v for v in available[i+1:] if v in neighbors[node]]
                
                if dfs(new_clique, new_available):
                    return True
                    
            return False
        
        # Initial available: all top_candidates
        # We need to be careful with the 'available' list logic in DFS
        # The standard Bron-Kerbosch or similar is better, but simple DFS with pruning works for small n
        
        # Let's refine the DFS to be more efficient
        # We pass a list of candidates that are compatible with all nodes in current_clique
        
        def dfs_opt(current_clique, candidates_list):
            if time.time() > deadline:
                return False
            if len(current_clique) == n:
                found_clique.extend(current_clique)
                return True
            
            if not candidates_list:
                return False
            
            for i, node in enumerate(candidates_list):
                new_clique = current_clique + [node]
                
                # Next candidates must be in candidates_list[i+1:] AND connected to node
                next_candidates = [v for v in candidates_list[i+1:] if v in neighbors[node]]
                
                if dfs_opt(new_clique, next_candidates):
                    return True
            return False
        
        if dfs_opt([], top_candidates):
            return found_clique
            
        return None

    # Try each prime set
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        solution = find_clique(n, max_val, primes)
        if solution:
            # Verify solution
            prime_factors = set()
            for i in range(n):
                for j in range(i+1, n):
                    s = solution[i] + solution[j]
                    # Factor s
                    temp = s
                    for p in primes:
                        while temp % p == 0:
                            prime_factors.add(p)
                            temp //= p
                    # If temp > 1, it has a prime factor not in primes
                    if temp > 1:
                        # This shouldn't happen if find_clique worked correctly
                        # But just in case, add it
                        # Actually, if it has other factors, the score is higher
                        # We need to calculate the ACTUAL score
                        pass
            
            # Calculate actual score
            actual_primes = set()
            for i in range(n):
                for j in range(i+1, n):
                    s = solution[i] + solution[j]
                    temp = s
                    # Simple factorization
                    d = 2
                    while d * d <= temp:
                        while temp % d == 0:
                            actual_primes.add(d)
                            temp //= d
                        d += 1
                    if temp > 1:
                        actual_primes.add(temp)
            
            score = len(actual_primes)
            if score < best_score:
                best_score = score
                best_solution = solution
                
            # If score is very low, we can stop
            if score <= 5:
                break

    # Fallback: Local Search
    if not best_solution or time.time() < deadline - 5.0:
        # Simple local search
        # Start with random 12
        current_set = random.sample(range(1, max_val + 1), n)
        
        def get_prime_factors(x):
            factors = set()
            d = 2
            while d * d <= x:
                while x % d == 0:
                    factors.add(d)
                    x //= d
                d += 1
            if x > 1:
                factors.add(x)
            return factors
        
        def calc_score(s):
            pf = set()
            for i in range(n):
                for j in range(i+1, n):
                    pf.update(get_prime_factors(s[i] + s[j]))
            return len(pf)
        
        current_score = calc_score(current_set)
        if current_score < best_score:
            best_score = current_score
            best_solution = current_set
            
        # Pool of candidates
        # Use numbers that are "smooth" themselves? Or just random.
        # Let's use a pool of 1000 random numbers + the current set
        pool = set(current_set)
        while len(pool) < 1000:
            pool.add(random.randint(1, max_val))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Try swapping each element
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                for new_val in pool:
                    if new_val in current_set:
                        continue
                    # Swap
                    old_val = current_set[i]
                    current_set[i] = new_val
                    
                    new_score = calc_score(current_set)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_solution = list(current_set)
                        break # Restart loop? Or continue?
                    else:
                        current_set[i] = old_val
                        
    if best_solution:
        return best_solution
    else:
        return list(range(1, n + 1))
# EVOLVE-BLOCK-END
