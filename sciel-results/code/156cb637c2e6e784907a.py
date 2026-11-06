"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill

    # Precompute Smallest Prime Factor (SPF) up to 20000
    max_val = 20000
    spf = list(range(max_val + 1))
    for i in range(2, int(max_val**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_val + 1, i):
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

    def is_7_smooth(n):
        """Check if n is 7-smooth (all prime factors in {2,3,5,7})."""
        if n <= 0:
            return False
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                n //= p
        return n == 1

    # Try to find a set of 12 integers where all pairwise sums are 7-smooth.
    # If found, distinct primes = {2,3,5,7} -> Count 4. This is likely optimal.
    
    target_primes = [2, 3, 5, 7]
    
    def find_clique_7_smooth(max_search_val=200):
        # Build adjacency list for vertices 1..max_search_val
        # Edge (u,v) if u+v is 7-smooth
        adj = {v: set() for v in range(1, max_search_val + 1)}
        for u in range(1, max_search_val + 1):
            for v in range(u + 1, max_search_val + 1):
                if is_7_smooth(u + v):
                    adj[u].add(v)
                    adj[v].add(u)
                    
        # Backtracking search for clique of size 12
        # Use a simple DFS with pruning.
        
        def backtrack(clique, candidates):
            if len(clique) == 12:
                return clique
            if not candidates:
                return None
            if time.time() - start_time > timeout - 10:
                return None
            if len(candidates) < 12 - len(clique):
                return None
                
            for v in candidates:
                new_clique = clique + [v]
                # New candidates: neighbors of v that are in candidates and greater than v
                # This ensures we don't revisit permutations
                new_candidates = [u for u in candidates if u > v and u in adj[v]]
                result = backtrack(new_clique, new_candidates)
                if result is not None:
                    return result
            return None
        
        initial_candidates = list(range(1, max_search_val + 1))
        return backtrack([], initial_candidates)

    # Attempt to find a 7-smooth clique
    clique_result = None
    for ms in [100, 150, 200]:
        if time.time() - start_time > timeout - 30:
            break
        clique_result = find_clique_7_smooth(ms)
        if clique_result:
            break

    if clique_result:
        # Scale by 210 to ensure all sums are 7-smooth * 210
        # 210 = 2 * 3 * 5 * 7
        # Sum = 210 * (k_i + k_j). Since k_i+k_j is 7-smooth, 210*(k_i+k_j) is 7-smooth.
        # Thus prime factors are subset of {2,3,5,7}.
        answer = [x * 210 for x in clique_result]
        if len(set(answer)) == 12 and all(1 <= x <= 10000 for x in answer):
            return answer

    # Fallback: Local search on scaled integers (multiples of 6)
    # a_i = 6 * b_i
    # Sums = 6 * (b_i + b_j)
    # Prime factors of 6*(b_i+b_j) = {2, 3} U prime_factors(b_i+b_j)
    # So minimizing distinct primes of (b_i+b_j) minimizes total.
    
    def eval_b(arr_b):
        primes = set()
        n = len(arr_b)
        for i in range(n):
            for j in range(i+1, n):
                s = arr_b[i] + arr_b[j]
                if s <= max_val:
                    primes.update(get_prime_factors(s))
                else:
                    # Fallback factorization for large sums (unlikely with b_i <= 1666)
                    temp = s
                    d = 2
                    while d * d <= temp:
                        while temp % d == 0:
                            primes.add(d)
                            temp //= d
                        d += 1
                    if temp > 1:
                        primes.add(temp)
        return len(primes), primes

    # Initial heuristic: Odd numbers scaled by 6
    # b_i in {1, 3, ..., 23}
    b = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    best_b = b[:]
    current_score, _ = eval_b(best_b)
    
    max_iters = 5000
    random.seed(42)
    
    for _ in range(max_iters):
        if time.time() - start_time > timeout - 5:
            break
        
        # Pick a random index to mutate
        k = random.randint(0, 11)
        old_val = best_b[k]
        
        # Try a few random new values
        improved = False
        for _ in range(5):
            # New value should be distinct from others
            # Range 1 to 1666 (since 6*1666 = 9996 <= 10000)
            new_val = random.randint(1, 1666)
            if new_val in best_b and new_val != old_val:
                continue
            if new_val == old_val:
                continue
                
            # Swap
            best_b[k] = new_val
            new_score, _ = eval_b(best_b)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                break
            elif new_score == current_score:
                # Tie-breaker: accept with small probability to escape plateaus
                if random.random() < 0.1:
                    improved = True
                    break
                else:
                    best_b[k] = old_val
            else:
                best_b[k] = old_val
        
        if improved:
            pass
        else:
            # If no improvement, continue
            pass

    # Final result
    a = [6 * x for x in best_b]
    # Ensure distinctness and bounds
    if len(set(a)) == 12 and all(1 <= x <= 10000 for x in a):
        return a
    
    # Absolute fallback
    return list(range(1, 13))

# EVOLVE-BLOCK-END
