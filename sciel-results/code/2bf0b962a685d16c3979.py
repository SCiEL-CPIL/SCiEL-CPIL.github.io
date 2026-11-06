"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
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
    
    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return primes
    
    # Strategy: Graph-based clique search
    # Build a graph where vertices are integers 1..10000
    # Edge between u and v if u+v is P-smooth for some small prime set P
    # Find a clique of size 12 in this graph
    
    # Precompute prime factor sets for all sums up to 20000
    # Use a more efficient approach: for each candidate prime set P,
    # check if sum is P-smooth
    
    # Small prime sets to try (ordered by likelihood of finding large cliques)
    prime_sets_to_try = [
        {2, 3},
        {2, 5},
        {2, 7},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {3, 5},
        {3, 7},
        {5, 7},
        {2, 3, 5, 7},
        {2, 11},
        {2, 13},
        {3, 11},
        {3, 13},
        {5, 11},
        {5, 13},
        {7, 11},
        {7, 13},
        {2, 3, 11},
        {2, 3, 13},
        {2, 5, 11},
        {2, 5, 13},
        {2, 7, 11},
        {2, 7, 13},
        {3, 5, 11},
        {3, 5, 13},
        {3, 7, 11},
        {3, 7, 13},
        {5, 7, 11},
        {5, 7, 13},
        {7, 11, 13},
        {2, 3, 5, 11},
        {2, 3, 5, 13},
        {2, 3, 7, 11},
        {2, 3, 7, 13},
        {2, 5, 7, 11},
        {2, 5, 7, 13},
        {3, 5, 7, 11},
        {3, 5, 7, 13},
        {5, 7, 11, 13},
    ]
    
    def is_smooth(n, prime_set):
        """Check if n is smooth with respect to prime_set."""
        if n <= 1:
            return True
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    def greedy_clique_for_prime_set(prime_set, max_val=10000, clique_size=12, time_limit=10.0):
        """Find a clique of given size in the smooth-sum graph using greedy heuristic."""
        # For efficiency, only consider a subset of candidate values
        # Focus on values that are likely to form smooth sums
        # Strategy: try multiples of small numbers
        
        # Build candidate list
        candidates = []
        for m in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
            for k in range(1, max_val // m + 1):
                val = k * m
                if 1 <= val <= max_val:
                    candidates.append(val)
        candidates = list(set(candidates))
        candidates.sort()
        
        # Limit candidates to keep computation manageable
        if len(candidates) > 5000:
            # Keep evenly spaced samples
            step = len(candidates) // 5000
            candidates = candidates[::step]
        
        n_cand = len(candidates)
        
        # Build adjacency: adj[i][j] = True if candidates[i] + candidates[j] is smooth
        # Use a bitset for efficiency
        adj = [0] * n_cand  # adj[i] is a bitset where bit j is set if i and j are connected
        
        for i in range(n_cand):
            ci = candidates[i]
            for j in range(i + 1, n_cand):
                cj = candidates[j]
                if is_smooth(ci + cj, prime_set):
                    # Set bit j in adj[i] and bit i in adj[j]
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Greedy maximal clique: repeatedly pick the vertex with highest degree
        # among remaining candidates, then filter to its neighbors
        best_clique = []
        
        for attempt in range(20):  # Multiple random restarts
            if time.time() > time_limit:
                break
            
            # Random subset of candidates to start
            random_start = random.randint(0, n_cand - 1)
            current_candidates = list(range(n_cand))
            random.shuffle(current_candidates)
            
            clique = []
            remaining = set(current_candidates)
            
            while len(clique) < clique_size and remaining:
                # Find vertex in remaining with highest degree within remaining
                best_v = -1
                best_deg = -1
                for v in remaining:
                    deg = 0
                    for u in remaining:
                        if u != v:
                            if (adj[v] >> u) & 1:
                                deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                
                if best_v == -1:
                    break
                
                clique.append(best_v)
                remaining.discard(best_v)
                
                # Filter remaining to only those connected to best_v
                new_remaining = set()
                for u in remaining:
                    if (adj[best_v] >> u) & 1:
                        new_remaining.add(u)
                remaining = new_remaining
            
            if len(clique) > len(best_clique):
                best_clique = clique[:clique_size]
                if len(best_clique) == clique_size:
                    break
        
        if len(best_clique) >= clique_size:
            return [candidates[i] for i in best_clique[:clique_size]]
        return None
    
    # Try each prime set and find a clique
    best_result = None
    best_score = float('inf')
    
    for prime_set in prime_sets_to_try:
        if time.time() > deadline - 5.0:
            break
        
        # Time limit per prime set
        remaining_sets = len(prime_sets_to_try) - prime_sets_to_try.index(prime_set)
        time_per_set = max(2.0, (deadline - time.time() - 5.0) / remaining_sets)
        time_limit = time.time() + time_per_set
        
        clique = greedy_clique_for_prime_set(prime_set, max_val=10000, clique_size=12, time_limit=time_limit)
        
        if clique is not None and len(clique) == 12:
            # Verify all elements are distinct and within bounds
            if len(set(clique)) == 12 and all(1 <= x <= 10000 for x in clique):
                score = evaluate(clique)
                if score < best_score:
                    best_score = score
                    best_result = clique
                    if best_score <= 3:
                        # Excellent result, return early
                        return best_result
    
    # If no good clique found, fall back to local search from a good initial point
    if best_result is None or best_score > 5:
        # Start with multiples of 6
        best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        best_score = evaluate(best)
        
        random.seed(42)
        
        # Generate candidate values
        candidate_pool = []
        for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
            for k in range(1, 10000 // m + 1):
                val = k * m
                if 1 <= val <= 10000:
                    candidate_pool.append(val)
        for _ in range(3000):
            candidate_pool.append(random.randint(1, 10000))
        candidate_pool = list(set(candidate_pool))
        random.shuffle(candidate_pool)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(12):
                old_val = best[i]
                for new_val in candidate_pool:
                    if time.time() > deadline - 2.0:
                        break
                    if new_val == old_val or new_val in best:
                        continue
                    candidate = best.copy()
                    candidate[i] = new_val
                    score = evaluate(candidate)
                    if score < best_score:
                        best = candidate
                        best_score = score
                        improved = True
                        break
        
        if best_score < best_score if best_result is None else best_score:
            best_result = best
    
    return best_result if best_result is not None else [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
# EVOLVE-BLOCK-END
