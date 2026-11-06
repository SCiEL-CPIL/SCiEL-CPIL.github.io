import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    MAX_VAL = 20000
    # Sieve of Eratosthenes for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def is_smooth(n, prime_set):
        """Check if all prime factors of n are in prime_set."""
        if n <= 1:
            return True
        while n > 1:
            p = spf[n]
            if p not in prime_set:
                return False
            while n % p == 0:
                n //= p
        return True

    def build_adj(N, prime_set):
        """Build adjacency list for graph where edge (u,v) exists if u+v is smooth w.r.t prime_set."""
        adj = [set() for _ in range(N + 1)]
        for u in range(1, N + 1):
            for v in range(u + 1, N + 1):
                if is_smooth(u + v, prime_set):
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def find_clique_size_12(adj, N):
        """
        Find a clique of size 12 in the graph defined by adj[1..N].
        Uses Bron-Kerbosch with pivoting and pruning.
        """
        # Pre-filter: nodes must have degree >= 11
        candidates = [i for i in range(1, N + 1) if len(adj[i]) >= 11]
        
        # Sort candidates by degree descending to improve pruning
        candidates.sort(key=lambda x: -len(adj[x]))
        
        if len(candidates) < 12:
            return None

        def expand(clique, cand):
            if len(clique) == 12:
                return clique
            if not cand:
                return None
            
            # Pruning
            if len(clique) + len(cand) < 12:
                return None

            # Choose a pivot to reduce branching
            # Standard BK: pick v in cand that maximizes |cand \ N(v)|? 
            # Actually, to find ANY clique of size K, we can just iterate.
            # But picking a pivot helps prune.
            # Let's pick the node with highest degree in cand as a heuristic for the "next" node?
            # No, standard BK picks pivot v, then iterates over cand \ N(v).
            
            # For speed, let's just iterate through cand, but order by degree in cand?
            # Or just simple iteration.
            
            # Optimization: If we can't find a clique of size 12 in the subgraph induced by cand, return None.
            # This is expensive to check exactly, so we rely on the recursive structure.
            
            for u in cand:
                new_clique = clique + [u]
                new_cand = [v for v in cand if v in adj[u]]
                
                res = expand(new_clique, new_cand)
                if res is not None:
                    return res
                    
            return None

        return expand([], candidates)

    def evaluate(arr):
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)

    # Strategy:
    # 1. Try to find a clique of size 12 in the "smooth sum" graph for small prime sets.
    #    If found, the number of distinct primes is exactly the size of that prime set.
    # 2. If not found within time, fall back to local search with a smart candidate pool.

    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11]
    ]
    
    # Determine N for each prime set
    # For P={2}, sums are powers of 2. 1..10000 has ~14 powers of 2.
    # For P={2,3}, sums are 2^a*3^b. Density is higher.
    # Let's try N=2000 for small sets, N=1000 for larger.
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
            
        N = 2000 if len(primes) <= 2 else 1000
        
        adj = build_adj(N, set(primes))
        
        if time.time() > deadline - 5:
            continue
            
        clique = find_clique_size_12(adj, N)
        
        if clique:
            if len(set(clique)) == 12 and max(clique) <= 10000:
                return clique

    # Fallback: Local search
    random.seed(42)
    
    # Start with a decent initial guess: multiples of 6
    best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    best_score = evaluate(best)
    
    # Generate candidate pool: focus on multiples of small numbers to encourage smooth sums
    candidate_pool = []
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        for k in range(1, 10000 // m + 1):
            val = k * m
            if 1 <= val <= 10000:
                candidate_pool.append(val)
    # Add some random values
    for _ in range(5000):
        candidate_pool.append(random.randint(1, 10000))
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if time.time() > deadline:
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

    return best
