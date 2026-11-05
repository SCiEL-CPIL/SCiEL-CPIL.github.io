import time
import random
from itertools import combinations

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

# Primes up to 20000 (max sum is 10000+10000=20000)
PRIMES = get_primes(20000)
PRIME_SET = set(PRIMES)

def factorize(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            factors.add(d)
        d += 1 if d == 2 else 2
    if n > 1:
        factors.add(n)
    return factors

# Precompute prime factors for all sums 2..20000
SUM_PRIME_FACTORS = [set() for _ in range(20001)]
for s in range(2, 20001):
    SUM_PRIME_FACTORS[s] = factorize(s)

def count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums."""
    all_primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            all_primes.update(SUM_PRIME_FACTORS[nums[i] + nums[j]])
    return len(all_primes)

def find_clique_in_graph(adj, min_size, deadline):
    """
    Find a clique of size min_size in the graph using Bron-Kerbosch with pivot.
    adj: dict mapping node -> set of neighbors
    Returns list of nodes forming the clique, or None.
    """
    nodes = list(adj.keys())
    if len(nodes) < min_size:
        return None
    
    # Order nodes by degree (descending) for better pruning
    nodes.sort(key=lambda x: -len(adj[x]))
    
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if len(R) >= min_size:
            best_clique = R[:]
            return
        if not P:
            return
        
        # Pivot selection: choose vertex in P∪X with max degree in P
        pivot_candidates = P | X
        if not pivot_candidates:
            return
        pivot = max(pivot_candidates, key=lambda v: len(P & adj[v]))
        
        # Only consider vertices in P that are NOT adjacent to pivot
        candidates = [v for v in P if v not in adj[pivot]]
        
        for v in candidates:
            if time.time() > deadline:
                return
            R.append(v)
            new_P = P & adj[v]
            new_X = X & adj[v]
            P.remove(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            X.add(v)
            if len(R) >= min_size:
                return
    
    bron_kerbosch([], set(nodes), set())
    return best_clique if len(best_clique) >= min_size else None

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # Generate candidate pool: focus on small numbers and smooth numbers
    def generate_smooth_numbers(max_val, primes):
        nums = {1}
        for p in primes:
            current = list(nums)
            for x in current:
                y = x
                while y * p <= max_val:
                    y *= p
                    nums.add(y)
        return sorted(nums)
    
    # Use 2,3,5,7-smooth numbers as they produce sums with few prime factors
    smooth_nums = generate_smooth_numbers(10000, [2, 3, 5, 7])
    # Also include small integers 1-300 as they have diverse but manageable sums
    small_ints = list(range(1, 301))
    
    candidate_pool = list(set(smooth_nums + small_ints))
    candidate_pool.sort()
    
    # Strategy: Enumerate prime sets, build compatibility graph, find clique
    # Start with small prime sets and expand
    
    best_result = None
    best_score = float('inf')
    
    # Small prime sets to try (ordered by size, then lexicographic)
    # We'll try sets of size 2, 3, 4, 5, etc.
    # Focus on small primes first
    
    def try_prime_set(prime_set, pool, deadline):
        """
        Given a set of allowed primes, find 12 numbers from pool
        where all pairwise sums have prime factors only in prime_set.
        Returns the 12 numbers or None.
        """
        if len(prime_set) < 2:
            return None
        
        # Build compatibility graph
        # Node: number from pool
        # Edge: a-b if prime factors of (a+b) ⊆ prime_set
        
        adj = {}
        for a in pool:
            adj[a] = set()
        
        for i in range(len(pool)):
            a = pool[i]
            for j in range(i + 1, len(pool)):
                b = pool[j]
                s = a + b
                if s > 20000:
                    continue
                pf = SUM_PRIME_FACTORS[s]
                if pf.issubset(prime_set):
                    adj[a].add(b)
                    adj[b].add(a)
        
        # Remove isolated nodes
        adj = {k: v for k, v in adj.items() if len(v) >= 11}
        
        if len(adj) < 12:
            return None
        
        # Find clique of size 12
        clique = find_clique_in_graph(adj, 12, deadline)
        return clique
    
    # Try prime sets of increasing size
    # Start with the most promising: small primes
    small_primes = PRIMES[:30]  # First 30 primes
    
    # Generate candidate prime sets
    prime_sets_to_try = []
    
    # Size 2: all pairs of first 15 primes
    for i in range(15):
        for j in range(i + 1, 15):
            prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j]]))
    
    # Size 3: all triples of first 10 primes
    for i in range(10):
        for j in range(i + 1, 10):
            for k in range(j + 1, 10):
                prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k]]))
    
    # Size 4: some quadruples of first 8 primes
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                for l in range(k + 1, 8):
                    prime_sets_to_try.append(frozenset([small_primes[i], small_primes[j], small_primes[k], small_primes[l]]))
    
    # Sort by size, then by max prime
    prime_sets_to_try.sort(key=lambda ps: (len(ps), max(ps)))
    
    # Try each prime set
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        
        # Use a subset of the pool for speed
        # Focus on smaller numbers first
        pool_subset = candidate_pool[:500]
        
        clique = try_prime_set(set(ps), pool_subset, deadline - 3)
        
        if clique is not None and len(clique) >= 12:
            # Verify the solution
            score = count_distinct_primes(clique[:12])
            if score < best_score:
                best_score = score
                best_result = clique[:12]
                # If we found a very good solution, we can stop early
                if best_score <= 3:
                    break
    
    # If clique search didn't find a good solution, fall back to local search
    if best_result is None or best_score > 5:
        # Generate a good initial solution using smooth numbers
        initial = smooth_nums[:12]
        current_score = count_distinct_primes(initial)
        
        if current_score < best_score:
            best_score = current_score
            best_result = initial[:]
        
        # Local search: try to improve by swapping elements
        def local_search(initial_set, pool, max_iters=3000):
            current = initial_set[:]
            current_score = count_distinct_primes(current)
            best = current[:]
            best_score_local = current_score
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break
                
                improved = False
                idx = random.randint(0, len(current) - 1)
                old_val = current[idx]
                
                # Try a few random replacements
                for new_val in random.sample(pool, min(30, len(pool))):
                    if new_val == old_val or new_val in current:
                        continue
                    
                    trial = current[:]
                    trial[idx] = new_val
                    new_score = count_distinct_primes(trial)
                    
                    if new_score < current_score:
                        current = trial
                        current_score = new_score
                        improved = True
                        
                        if new_score < best_score_local:
                            best_score_local = new_score
                            best = trial[:]
                        break
                
                if not improved:
                    # Random perturbation
                    idx = random.randint(0, len(current) - 1)
                    new_val = random.choice(pool)
                    if new_val not in current:
                        trial = current[:]
                        trial[idx] = new_val
                        current = trial
                        current_score = count_distinct_primes(current)
            
            return best, best_score_local
        
        refined, refined_score = local_search(best_result, candidate_pool[:500])
        if refined_score < best_score:
            best_score = refined_score
            best_result = refined
    
    if best_result is None:
        # Last resort
        best_result = smooth_nums[:12]
    
    return best_result
