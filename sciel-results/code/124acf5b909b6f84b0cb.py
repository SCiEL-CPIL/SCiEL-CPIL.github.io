import time
import random
import math

MAX_VAL = 20000
spf = [0] * (MAX_VAL + 1)
for i in range(2, MAX_VAL + 1):
    if spf[i] == 0:
        for j in range(i, MAX_VAL + 1, i):
            if spf[j] == 0:
                spf[j] = i

def get_prime_factors(n):
    factors = set()
    if n < 2:
        return factors
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes.update(get_prime_factors(s))
    return len(primes)

def is_P_smooth(n, P):
    """Check if n is P-smooth (all prime factors in P)"""
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    while n % 7 == 0:
        n //= 7
    while n % 11 == 0:
        n //= 11
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Precompute prime sets to try
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    # Generate P-smooth numbers for each set
    def generate_smooth_numbers(P, limit=20000):
        """Generate all P-smooth numbers up to limit"""
        smooth = set()
        
        def gen(idx, current):
            if current > limit:
                return
            if idx == len(P):
                smooth.add(current)
                return
            p = P[idx]
            val = current
            while val <= limit:
                smooth.add(val)
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        # Remove 1 since we need positive integers and sums >= 2
        smooth.discard(1)
        return smooth
    
    best_found = None
    best_cost = float('inf')
    
    for P in prime_sets:
        if time.time() > deadline - 10:
            break
            
        # Generate P-smooth numbers
        smooth_nums = generate_smooth_numbers(P)
        
        # Build adjacency list as sets for fast intersection
        # adj[x] is set of y such that x+y is P-smooth
        adj = [set() for _ in range(10001)]
        
        for S in smooth_nums:
            # x + y = S => y = S - x
            low = max(1, S - 10000)
            high = min(10000, S - 1)
            for x in range(low, high + 1):
                y = S - x
                if 1 <= y <= 10000 and x != y:
                    adj[x].add(y)
        
        # Find clique of size 12 using optimized backtracking
        # Sort nodes by degree descending
        nodes = list(range(1, 10001))
        nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # We only consider top K nodes to keep search space manageable
        # If a clique of 12 exists, it's likely in the dense core
        K = 500 
        top_nodes = nodes[:K]
        
        found_clique = None
        
        # Try to find a clique among top_nodes
        # We use a recursive search with set intersections
        def find_clique_optimized(current_clique, candidates_set, target_size):
            if len(current_clique) == target_size:
                return current_clique[:]
            
            if len(candidates_set) < target_size - len(current_clique):
                return None
            
            # Optimization: if candidates are few, try them
            # We pick a node from candidates, intersect neighbors with remaining candidates
            # To be deterministic and fast, we can iterate through candidates
            # But to avoid symmetry, we can just pick the first one? No, we need to explore.
            # A better heuristic: pick the node in candidates with highest degree in the induced subgraph?
            # For simplicity and speed, we iterate.
            
            # Convert to list for iteration, but maintain set for intersection
            candidates_list = list(candidates_set)
            
            for i, v in enumerate(candidates_list):
                # New candidates must be in candidates_set (excluding v and earlier)
                # AND be neighbors of v
                # AND be neighbors of all in current_clique (already satisfied if we maintain intersection)
                
                new_candidates = candidates_set.intersection(adj[v])
                # Remove v and any node processed before v in this loop? 
                # Actually, standard backtracking removes the chosen node and restricts to nodes "after" it in some ordering to avoid duplicates.
                # Here, using sets, we can just remove v. But we might find the same clique in different orders.
                # To prune, we can maintain an ordering. 
                # However, for a small K (500), and target 12, simple set intersection is fast enough if we prune well.
                
                # Pruning: if new_candidates size < needed, skip
                if len(new_candidates) < target_size - len(current_clique) - 1:
                    continue
                
                res = find_clique_optimized(current_clique + [v], new_candidates, target_size)
                if res:
                    return res
            
            return None
        
        # Initial candidates: all top_nodes
        initial_candidates_set = set(top_nodes)
        
        # We need to start with an empty clique? 
        # The function above assumes current_clique is valid and candidates are neighbors of all in current_clique.
        # For the first step, candidates are just the pool.
        # So we can call find_clique_optimized([], initial_candidates_set, 12)
        
        # To speed up, we can try starting from specific high-degree nodes
        # But the recursive function handles it.
        # Let's add a timeout check inside the recursion? 
        # Python recursion limit might be an issue? Depth is 12, so no.
        # But the branching factor can be large.
        
        # Let's try a few restarts with different subsets of top_nodes if the first fails
        # Or just run it once. The graph is symmetric-ish.
        
        # One issue: The recursive function might be slow if the graph is dense.
        # Let's add a node limit check.
        
        # We will try to find the clique.
        # If it takes too long, we break.
        
        # We'll wrap the search in a way that we can interrupt it if time is running out.
        # Since we can't easily interrupt recursion, we rely on the K=500 limit and the pruning.
        
        res = find_clique_optimized([], initial_candidates_set, 12)
        if res:
            found_clique = res
        
        if found_clique:
            cost = count_distinct_primes(found_clique)
            if cost <= len(P):
                best_found = found_clique
                best_cost = cost
                if cost <= 2:
                    break
    
    # If we found a good solution, return it
    if best_found is not None and best_cost <= 4:
        return best_found
    
    # Fallback: Simulated Annealing with better initialization
    # Start with a set that has low prime count
    best_arr = [3 + 6 * k for k in range(12)]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.9995
    
    while time.time() < deadline - 2:
        if T < 0.01:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        new_cost = count_distinct_primes(current_arr)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_arr = current_arr[:]
        else:
            delta = new_cost - current_cost
            if random.random() < math.exp(-delta / T):
                current_cost = new_cost
            else:
                current_arr[idx] = old_val
        
        T *= cooling_rate
    
    return best_arr
