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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Prime sets to try, ordered by size (smaller is better)
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    best_found = None
    best_cost = float('inf')
    
    # Generate P-smooth numbers
    def generate_smooth_numbers(P, limit=20000):
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
        smooth.discard(1)
        return smooth

    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        smooth_nums = generate_smooth_numbers(P)
        
        # Build adjacency list
        # adj[x] is a set of y such that x+y is in smooth_nums
        # We only care about x, y in [1, 10000]
        adj = [set() for _ in range(10001)]
        
        for s in smooth_nums:
            # x + y = s => y = s - x
            # x must be in [1, 10000] and y must be in [1, 10000] and x != y
            low = max(1, s - 10000)
            high = min(10000, s - 1)
            for x in range(low, high + 1):
                y = s - x
                if 1 <= y <= 10000 and x != y:
                    adj[x].add(y)
        
        # Find clique of size 12
        # Sort nodes by degree descending to prioritize dense areas
        nodes = list(range(1, 10001))
        nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # We limit the search to top K nodes to avoid explosion
        # If a clique exists, it's likely in the high-degree core
        K = 1000
        top_nodes = nodes[:K]
        top_set = set(top_nodes)
        
        found_clique = None
        
        # Recursive clique search with pruning
        def find_clique(candidates, current_clique, target_size):
            if len(current_clique) == target_size:
                return current_clique[:]
            
            if len(candidates) < target_size - len(current_clique):
                return None
            
            # Heuristic: Pick the node in candidates with the highest degree within candidates
            # This is expensive if done naively. 
            # Approximation: Use global degree as a proxy, or precompute induced degrees?
            # For now, iterate and pick the first one that has enough neighbors?
            # Or just pick the one with max global degree from the list.
            
            # Convert to list for iteration
            cand_list = list(candidates)
            
            # Sort by global degree descending to try "central" nodes first
            cand_list.sort(key=lambda x: len(adj[x]), reverse=True)
            
            for v in cand_list:
                # New candidates must be in candidates (excluding v) and neighbors of v
                new_candidates = candidates.intersection(adj[v])
                new_candidates.discard(v)
                
                if len(new_candidates) < target_size - len(current_clique) - 1:
                    continue
                
                res = find_clique(new_candidates, current_clique + [v], target_size)
                if res:
                    return res
            
            return None
        
        # Start search
        # We can try starting from a few different high-degree nodes to break symmetry
        # But the recursive search above handles ordering.
        # To speed up, we can restrict initial candidates to top_nodes.
        
        # Optimization: If the graph is very dense, this might be fast.
        # If sparse, it might fail quickly.
        
        # We run the search with a time check inside? 
        # Python recursion doesn't easily allow time checks without global state.
        # We rely on K=1000 and the pruning.
        
        # Let's add a simple timeout wrapper by checking time at top level and breaking if too long
        # But the recursion is deep. 
        # Alternative: Use a non-recursive stack-based search?
        # For now, let's assume the recursion finishes or fails quickly for K=1000.
        
        # If we want to be safer, we can reduce K or add a node visit counter.
        # Let's stick with K=1000 for now.
        
        res = find_clique(top_set, [], 12)
        if res:
            found_clique = res
            cost = count_distinct_primes(found_clique)
            if cost <= len(P):
                best_found = found_clique
                best_cost = cost
                if cost <= 2:
                    break
    
    if best_found is not None:
        return best_found
    
    # Fallback: Simulated Annealing
    # Initialize with a set that has relatively few prime factors
    # e.g., multiples of 30 (2,3,5)
    init_arr = [30 * (k + 1) for k in range(12)]
    # Ensure distinct and <= 10000
    # 30*12 = 360, well within limits.
    
    best_arr = init_arr[:]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() < deadline - 1:
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
