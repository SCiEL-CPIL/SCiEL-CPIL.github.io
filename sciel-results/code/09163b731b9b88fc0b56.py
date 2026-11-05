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
    
    # Precompute prime sets to try
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
    ]
    
    # Precompute P-smooth numbers for each set?
    # Or just generate on the fly for the graph construction.
    
    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        # Generate P-smooth numbers up to 20000
        smooth_nums = set()
        # Use recursive generation or simple loops
        # For small P, simple loops are fine
        if len(P) == 2:
            p1, p2 = P
            for a in range(1, 16):
                v1 = p1 ** a
                if v1 > 20000: break
                for b in range(1, 16):
                    v2 = v1 * (p2 ** b)
                    if v2 > 20000: break
                    smooth_nums.add(v2)
        elif len(P) == 3:
            p1, p2, p3 = P
            for a in range(1, 16):
                v1 = p1 ** a
                if v1 > 20000: break
                for b in range(1, 16):
                    v2 = v1 * (p2 ** b)
                    if v2 > 20000: break
                    for c in range(1, 16):
                        v3 = v2 * (p3 ** c)
                        if v3 > 20000: break
                        smooth_nums.add(v3)
        elif len(P) == 4:
            p1, p2, p3, p4 = P
            for a in range(1, 16):
                v1 = p1 ** a
                if v1 > 20000: break
                for b in range(1, 16):
                    v2 = v1 * (p2 ** b)
                    if v2 > 20000: break
                    for c in range(1, 16):
                        v3 = v2 * (p3 ** c)
                        if v3 > 20000: break
                        for d in range(1, 16):
                            v4 = v3 * (p4 ** d)
                            if v4 > 20000: break
                            smooth_nums.add(v4)
        elif len(P) == 5:
            p1, p2, p3, p4, p5 = P
            for a in range(1, 16):
                v1 = p1 ** a
                if v1 > 20000: break
                for b in range(1, 16):
                    v2 = v1 * (p2 ** b)
                    if v2 > 20000: break
                    for c in range(1, 16):
                        v3 = v2 * (p3 ** c)
                        if v3 > 20000: break
                        for d in range(1, 16):
                            v4 = v3 * (p4 ** d)
                            if v4 > 20000: break
                            for e in range(1, 16):
                                v5 = v4 * (p5 ** e)
                                if v5 > 20000: break
                                smooth_nums.add(v5)
        
        if not smooth_nums:
            continue
            
        # Build adjacency list
        # Nodes are 1..10000
        # adj[x] is list of y such that x+y in smooth_nums
        adj = [[] for _ in range(10001)]
        for S in smooth_nums:
            # x + y = S => y = S - x
            # x in [1, 10000], y in [1, 10000]
            # So S - 10000 <= x <= S - 1
            # Also 1 <= x <= 10000
            low = max(1, S - 10000)
            high = min(10000, S - 1)
            for x in range(low, high + 1):
                y = S - x
                if 1 <= y <= 10000 and x != y:
                    adj[x].append(y)
        
        # Find clique of size 12
        # Simple backtracking
        # Sort nodes by degree to prune?
        # Or just iterate.
        
        # To speed up, we can try to find the clique starting from a random node or highest degree node.
        # Since we need ANY clique of size 12, we can try a greedy approach first.
        
        found_clique = None
        
        # Try a few random starting points or high degree nodes
        # High degree nodes are more likely to be in a large clique
        nodes = list(range(1, 10001))
        # Sort by degree descending
        nodes.sort(key=lambda x: len(adj[x]), reverse=True)
        
        # Try to build clique from top K nodes
        # If the top nodes don't form a clique, try others?
        # This is heuristic.
        
        # Better: Bron-Kerbosch with pivot?
        # Or simple DFS with pruning.
        
        def find_clique(candidates, current_clique, target_size):
            if len(current_clique) == target_size:
                return current_clique[:]
            if len(candidates) < target_size - len(current_clique):
                return None
            
            # Pruning: if the number of candidates that are connected to all current clique members is too small
            # Filter candidates that are connected to all in current_clique
            valid_candidates = []
            for v in candidates:
                is_connected = True
                for u in current_clique:
                    if v not in adj[u]: # O(N) check, slow
                        is_connected = False
                        break
                if is_connected:
                    valid_candidates.append(v)
            
            if len(valid_candidates) < target_size - len(current_clique):
                return None
            
            # Recurse
            # Pick a node
            # To improve speed, pick the node with highest degree in valid_candidates
            # Or just the first one
            v = valid_candidates[0]
            new_clique = current_clique + [v]
            # New candidates must be in valid_candidates and connected to v
            new_candidates = [u for u in valid_candidates if u > v and u in adj[v]]
            
            res = find_clique(new_candidates, new_clique, target_size)
            if res:
                return res
            
            # Backtrack: remove v from valid_candidates and try next
            # This is standard Bron-Kerbosch without pivot, which can be slow.
            # For size 12, it might be okay if the graph is dense enough.
            
            # Optimization: Instead of checking `v not in adj[u]` (list search), use a set or bitset.
            # Given time constraints, let's use a set for adj[u] for faster lookup.
            
            return None
        
        # Convert adj lists to sets for O(1) lookup
        adj_sets = [set(lst) for lst in adj]
        
        # Redefine find_clique with sets
        def find_clique_fast(candidates, current_clique, target_size):
            if len(current_clique) == target_size:
                return current_clique[:]
            if len(candidates) < target_size - len(current_clique):
                return None
            
            valid_candidates = []
            for v in candidates:
                is_connected = True
                for u in current_clique:
                    if v not in adj_sets[u]:
                        is_connected = False
                        break
                if is_connected:
                    valid_candidates.append(v)
            
            if len(valid_candidates) < target_size - len(current_clique):
                return None
            
            # Try to find clique in valid_candidates
            # Pick v with max degree in valid_candidates?
            # Or just iterate
            for i, v in enumerate(valid_candidates):
                new_clique = current_clique + [v]
                # New candidates are those in valid_candidates after i that are connected to v
                new_candidates = [u for u in valid_candidates[i+1:] if u in adj_sets[v]]
                
                res = find_clique_fast(new_candidates, new_clique, target_size)
                if res:
                    return res
            
            return None

        # Start search
        # We can restrict the initial candidates to top 500 nodes by degree
        # to keep the search space small. If a clique exists, it's likely to involve high-degree nodes.
        initial_candidates = nodes[:1000]
        
        # Randomize the order of initial candidates to avoid getting stuck in a bad local structure?
        # Or just try the top ones.
        
        # Let's try a few random subsets of high-degree nodes
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            # Shuffle a subset of top nodes
            subset = random.sample(nodes[:2000], 1000)
            res = find_clique_fast(subset, [], 12)
            if res:
                found_clique = res
                break
        
        if found_clique:
            # Verify
            cost = count_distinct_primes(found_clique)
            if cost <= len(P):
                return found_clique
    
    # Fallback: Simulated Annealing
    best_arr = [3 + 6 * k for k in range(12)]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
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
