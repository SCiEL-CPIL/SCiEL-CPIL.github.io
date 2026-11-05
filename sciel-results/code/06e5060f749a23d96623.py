import time
import random
import math
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(20000)

MAX_VAL = 20000
LIMIT = 10000
N_NODES = 12

# Precompute smallest prime factor for fast factorization
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

def generate_smooth_numbers(primes, max_val):
    """Generate all numbers <= max_val whose prime factors are all in 'primes'"""
    smooth = set()
    primes = list(primes)
    
    def dfs(idx, current):
        if current > max_val:
            return
        if current > 1:
            smooth.add(current)
        if idx == len(primes):
            return
        p = primes[idx]
        val = current
        while val <= max_val:
            dfs(idx + 1, val)
            if val > max_val // p:
                break
            val *= p

    dfs(0, 1)
    return smooth

def build_adjacency(smooth_nums):
    """
    Build adjacency sets for nodes 1..LIMIT.
    adj[x] is a set of y such that x+y is in smooth_nums.
    """
    adj = [set() for _ in range(LIMIT + 1)]
    # For each smooth number S, add edges (x, S-x)
    for S in smooth_nums:
        # x + y = S => y = S - x
        # Constraints: 1 <= x <= LIMIT, 1 <= y <= LIMIT, x != y
        low = max(1, S - LIMIT)
        high = min(LIMIT, S - 1)
        for x in range(low, high + 1):
            y = S - x
            if 1 <= y <= LIMIT and x != y:
                adj[x].add(y)
    return adj

def find_clique(adj, target_size, deadline):
    """
    Find a clique of size target_size in the graph defined by adj.
    Uses a greedy backtracking approach with set intersections.
    """
    # Sort nodes by degree descending to prioritize high-connectivity nodes
    nodes = list(range(1, LIMIT + 1))
    nodes.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # We only need to consider a subset of high-degree nodes to keep search space manageable
    # If a clique exists, it's highly likely to be among the most connected nodes.
    # Let's take top 2000 nodes.
    candidates = set(nodes[:2000])
    
    # If the graph is very dense, we might need more, but 2000 is a good heuristic balance.
    # We can also try a few random subsets if the first one fails, but let's stick to one robust attempt per prime set.
    
    def backtrack(current_clique, candidates_set):
        if len(current_clique) == target_size:
            return current_clique[:]
        
        if time.time() > deadline:
            return None
            
        if len(candidates_set) < target_size - len(current_clique):
            return None
            
        # Pick a node from candidates
        # Heuristic: pick the node with highest degree within the current candidates
        # To avoid O(N^2) sorting at every step, we can just pick one, 
        # but sorting helps prune faster.
        # Since candidates_set changes, we can't pre-sort easily.
        # Let's just iterate. To improve, we can try nodes with higher global degree first.
        # We stored global degree in 'nodes' order. 
        # Let's create a list of candidates sorted by global degree for this step.
        # This is expensive if done at every recursion level.
        # Alternative: Just pick the first one? No, bad for pruning.
        # Let's use a simple heuristic: pick the node with max len(adj[v]) among candidates.
        
        # Optimization: If candidates_set is small, sort it. If large, just pick one.
        # Given the structure, candidates shrink fast.
        
        # To speed up, we can just iterate through the sorted 'nodes' list and pick the first one in candidates_set
        # But 'nodes' is sorted by global degree.
        
        found_node = None
        for v in nodes:
            if v in candidates_set:
                found_node = v
                break
                
        if found_node is None:
            return None
            
        # Branch on this node
        new_clique = current_clique + [found_node]
        
        # New candidates must be neighbors of found_node AND in current candidates_set AND not in new_clique
        new_candidates = candidates_set.intersection(adj[found_node])
        new_candidates.difference_update(new_clique)
        
        res = backtrack(new_clique, new_candidates)
        if res:
            return res
            
        # Backtrack: remove found_node from candidates and try next
        # To do this efficiently, we need to be able to "undo" the pick.
        # The standard Bron-Kerbosch does this by passing a set of candidates and removing one.
        # Here, we picked a specific node. We need to try other nodes.
        
        # Let's restructure to iterate over candidates properly.
        return None

    # The simple "pick first" above only finds one path. We need to explore branches.
    # Let's use a standard DFS that iterates over candidates.
    
    def dfs(clique, cand_set):
        if time.time() > deadline:
            return None
        if len(clique) == target_size:
            return clique[:]
        if len(cand_set) < target_size - len(clique):
            return None
            
        # Iterate through candidates
        # To maintain order (high degree first), we iterate through 'nodes' and check membership
        # This is O(N) per step, which is acceptable if depth is small (12).
        
        # However, iterating through 10000 nodes at every node of the tree is 12 * 10000 = 120k checks, which is fine.
        # But we must ensure we don't pick nodes already in clique.
        
        for v in nodes:
            if v not in cand_set:
                continue
                
            # Pick v
            new_clique = clique + [v]
            
            # Intersect candidates with neighbors of v
            new_cand = cand_set.intersection(adj[v])
            # Remove v and any nodes already in new_clique (though they shouldn't be in cand_set if we maintain disjointness)
            # Actually, cand_set contains nodes NOT in clique.
            # So we just need to intersect with adj[v].
            # And we must exclude v itself from future candidates (it's in clique now).
            # adj[v] does not contain v (since x!=y in graph construction).
            
            res = dfs(new_clique, new_cand)
            if res:
                return res
            
            # Backtrack: remove v from cand_set for the next iteration
            cand_set.remove(v)
            
        return None

    # Start with all high-degree nodes
    initial_cand = set(nodes[:3000])
    return dfs([], initial_cand)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    prime_sets = [
        (2, 3),
        (2, 3, 5),
        (2, 3, 5, 7),
        (2, 3, 5, 7, 11)
    ]
    
    best_solution = None
    best_cost = float('inf')
    
    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        # Generate smooth numbers
        smooth_nums = generate_smooth_numbers(P, MAX_VAL)
        
        # Build adjacency
        adj = build_adjacency(smooth_nums)
        
        # Find clique
        clique = find_clique(adj, N_NODES, deadline)
        
        if clique:
            # Verify
            cost = count_distinct_primes(clique)
            if cost < best_cost:
                best_cost = cost
                best_solution = clique[:]
                
            # If we found a solution with cost equal to len(P), it's optimal for this P.
            # If cost is 2, we can stop immediately as it's the theoretical minimum for distinct sums > 1.
            if cost <= 2:
                return clique
                
            # If cost is 3, it's very good. We might still check if a 2-prime solution exists in another set?
            # No, if P={2,3} failed to find a clique, there is no 2-prime solution using only 2 and 3.
            # But a 2-prime solution could use {2,5}? 
            # The problem asks to minimize the number of distinct primes.
            # If we find a set with primes {2,3}, cost is 2.
            # If we find a set with primes {2,3,5}, cost is 3.
            # We should try all small prime sets to find the global minimum.
            # However, finding a clique in a larger graph is harder.
            # If we found a clique for {2,3}, we are done.
            # If not, we try {2,3,5}. If we find one, cost is 3.
            # Could there be a set with cost 2 using {2,5}? Yes.
            # So we should try {2,5}, {2,7}, {3,5} etc?
            # Given time constraints, {2,3} is the most likely to have a dense graph.
            # {2,3,5} is next.
            # If we find a cost 3 solution, we can return it if we don't have time to check other pairs.
            # But let's stick to the ordered list.
            
    if best_solution:
        return best_solution
        
    # Fallback: Simulated Annealing
    # Initialize with a heuristic set
    # Multiples of 6: 6, 12, 18... Sums are multiples of 6.
    # Factors of 6k are 2, 3 and factors of k.
    # If k is 2,3 smooth, then sums are 2,3 smooth.
    # Let's try to pick 12 numbers from 2,3-smooth numbers * 6?
    # Or just start with random and anneal.
    
    best_arr = [3 + 6 * k for k in range(12)]
    # Ensure distinct and <= 10000
    best_arr = list(set(best_arr))
    while len(best_arr) < 12:
        best_arr.append(random.randint(1, 10000))
    best_arr = best_arr[:12]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.995
    
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

# Keep the interface
def construct():
    return _construct_impl()

# Alias to match the prompt's requirement if it expects 'construct' directly
# The prompt says "Keep the function names and signatures of the seed program."
# The seed program has `def construct():`.
# So I will just define `construct` directly.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    prime_sets = [
        (2, 3),
        (2, 3, 5),
        (2, 3, 5, 7),
        (2, 3, 5, 7, 11)
    ]
    
    best_solution = None
    best_cost = float('inf')
    
    for P in prime_sets:
        if time.time() > deadline - 5:
            break
            
        # Generate smooth numbers
        smooth_nums = generate_smooth_numbers(P, MAX_VAL)
        
        # Build adjacency
        adj = build_adjacency(smooth_nums)
        
        # Find clique
        clique = find_clique(adj, N_NODES, deadline)
        
        if clique:
            # Verify
            cost = count_distinct_primes(clique)
            if cost < best_cost:
                best_cost = cost
                best_solution = clique[:]
                
            if cost <= 2:
                return clique
                
    if best_solution:
        return best_solution
        
    # Fallback: Simulated Annealing
    best_arr = [3 + 6 * k for k in range(12)]
    best_arr = list(set(best_arr))
    while len(best_arr) < 12:
        best_arr.append(random.randint(1, 10000))
    best_arr = best_arr[:12]
    best_cost = count_distinct_primes(best_arr)
    
    current_arr = best_arr[:]
    current_cost = best_cost
    
    T = 5.0
    cooling_rate = 0.995
    
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
