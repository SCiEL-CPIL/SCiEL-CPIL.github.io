import time
import random
import math
import sys

# Precompute Smallest Prime Factors (SPF) for fast factorization
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

def is_smooth(n, primes_set):
    """Check if n is composed only of primes in primes_set."""
    if n < 1:
        return False
    for p in primes_set:
        while n % p == 0:
            n //= p
    return n == 1

def find_clique_for_primes(primes_set, max_k, target_size=12, time_limit=10.0):
    """
    Find a clique of size target_size in the graph where nodes are 1..max_k
    and an edge exists if u+v is composed only of primes in primes_set.
    """
    start_time = time.time()
    
    # Precompute valid sums
    # Max sum is 2 * max_k
    max_sum = 2 * max_k
    valid_sum = [False] * (max_sum + 1)
    for s in range(2, max_sum + 1):
        if is_smooth(s, primes_set):
            valid_sum[s] = True
            
    # Build adjacency list
    # Nodes are 1..max_k
    adj = [[] for _ in range(max_k + 1)]
    for u in range(1, max_k + 1):
        for v in range(u + 1, max_k + 1):
            if valid_sum[u + v]:
                adj[u].append(v)
                adj[v].append(u)
                
    # Heuristic: Order nodes by degree (descending) to find clique faster
    nodes = list(range(1, max_k + 1))
    nodes.sort(key=lambda x: len(adj[x]), reverse=True)
    
    # Map original node to new index for efficiency?
    # Just use the sorted list for backtracking.
    
    best_clique = []
    
    def backtrack(current_clique, available_nodes):
        nonlocal best_clique
        if len(current_clique) == target_size:
            best_clique = current_clique[:]
            return True
        if time.time() - start_time > time_limit:
            return False
            
        # Pruning: If we can't reach target_size with available nodes
        if len(current_clique) + len(available_nodes) < target_size:
            return False
            
        # Try adding nodes from available_nodes
        # We want to pick nodes that connect to all in current_clique
        # available_nodes already filtered to connect to last added? 
        # Standard Bron-Kerbosch variant or simple recursive search.
        
        for i, node in enumerate(available_nodes):
            # Check if node connects to all in current_clique
            # Optimization: If we maintain 'available' as intersection of neighbors, 
            # then all nodes in 'available' connect to all in 'current_clique'.
            
            new_clique = current_clique + [node]
            # Find neighbors of 'node' that are also in 'available_nodes' after 'i'
            # to avoid duplicates
            new_available = []
            for j in range(i + 1, len(available_nodes)):
                candidate = available_nodes[j]
                if candidate in adj[node]:
                    new_available.append(candidate)
                    
            if backtrack(new_clique, new_available):
                return True
                
        return False

    # Initial available nodes
    initial_available = nodes[:]
    
    # To speed up, we can start with a high-degree node
    # But simple backtrack on sorted list is okay for 333 nodes if solution exists.
    # However, 333 nodes might be too large for naive backtrack if density is low.
    # Let's add a randomization or restart if it's slow?
    # Or use a greedy heuristic to find a large clique first?
    
    # Greedy Heuristic
    greedy_clique = []
    remaining = set(nodes)
    while len(greedy_clique) < target_size:
        if not remaining:
            break
        # Pick node with max degree in induced subgraph of 'remaining'
        best_node = None
        max_deg = -1
        for n in remaining:
            deg = sum(1 for m in remaining if m in adj[n])
            if deg > max_deg:
                max_deg = deg
                best_node = n
        if best_node is None:
            break
        greedy_clique.append(best_node)
        remaining.remove(best_node)
        # Filter remaining to only those connected to best_node
        remaining = {n for n in remaining if best_node in adj[n]}
        
    if len(greedy_clique) >= target_size:
        best_clique = greedy_clique[:target_size]
        return best_clique

    # If greedy failed, try backtrack with time limit
    # The backtrack above is exponential. For 333 nodes, it might be slow.
    # Let's try a few random restarts of the greedy or a limited backtrack.
    
    # Given the time constraint, if greedy fails, we might not find it.
    # But for 2 or 3 primes, the graph is likely dense enough.
    
    if best_clique:
        return best_clique
    return None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # Try to find a set with 2 primes {2, 3}
    # k_i <= 10000 / 6 = 1666? 
    # If a_i = 6 * k_i, sum = 6(k_i+k_j). Primes 2,3.
    # k_i <= 1666.
    # Let's try max_k = 1000 first for speed.
    
    # If 2 primes fail, try 3 primes {2, 3, 5}
    # a_i = 30 * k_i. k_i <= 333.
    
    # We want to minimize the number of primes.
    # So we try 2 primes first.
    
    # Strategy:
    # 1. Try to find clique of size 12 for primes {2, 3} with max_k = 1000.
    # 2. If found, return [6*k for k in clique].
    # 3. If not found, try primes {2, 3, 5} with max_k = 333.
    # 4. If found, return [30*k for k in clique].
    # 5. If not found, fallback to current SA or random search.
    
    # Note: The graph search might be slow if no clique exists.
    # We set a time limit for each search.
    
    # Search 1: 2 primes
    clique_2 = find_clique_for_primes({2, 3}, 1000, 12, time_limit=10.0)
    if clique_2:
        arr = [6 * k for k in clique_2]
        # Verify
        if len(set(arr)) == 12 and all(a <= 10000 for a in arr):
            return arr
        # If invalid (shouldn't happen), continue
        
    # Search 2: 3 primes
    clique_3 = find_clique_for_primes({2, 3, 5}, 333, 12, time_limit=20.0)
    if clique_3:
        arr = [30 * k for k in clique_3]
        if len(set(arr)) == 12 and all(a <= 10000 for a in arr):
            return arr
            
    # Search 3: 4 primes {2, 3, 5, 7}
    # a_i = 210 * k_i. k_i <= 47.
    # Max sum 94.
    clique_4 = find_clique_for_primes({2, 3, 5, 7}, 47, 12, time_limit=5.0)
    if clique_4:
        arr = [210 * k for k in clique_4]
        if len(set(arr)) == 12 and all(a <= 10000 for a in arr):
            return arr
            
    # Fallback: Randomized Search / Simulated Annealing
    # Use the logic from the previous program as a fallback
    # But since we have a deadline, we just do a quick random check or return a known valid set.
    
    # Known valid set with few primes?
    # If we can't find a small prime set, we just return a random valid set.
    # But we want to minimize.
    # Let's try a quick SA for 5 seconds.
    
    if time.time() < deadline - 5.0:
        # Quick SA
        best_arr = [3 + 6 * k for k in range(12)]
        best_cost = count_distinct_primes(best_arr)
        current_arr = best_arr[:]
        current_cost = best_cost
        T = 10.0
        cooling_rate = 0.999
        
        while time.time() < deadline - 5.0:
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
        
    return [3 + 6 * k for k in range(12)]
# EVOLVE-BLOCK-END
