import time
import numpy as np

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
factors_cache = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors_cache[i].add(p)
        while n % p == 0:
            n //= p

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= factors_cache[s]
    return len(all_primes)

def is_smooth(n, target_primes):
    """Check if n is smooth with respect to target_primes."""
    if n <= 1:
        return True
    return factors_cache[n].issubset(target_primes)

def find_clique_fast(target_primes, clique_size, deadline):
    """
    Fast clique finder: build full compatibility matrix, use greedy+backtracking.
    """
    # Step 1: Build smooth sum indicator
    is_smooth_sum = np.zeros(MAX_SUM + 1, dtype=bool)
    for s in range(2, MAX_SUM + 1):
        if factors_cache[s].issubset(target_primes):
            is_smooth_sum[s] = True
    
    # Step 2: Compute degree of each node (1..10000)
    # Node i is compatible with node j if is_smooth_sum[i+j]
    # Degree of i = count of j in [1,10000], j!=i, such that is_smooth_sum[i+j]
    
    # Use numpy for fast computation
    smooth_sums = is_smooth_sum  # bool array
    
    # For each i, count how many j in 1..10000 have smooth sum i+j
    # i+j ranges from i+1 to i+10000
    degrees = np.zeros(10001, dtype=np.int32)
    
    for i in range(1, 10001):
        lo = i + 1
        hi = min(i + 10000, MAX_SUM)
        if lo <= hi:
            cnt = np.sum(smooth_sums[lo:hi+1])
            # Subtract if 2*i is in range and smooth (self-sum)
            two_i = 2 * i
            if lo <= two_i <= hi and smooth_sums[two_i]:
                cnt -= 1
            degrees[i] = cnt
    
    # Filter nodes with sufficient degree
    min_deg = clique_size - 1
    valid_mask = degrees >= min_deg
    valid_nodes = np.where(valid_mask)[0]
    valid_nodes = valid_nodes[valid_nodes >= 1]  # exclude 0
    
    if len(valid_nodes) < clique_size:
        return None
    
    # Take top candidates by degree
    if len(valid_nodes) > 500:
        top_idx = np.argsort(degrees[valid_nodes])[::-1][:500]
        valid_nodes = valid_nodes[top_idx]
    
    n_sub = len(valid_nodes)
    
    # Build adjacency matrix
    adj = np.zeros((n_sub, n_sub), dtype=bool)
    for u_idx in range(n_sub):
        u = valid_nodes[u_idx]
        for v_idx in range(u_idx + 1, n_sub):
            v = valid_nodes[v_idx]
            if is_smooth_sum[u + v]:
                adj[u_idx, v_idx] = True
                adj[v_idx, u_idx] = True
    
    # Compute subgraph degrees
    sub_deg = adj.sum(axis=1)
    
    # Greedy + backtracking clique search
    # Order nodes by degree (descending)
    order = np.argsort(sub_deg)[::-1]
    
    # Greedy initialization
    chosen = []
    available = set(range(n_sub))
    
    # Try greedy first
    while len(chosen) < clique_size and available:
        # Pick node with highest degree among available
        best = -1
        best_deg = -1
        for node in available:
            if sub_deg[node] > best_deg:
                best_deg = sub_deg[node]
                best = node
        if best == -1:
            break
        chosen.append(best)
        # Remove non-neighbors
        new_available = set()
        for node in available:
            if node != best and adj[best, node]:
                new_available.add(node)
        available = new_available
    
    if len(chosen) == clique_size:
        return [int(valid_nodes[i]) for i in chosen]
    
    # Backtracking with branch and bound
    # Use the greedy solution as initial upper bound
    best_clique_size = len(chosen)
    
    # Recursive backtracking
    def backtrack(chosen_set, candidates, depth):
        nonlocal best_clique_size
        
        if len(chosen_set) == clique_size:
            return chosen_set[:]
        
        if not candidates:
            return None
        
        # Upper bound: current size + max possible from candidates
        if len(chosen_set) + len(candidates) < clique_size:
            return None
        
        # Try each candidate
        for i in range(len(candidates)):
            node = candidates[i]
            new_chosen = chosen_set + [node]
            
            # New candidates: neighbors of node that come after i
            new_cands = []
            for j in range(i + 1, len(candidates)):
                other = candidates[j]
                if adj[node, other]:
                    new_cands.append(other)
            
            result = backtrack(new_chosen, new_cands, depth + 1)
            if result is not None:
                return result
        return None
    
    # Order initial candidates by degree
    initial_cands = [int(i) for i in order]
    
    try:
        result = backtrack([], initial_cands, 0)
        if result is not None:
            return [int(valid_nodes[i]) for i in result]
    except RecursionError:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # List of first few primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Try increasing k (number of primes)
    for k in range(4, 10):
        if time.time() > deadline - 5:
            break
        
        target_primes = set(primes[:k])
        
        clique = find_clique_fast(target_primes, 12, deadline)
        if clique:
            return clique
    
    # Fallback: local search
    best_score = 999
    best_arr = list(range(1, 13))
    best_score = count_distinct_primes(best_arr)
    
    import random
    random.seed(42)
    
    current_arr = best_arr[:]
    current_score = best_score
    
    for _ in range(2000):
        if time.time() > deadline - 2:
            break
        
        idx = random.randint(0, 11)
        old_val = current_arr[idx]
        new_val = random.randint(1, 10000)
        if new_val in current_arr and new_val != old_val:
            continue
        
        current_arr[idx] = new_val
        score = count_distinct_primes(current_arr)
        
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_arr = current_arr[:]
        else:
            current_arr[idx] = old_val
    
    return best_arr
