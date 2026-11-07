import time
from math import isqrt, gcd
from itertools import permutations

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    def time_check():
        return time.time() < deadline
    
    N = 256
    
    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Calculate max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
        
    # Identify strong primes (those with M_p > p) and weak primes (M_p == p)
    # Actually, we just need to identify which primes are candidates for merging.
    # A prime p is a candidate if there exists a prime q such that p*q <= N 
    # and p*q > M_p + M_q.
    
    # Let's just check all pairs for potential merges
    # Since N is small, we can find the best matching for all primes
    # But the graph is bipartite-ish or small components.
    
    # Simpler approach:
    # 1. Start with all M_p
    # 2. Find all pairs (p, q) with gain > 0
    # 3. Solve max weight matching on the graph of these pairs
    
    # Given the analysis, only {7, 11, 13} interact with weak primes.
    # And weak primes don't interact with each other.
    # So we can just hardcode the logic or solve the small subgraph.
    
    # Let's solve the general max weight matching for the subgraph of positive gains.
    # Nodes are indices in primes list.
    
    n = len(primes)
    # gain[i][j]
    gain = [[0]*n for _ in range(n)]
    valid_edges = []
    
    for i in range(n):
        p = primes[i]
        for j in range(i+1, n):
            q = primes[j]
            prod = p * q
            if prod > N:
                break
            g = prod - prime_powers[p] - prime_powers[q]
            if g > 0:
                gain[i][j] = g
                gain[j][i] = g
                valid_edges.append((i, j, g))
                
    # The graph is sparse.
    # We can use a simple recursive search with memoization (bitmask) 
    # if the number of nodes involved is small.
    # However, some nodes (like 2, 3, 5) are not in any valid edge.
    # We only need to consider nodes that are in valid_edges.
    
    nodes_in_edges = set()
    for i, j, g in valid_edges:
        nodes_in_edges.add(i)
        nodes_in_edges.add(j)
        
    if not nodes_in_edges:
        result = [1] + [prime_powers[p] for p in primes]
        return result

    # Map original indices to compact indices for bitmask DP
    # Actually, the number of nodes in edges is small (7, 11, 13, 17, 19, 23, 29, 31...)
    # Let's just use a greedy + local search or exact match for this small set.
    
    # Since the graph is small, we can try all matchings.
    # But bitmask DP is robust.
    
    # Let's extract the subgraph
    subgraph_nodes = sorted(list(nodes_in_edges))
    node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}
    m = len(subgraph_nodes)
    
    # Build adjacency list and weights for the subgraph
    # adj[i] = list of (j, weight)
    adj = [[] for _ in range(m)]
    for i, j, g in valid_edges:
        ui = node_to_idx[i]
        uj = node_to_idx[j]
        adj[ui].append((uj, g))
        adj[uj].append((ui, g))
        
    # DP over bitmask of subgraph_nodes
    # State: mask of used nodes
    # But matching is not just "used", it's "paired".
    # Standard DP for matching:
    # dp[mask] = max weight matching for the subgraph induced by mask?
    # No, mask represents which nodes are ALREADY PAIRED/USED.
    # Or mask represents which nodes are AVAILABLE?
    
    # Let's use: dp[mask] = max gain obtainable from the set of nodes indicated by mask
    # where mask has 1s for nodes that are STILL AVAILABLE to be matched.
    # Base case: mask == 0 -> 0.
    # Transition: pick the first available node i (lowest set bit).
    # Option 1: Don't match i. Remove i from mask. dp[mask without i].
    # Option 2: Match i with some other available node j.
    #            Add gain[i][j] + dp[mask without i and without j].
    
    from functools import lru_cache
    
    # Convert adj to a 2D weight matrix for fast access in DP
    # w[i][j] = gain if edge exists, else -1 (or 0 if we handle separately)
    w = [[-1]*m for _ in range(m)]
    for i in range(m):
        for j, g in adj[i]:
            w[i][j] = g
            
    @lru_cache(maxsize=None)
    def dp(mask):
        if mask == 0:
            return 0
        
        # Find first available node
        i = 0
        while i < m:
            if mask & (1 << i):
                break
            i += 1
            
        if i >= m:
            return 0
            
        # Option 1: Leave i unmatched
        best = dp(mask ^ (1 << i))
        
        # Option 2: Match i with some j > i in mask
        # To avoid double counting, we only match i with j > i
        # But in bitmask, we just iterate all other bits in mask
        # However, matching (i,j) is same as (j,i).
        # By always picking the smallest i, we ensure each matching is counted once.
        
        rem_mask = mask ^ (1 << i)
        j = i + 1
        while j < m:
            if rem_mask & (1 << j):
                if w[i][j] > 0: # Edge exists and positive
                    # We can only match if they are connected
                    val = w[i][j] + dp(rem_mask ^ (1 << j))
                    if val > best:
                        best = val
            j += 1
            
        return best
        
    full_mask = (1 << m) - 1
    
    if not time_check():
        return [1] + [prime_powers[p] for p in primes]
    
    # Now we need to reconstruct which pairs were chosen to build the result.
    choice = [None] * (1 << m)
    
    def dp2(mask):
        if mask == 0:
            return 0, ()
        
        i = 0
        while i < m:
            if mask & (1 << i):
                break
            i += 1
            
        if i >= m:
            return 0, ()
            
        rem_mask = mask ^ (1 << i)
        
        # Option 1: Unmatched
        val1, pairs1 = dp2(rem_mask)
        best_val = val1
        best_pairs = pairs1
        best_choice = 'unmatched'
        
        # Option 2: Matched with j
        j = i + 1
        while j < m:
            if rem_mask & (1 << j):
                if w[i][j] > 0:
                    val2, pairs2 = dp2(rem_mask ^ (1 << j))
                    total = w[i][j] + val2
                    if total > best_val:
                        best_val = total
                        best_pairs = pairs2 + ((i, j),)
                        best_choice = (i, j)
            j += 1
            
        # Store choice for this mask
        choice[mask] = (best_choice, best_pairs)
        return best_val, best_pairs
        
    final_gain, _ = dp2(full_mask)
    
    # Reconstruct pairs
    # We need to traverse the choices
    # Actually, the tuple returned by dp2 contains the pairs from the subproblem.
    # The `best_pairs` in the winning branch contains the pairs from `dp2(rem_mask...)`.
    # So we just need to extract them.
    
    # Let's just re-extract by walking the tree?
    # No, the `best_pairs` returned by dp2(full_mask) IS the list of pairs (indices in subgraph).
    
    # Wait, the return type of dp2 was (val, pairs).
    # pairs is a tuple of pairs (i, j).
    
    # Let's get the pairs
    _, selected_pairs_sub = dp2(full_mask)
    
    # Map subgraph indices back to original prime indices
    selected_prime_pairs = []
    for i_sub, j_sub in selected_pairs_sub:
        i_orig = subgraph_nodes[i_sub]
        j_orig = subgraph_nodes[j_sub]
        selected_prime_pairs.append((i_orig, j_orig))
        
    # Construct result
    # Start with 1
    result = [1]
    
    # Set of original prime indices that are matched
    matched_indices = set()
    for i, j in selected_prime_pairs:
        matched_indices.add(i)
        matched_indices.add(j)
        
    # Add unmatched primes with their M_p
    for i, p in enumerate(primes):
        if i not in matched_indices:
            result.append(prime_powers[p])
            
    # Add matched products
    for i, j in selected_prime_pairs:
        p = primes[i]
        q = primes[j]
        result.append(p * q)
        
    # Final checks
    if len(set(result)) != len(result):
        # Fallback
        return [1] + [prime_powers[p] for p in primes]
        
    for x in result:
        if x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]
            
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]
                
    return result
