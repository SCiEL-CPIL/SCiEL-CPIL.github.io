"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _build_conflict_graph(n=512):
    """Build adjacency bitsets for the conflict graph.
    Vertices are integers 1..n-1 (i.e., 1..511 for n=512).
    Edge between a and b if gcd(a,b) > 1.
    Returns a list of int bitsets where bit i is set if vertex i+1 conflicts with vertex i+1.
    """
    size = n - 1  # 511 vertices
    # adj[i] is a bitmask of vertices (0-indexed) that conflict with vertex i (0-indexed, corresponding to value i+1)
    adj = [0] * size
    
    # For each prime p < n, all multiples of p in [1, n-1] conflict with each other
    # Actually, two numbers conflict if they share a common factor > 1, which means they share a prime factor.
    # So for each prime p, the set of multiples of p in [1, n-1] forms a clique in the conflict graph.
    
    # Sieve for primes up to n-1
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n) if is_prime[i]]
    
    for p in primes:
        # Multiples of p in [1, n-1]: p, 2p, 3p, ..., floor((n-1)/p)*p
        multiples = []
        k = 1
        while k * p < n:
            multiples.append(k * p - 1)  # 0-indexed
            k += 1
        m = len(multiples)
        for i in range(m):
            for j in range(i + 1, m):
                a_idx = multiples[i]
                b_idx = multiples[j]
                adj[a_idx] |= (1 << b_idx)
                adj[b_idx] |= (1 << a_idx)
    
    return adj

def _branch_and_bound(n=512, time_limit=140.0):
    """Solve Maximum Weight Independent Set via branch-and-bound with bitset pruning."""
    size = n - 1  # 511
    start_time = time.time()
    
    # Build conflict graph
    adj = _build_conflict_graph(n)
    
    # Weights: weight of vertex i (0-indexed, value i+1) is 1/(n - (i+1)) = 1/(511 - i)
    weights = [1.0 / (n - (i + 1)) for i in range(size)]
    
    # Order vertices by weight descending for better pruning
    order = list(range(size))
    order.sort(key=lambda i: weights[i], reverse=True)
    
    # Remap: new index j corresponds to old index order[j]
    remap = [0] * size
    for j, i in enumerate(order):
        remap[i] = j
    
    # Build new adjacency in remapped order
    new_adj = [0] * size
    for i in range(size):
        old_mask = adj[i]
        new_mask = 0
        # For each bit set in old_mask, map to new index
        m = old_mask
        while m:
            lowbit = m & (-m)
            j = lowbit.bit_length() - 1  # old index
            new_mask |= (1 << remap[j])
            m ^= lowbit
        new_adj[remap[i]] = new_mask
    
    new_weights = [weights[order[i]] for i in range(size)]
    
    # Global best
    best_value = 0.0
    best_set = []
    
    # Fractional upper bound using greedy coloring (for branch and bound)
    # Use a simple approach: sort by weight, use a greedy independent set bound
    
    def bound(available_mask, current_value):
        """Compute an upper bound on the additional weight from available_mask.
        Uses a greedy approach: repeatedly pick the vertex with highest weight
        and add its weight, then remove it and its neighbors.
        This gives an upper bound since we're relaxing the independence constraint partially.
        Actually, a better bound: sort available vertices by weight descending,
        and for each, if it's not adjacent to any previously selected in the bound, add its weight.
        This is still an upper bound (not necessarily tight but valid).
        """
        if available_mask == 0:
            return current_value
        
        # Greedy bound: process vertices in order of decreasing weight
        # We'll simulate selecting vertices greedily (this is an upper bound on the optimum
        # because we're not enforcing independence strictly in a way that could underestimate... 
        # actually this greedy gives a feasible independent set, so it's a lower bound.
        # For an upper bound, we need something different.
        
        # Simple upper bound: sum of all weights in available_mask
        # This is valid but loose. Let's use a better one.
        
        # Upper bound via: for each vertex in available, its weight plus... 
        # Actually, a standard bound for MWIS: use the fact that if we take a vertex v,
        # we can't take its neighbors. A simple bound is sum of weights of available vertices
        # that are not adjacent to each other... but that's the problem itself.
        
        # Let's use: upper_bound = current_value + sum of weights of all available vertices
        # This is valid (since we can't take negative weights) but loose.
        
        # Better: use a coloring-based bound. Color the subgraph induced by available vertices.
        # The number of colors needed is an upper bound on the size of a clique, but for MWIS
        # we need something else.
        
        # For now, use the sum of weights as a loose upper bound. With good ordering, 
        # the branch and bound should still be effective.
        total = current_value
        m = available_mask
        while m:
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            total += new_weights[idx]
            m ^= lowbit
        return total
    
    # Recursive branch and bound
    # State: available_mask (bitset of vertices still available), current_set (list of chosen vertex indices)
    # We branch on the first available vertex: include it or exclude it.
    
    def dfs(available_mask, current_set, current_value, depth=0):
        nonlocal best_value, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        if available_mask == 0:
            if current_value > best_value:
                best_value = current_value
                best_set = list(current_set)
            return
        
        # Upper bound
        ub = bound(available_mask, current_value)
        if ub <= best_value:
            return
        
        # Pick a vertex to branch on. Choose the one with highest weight among available.
        m = available_mask
        best_idx = -1
        best_w = -1
        while m:
            lowbit = m & (-m)
            idx = lowbit.bit_length() - 1
            if new_weights[idx] > best_w:
                best_w = new_weights[idx]
                best_idx = idx
            m ^= lowbit
        
        if best_idx < 0:
            return
        
        # Option 1: Include best_idx
        new_available = available_mask & ~(1 << best_idx) & ~new_adj[best_idx]
        current_set.append(best_idx)
        dfs(new_available, current_set, current_value + new_weights[best_idx], depth + 1)
        current_set.pop()
        
        # Option 2: Exclude best_idx
        new_available = available_mask & ~(1 << best_idx)
        dfs(new_available, current_set, current_value, depth + 1)
    
    dfs((1 << size) - 1, [], 0.0)
    
    # Convert back to original values
    result = []
    for idx in best_set:
        result.append(order[idx] + 1)
    
    return result, best_value

def construct():
    start = time.time()
    result, value = _branch_and_bound(n=512, time_limit=140.0)
    return result
# EVOLVE-BLOCK-END
