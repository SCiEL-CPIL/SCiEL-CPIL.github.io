"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 512
    
    # Precompute prime factorizations for 1..N
    # factors[i] will be a dict {prime: exponent}
    factors = [None] * (N + 1)
    factors[1] = {}
    for i in range(2, N + 1):
        x = i
        fac = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                fac[d] = fac.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            fac[x] = fac.get(x, 0) + 1
        factors[i] = fac
    
    # Precompute LCM matrix
    def lcm(a, b):
        # a and b are integers
        g = math.gcd(a, b)
        return a * b // g
    
    # Build conflict graph: edge between i,j if lcm(i,j) > N
    # We want maximum independent set in this graph (subset with no edges)
    # i.e., subset where all pairwise lcms <= N
    
    # Use bitsets for efficiency
    # nodes are 1..N
    # neighbors[i] is a bitset of j such that lcm(i,j) > N
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) > N:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
    
    # Maximum Weight Independent Set via branch and bound
    # We want to maximize cardinality, so all weights = 1
    
    # Use a greedy coloring bound for upper bound estimation
    # Color the remaining subgraph greedily; number of colors is an upper bound on IS size
    
    def greedy_color_bound(remaining, adj):
        """
        remaining: set of nodes still to consider
        adj: dict or list of neighbors (for the induced subgraph on remaining)
        Returns upper bound on independent set size in the induced subgraph.
        """
        if not remaining:
            return 0
        
        # Order nodes by degree in remaining (descending)
        nodes = list(remaining)
        # Sort by number of neighbors in remaining (descending)
        def degree_in_remaining(v):
            return bin(adj[v] & ((1 << 1024) - 1)).count('b')  # This won't work, need proper bitset
            # Actually, let's just count bits in adj[v] & remaining_mask
        
        # Build remaining mask
        rem_mask = 0
        for v in remaining:
            rem_mask |= (1 << v)
        
        def deg(v):
            return bin(adj[v] & rem_mask).count('1')
        
        nodes.sort(key=deg, reverse=True)
        
        # Greedy coloring
        color = {}
        next_color = 0
        for v in nodes:
            # Find smallest color not used by neighbors of v in remaining
            used_colors = set()
            for u in nodes:
                if u == v:
                    continue
                if u in color and (adj[v] >> u) & 1:
                    used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
            if c > next_color:
                next_color = c
        return next_color + 1
    
    # Better: use a standard maximum independent set branch and bound
    # Let's use a recursive approach with bitsets
    
    # Convert to 0-indexed for easier bit manipulation
    # Map node i (1..N) to index i-1 (0..N-1)
    # adj_bitset[i] is a bitset where bit j is 1 if node (i+1) and node (j+1) are connected
    
    adj = [0] * N  # adj[i] corresponds to node i+1
    for i in range(1, N + 1):
        # Convert neighbors[i] to 0-indexed
        n = neighbors[i]
        a = 0
        for j in range(1, N + 1):
            if (n >> j) & 1:
                a |= (1 << (j - 1))
        adj[i - 1] = a
    
    # Maximum independent set on graph with N nodes
    # Use branch and bound with coloring
    
    # We'll use a recursive solver
    # State: available nodes (bitset), current set size, best found
    
    best_size = 0
    best_set = set()
    
    # Initial lower bound: greedy heuristic
    # Start with empty set, greedily add nodes
    def greedy_is():
        avail = (1 << N) - 1
        result = set()
        while avail:
            # Pick node with minimum degree in available subgraph
            best_v = -1
            best_deg = N + 1
            for v in range(N):
                if (avail >> v) & 1:
                    d = bin(adj[v] & avail).count('1')
                    if d < best_deg:
                        best_deg = d
                        best_v = v
            if best_v == -1:
                break
            result.add(best_v + 1)  # convert back to 1-indexed
            avail &= ~(1 << best_v)
            avail &= ~adj[best_v]
        return result
    
    greedy_result = greedy_is()
    best_size = len(greedy_result)
    best_set = set(greedy_result)
    
    # Check time
    if time.time() > deadline - 10:
        return list(best_set)
    
    # Branch and bound
    # Use a stack-based DFS
    # State: (available_mask, current_size, current_set)
    # To save memory, we can pass current_set as a frozenset or just track size and reconstruct
    
    # For efficiency, let's use a recursive function with global best
    
    def solve(available, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline - 2:
            return
        
        if not available:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound: current_size + popcount(available)
        # Tighter: use greedy coloring on available subgraph
        # For now, use a simple bound
        remaining_count = bin(available).count('1')
        if current_size + remaining_count <= best_size:
            return
        
        # Pick a node to branch on
        # Heuristic: pick node with maximum degree in available
        best_v = -1
        best_deg = -1
        for v in range(N):
            if (available >> v) & 1:
                d = bin(adj[v] & available).count('1')
                if d > best_deg:
                    best_deg = d
                    best_v = v
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_available = available & ~(1 << best_v) & ~adj[best_v]
        solve(new_available, current_size + 1, current_set | {best_v + 1})
        
        # Branch 2: exclude best_v
        new_available = available & ~(1 << best_v)
        solve(new_available, current_size, current_set)
    
    # Start with all nodes available
    all_nodes = (1 << N) - 1
    solve(all_nodes, 0, set())
    
    # Final check
    if time.time() > deadline - 1:
        pass
    
    return list(best_set)

# EVOLVE-BLOCK-END
