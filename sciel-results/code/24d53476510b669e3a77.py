import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    max_modulus = 128
    
    # Focus on even moduli as they form the largest pairwise non-coprime set
    moduli = [m for m in range(2, max_modulus + 1) if m % 2 == 0]
    
    # Generate all (r, m) pairs
    nodes = []
    node_index = {}
    for m in moduli:
        for r in range(m):
            idx = len(nodes)
            nodes.append((r, m))
            node_index[(r, m)] = idx
            
    n_nodes = len(nodes)
    
    # Precompute adjacency masks
    # adj[i] is a bitmask where bit j is 1 if node i and node j CONFLICT (intersect)
    adj = [0] * n_nodes
    
    # Optimization: Group by modulus to speed up gcd calculation
    # For each pair of moduli, compute gcd once.
    # But N is ~4000, N^2 is 16M. GCD is fast.
    # 16M GCDs in Python might take ~1-2 seconds. Acceptable.
    
    for i in range(n_nodes):
        r1, m1 = nodes[i]
        mask = 0
        for j in range(i + 1, n_nodes):
            r2, m2 = nodes[j]
            d = gcd(m1, m2)
            # Check if r1 % d == r2 % d
            if r1 % d == r2 % d:
                mask |= (1 << j)
                adj[j] |= (1 << i)
        adj[i] = mask
        
    # Branch and Bound
    best_size = 0
    best_nodes = []
    
    # Initial greedy solution to set a good lower bound
    # Simple greedy: pick node, remove neighbors
    available = (1 << n_nodes) - 1
    greedy_set = []
    while available:
        # Pick first available node
        # For better heuristic, pick node with highest degree in available subgraph
        # But for speed, just pick lowest index or random
        # Let's pick the one with max degree in the available subgraph
        max_deg = -1
        best_v = -1
        v = 0
        while available:
            if available & 1:
                # Count bits in (adj[v] & available)
                deg = bin(adj[v] & available).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            available >>= 1
            v += 1
            
        if best_v == -1:
            break
            
        greedy_set.append(best_v)
        # Remove best_v and its neighbors
        available &= ~(1 << best_v)
        available &= ~adj[best_v]
        
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_nodes = greedy_set.copy()
        
    # Recursive BB
    def bb(available_mask, current_set, depth):
        nonlocal best_size, best_nodes
        
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_nodes = current_set.copy()
            return
            
        # Bound: Greedy Coloring
        # We need to color the graph induced by available_mask
        # Number of colors is an upper bound on the independent set size
        remaining = available_mask
        num_colors = 0
        while remaining:
            num_colors += 1
            # Pick a vertex to start a color class
            # Heuristic: pick vertex with max degree in remaining
            max_deg = -1
            start_v = -1
            r = remaining
            v = 0
            while r:
                if r & 1:
                    deg = bin(adj[v] & remaining).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        start_v = v
                r >>= 1
                v += 1
                
            if start_v == -1:
                break
                
            # Assign color to start_v and all non-adjacent vertices?
            # No, greedy coloring:
            # Color class C starts with start_v.
            # We can add any vertex u to C if u is not adjacent to any vertex in C.
            # To simplify, we just pick one vertex, remove it and its neighbors from 'remaining' for THIS color?
            # No, that's not coloring. That's independent set packing.
            # For the bound, we need a proper coloring.
            # Standard MCS bound:
            # Order vertices. Color greedily.
            
            # Let's use a simpler bound: The size of the maximum clique in the complement?
            # Or just use the number of vertices / average degree?
            # Actually, for sparse graphs, the bound is tight.
            # For dense graphs, the bound is loose.
            # Here, the graph is dense.
            
            # Let's try a simple bound: len(current_set) + popcount(available_mask)
            # This is always true but weak.
            
            # Better bound:
            # Find a clique in the available subgraph.
            # Any independent set can have at most 1 vertex from a clique.
            # So if we find a clique of size K, the max independent set size is <= (N_avail - K) + 1? No.
            # It's <= N_avail - (K - 1).
            
            # Let's stick to a simple heuristic bound to avoid overhead.
            # If we are in a dense region, the independent set is small.
            
            # Let's just use the count of available nodes as the bound for now, 
            # but prune if len(current_set) + popcount(available_mask) <= best_size.
            
            if len(current_set) + bin(available_mask).count('1') <= best_size:
                return
                
            # Pick a vertex to branch on
            # Heuristic: Max degree in available subgraph
            max_deg = -1
            branch_v = -1
            r = available_mask
            v = 0
            while r:
                if r & 1:
                    deg = bin(adj[v] & available_mask).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        branch_v = v
                r >>= 1
                v += 1
                
            if branch_v == -1:
                return
                
            # Branch 1: Include branch_v
            new_available = available_mask & ~(1 << branch_v) & ~adj[branch_v]
            current_set.append(branch_v)
            bb(new_available, current_set, depth + 1)
            current_set.pop()
            
            # Branch 2: Exclude branch_v
            new_available = available_mask & ~(1 << branch_v)
            bb(new_available, current_set, depth + 1)
            
    # Start BB
    # To make it faster, we can order the nodes initially
    # But the bitset approach handles it.
    
    # We need to be careful with recursion depth. n_nodes ~ 4000.
    # Python default recursion limit is 1000.
    # We should increase it or use an iterative stack.
    # For now, let's increase recursion limit.
    import sys
    sys.setrecursionlimit(10000)
    
    bb((1 << n_nodes) - 1, [], 0)
    
    # Convert best_nodes back to (r, m) pairs
    result = [nodes[i] for i in best_nodes]
    
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            r1, m1 = result[i]
            r2, m2 = result[j]
            d = gcd(m1, m2)
            if r1 % d == r2 % d:
                # Invalid
                return []
                
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} classes")
