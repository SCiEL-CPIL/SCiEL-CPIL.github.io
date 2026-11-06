import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    Strategy: 
    1. Iterate n from 43 down to 30.
    2. For each n, try to build a C4-free graph with alpha < 11 using greedy edge addition.
    3. Start with empty graph, add edges if they don't create a C4.
    4. Check alpha periodically.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def has_c4(n, adj):
        """Check if graph has a C4 using bitmasks."""
        # adj is a list of integers (bitmasks)
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):  # at least 2 bits set
                    return True
        return False

    def can_add_edge(u, v, adj):
        """
        Check if adding edge (u, v) creates a C4.
        A C4 is created if there is a path u-a-b-v of length 3 in the current graph.
        This is equivalent to: there exists a in N(u) and b in N(v) such that a is adjacent to b.
        """
        # Get neighbors of u and v
        mask_u = adj[u]
        mask_v = adj[v]
        
        # We need to check if any neighbor of a node in mask_u is in mask_v.
        # Iterate through bits in mask_u
        m = mask_u
        while m:
            # Get lowest set bit
            lsb = m & -m
            a = lsb.bit_length() - 1
            m -= lsb
            
            # Check if a has any neighbor in mask_v
            # adj[a] & mask_v
            if adj[a] & mask_v:
                return False
        
        return True

    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        """
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time.time() > deadline - 5:
                return
            if candidates_mask == 0:
                best[0] = max(best[0], current_size)
                return
            
            # Upper bound: current_size + number of bits in candidates_mask
            # Popcount
            if current_size + bin(candidates_mask).count('1') <= best[0]:
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with highest degree in candidate subgraph
            # Or just pick the first one for speed
            # Let's pick the one with highest degree in the induced subgraph
            # To avoid O(n^2) per step, just pick the lowest index
            lsb = candidates_mask & -candidates_mask
            v = lsb.bit_length() - 1
            
            # Option 1: Include v
            # New candidates are those in candidates_mask not adjacent to v
            new_candidates = candidates_mask & ~adj[v] & ~(1 << v)
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude v
            branch(candidates_mask & ~(1 << v), current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def verify_graph(n, adj):
        """Verify C4-free and alpha < 11."""
        if has_c4(n, adj):
            return False
        alpha = max_independent_set_size(n, adj)
        return alpha < 11

    best_graph = None
    
    # Try n from 43 down to 30
    for n in range(43, 29, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple random restarts
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
                
            random.seed(12345 + attempt)
            
            # Initialize empty graph
            adj = [0] * n
            
            # List of all possible edges
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            # Greedy add edges
            for u, v in edges:
                if time.time() > deadline - 5:
                    break
                if can_add_edge(u, v, adj):
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            # Check if this graph is valid
            if verify_graph(n, adj):
                # Convert adj to edge list
                edge_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edge_list.append([i, j])
                best_graph = {'n': n, 'edges': edge_list}
                found = True
                break
        
        if found:
            break

    if best_graph is not None:
        return best_graph
    
    # Fallback: return a known valid small graph (e.g., n=30)
    # The seed program's fallback was 30 vertices with 10 triangles.
    # Let's return that to be safe, though it should be beatable.
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
