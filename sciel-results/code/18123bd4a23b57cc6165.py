"""Combined approach: Clebsch graph seed + efficient bitset 3-color check + greedy edge addition."""
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n = 16
    
    # Build Clebsch graph: vertices 0..15, edge (u,v) iff Hamming distance of binary reps is 1 or 4
    # Binary representation: u = b3*b2*b1*b0 (4 bits)
    # Hamming distance 1: differ in exactly 1 bit
    # Hamming distance 4: differ in all 4 bits (complement)
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                edges.add((u, v))
    
    # Build adjacency as bitmasks for fast operations
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    def is_triangle_free(adj_mask, n):
        """Check triangle-free using bitmasks."""
        for u in range(n):
            # For each pair of neighbors of u, check they're not connected
            neighbors = adj_mask[u]
            while neighbors:
                lsb = neighbors & (-neighbors)
                v = lsb.bit_length() - 1
                neighbors ^= lsb
                # Check if v and any other neighbor of u are connected
                if adj_mask[v] & adj_mask[u]:
                    return False
        return True
    
    def is_3_colorable(adj_mask, n):
        """Check 3-colorability via backtracking with bitmasks and vertex ordering."""
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Get forbidden colors from colored neighbors
            forbidden = 0
            for u in range(n):
                if (adj_mask[v] >> u) & 1:
                    c = colors[u]
                    if c != -1:
                        forbidden |= (1 << c)
            for c in range(3):
                if not (forbidden >> c) & 1:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)
    
    # Verify initial graph
    assert is_triangle_free(adj, n), "Clebsch should be triangle-free"
    assert not is_3_colorable(adj, n), "Clebsch should not be 3-colorable"
    
    best_edges = set(edges)
    best_count = len(best_edges)
    
    # Local search: try to add edges greedily
    def try_add_edges(current_edges, adj_mask, deadline):
        """Try to add edges while maintaining triangle-free and non-3-colorable."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Collect candidate edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in current_edges:
                        # Check triangle-free: no common neighbor
                        common = adj_mask[u] & adj_mask[v]
                        if not common:
                            candidates.append((u, v))
            
            # Try adding each candidate
            for u, v in candidates:
                if time.time() > deadline - 5:
                    break
                new_edges = current_edges | {(u, v)}
                new_adj = adj_mask[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                
                if is_triangle_free(new_adj, n):
                    if not is_3_colorable(new_adj, n):
                        current_edges = new_edges
                        adj_mask = new_adj
                        changed = True
                        break
        
        return current_edges, adj_mask
    
    # Phase 1: Greedy edge addition
    current_edges, adj = try_add_edges(edges, adj, deadline - 30)
    best_edges = current_edges
    best_count = len(best_edges)
    
    # Phase 2: Try multiple random restarts with different edge removals
    if time.time() < deadline - 20:
        import random
        random.seed(42)
        
        # Try removing one edge and re-adding
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to remove
            edge_list = list(current_edges)
            if not edge_list:
                break
            remove_edge = random.choice(edge_list)
            test_edges = current_edges - {remove_edge}
            
            # Rebuild adj
            test_adj = [0] * n
            for u, v in test_edges:
                test_adj[u] |= (1 << v)
                test_adj[v] |= (1 << u)
            
            # Try to add edges from scratch
            new_edges, new_adj = try_add_edges(test_edges, test_adj, deadline - 10)
            
            if len(new_edges) > best_count:
                best_edges = new_edges
                best_count = len(new_edges)
                current_edges = new_edges
                adj = new_adj
    
    # Phase 3: Final greedy pass
    if time.time() < deadline - 10:
        current_edges, adj = try_add_edges(best_edges, adj, deadline - 5)
        best_edges = current_edges
        best_count = len(best_edges)
    
    return [list(e) for e in best_edges]
