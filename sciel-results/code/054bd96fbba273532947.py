import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 100  # Leave 20s buffer
    
    # Helper: Check if graph is 3-colorable
    def is_3_colorable(adj):
        # adj is a list of sets or bitmasks
        # Using bitmask for speed
        colors = [-1] * n
        # Order vertices by degree descending for better pruning
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Try colors 0, 1, 2
            # Available colors are those not used by neighbors
            used = 0
            for u in range(n):
                if colors[u] != -1 and (adj[v] >> u) & 1:
                    used |= (1 << colors[u])
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        return common != 0

    # Helper: Generate a random maximal triangle-free graph
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        # Random order of all possible edges
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    # Helper: Local search improvement
    def local_search(adj, edges):
        current_edges_count = len(edges)
        # Try removing one edge and re-optimizing
        # We do this a few times
        for _ in range(20):
            if len(edges) == 0: break
            # Pick a random edge to remove
            idx = random.randint(0, len(edges)-1)
            u, v = edges[idx]
            
            # Remove it
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop(idx)
            
            # Now greedily add edges back
            # Candidates are all edges not in the graph
            # To be fast, just iterate all possible and add if valid
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            added = 0
            for u2, v2 in possible_edges:
                if not ((adj[u2] >> v2) & 1): # If not already present
                    if not creates_triangle(adj, u2, v2):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        edges.append([u2, v2])
                        added += 1
            
            # If we didn't gain edges, this move was bad, but we can't easily undo
            # So we just keep the new state. It might be a local opt.
            # To avoid getting stuck in bad states, we could revert if len(edges) < current_edges_count
            # But let's just proceed. The random restarts handle the global search.
            
        return adj, edges

    # Initialize with Clebsch graph as baseline
    # Clebsch graph vertices: 5-bit even parity
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges_count = len(clebsch_edges)
    best_edges = clebsch_edges
    best_adj = clebsch_adj

    # Verify baseline
    if not is_3_colorable(best_adj):
        pass # It is 4-chromatic
    else:
        # Should not happen, but if it does, we have a problem
        best_edges_count = 0
        best_edges = []
        best_adj = [0]*n

    # Main search loop
    while time.time() < deadline:
        # Random restart
        adj, edges = random_maximal_tf()
        
        # Local search
        adj, edges = local_search(adj, edges)
        
        # Check validity
        if len(edges) > best_edges_count:
            if not is_3_colorable(adj):
                best_edges_count = len(edges)
                best_edges = edges
                best_adj = adj

    return best_edges
