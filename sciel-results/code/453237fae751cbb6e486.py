<<<<<<< SEARCH
def build_adjacency(m, voltages):
    """Build adjacency list for fast girth checking."""
    n_vertices = 8 * m
    adj = [[] for _ in range(n_vertices)]
    
    for a in range(4):
        for b in range(4):
            v = voltages[(a, b)]
            base_a = a
            base_b = 4 + b
            for i in range(m):
                u_idx = base_a * m + i
                j_idx = (i + v) % m
                v_idx = base_b * m + j_idx
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
    
    return adj

def update_adjacency(adj, m, a, b, new_v):
    """Update adjacency when voltage (a,b) changes to new_v."""
    base_a = a
    base_b = 4 + b
    
    # Remove old edges for this (a,b) pair
    for i in range(m):
        u_idx = base_a * m + i
        # Find and remove the edge to the corresponding vertex on side b
        # The old voltage is implicit; we need to know what it was
        # Actually, we should pass the old voltage or recompute
        # Let's just rebuild this pair's edges
        pass
    
    # Better approach: rebuild all edges for this (a,b) pair
    # First, clear all edges involving vertices on side a that connect to side b
    # This is complex; let's just rebuild the entire adjacency for simplicity
    # But that defeats the purpose. Instead, let's track which edges belong to which (a,b)
    
    # For now, let's use a simpler approach: just rebuild from voltages
    # This is O(16*m) which is acceptable for m up to 50
    pass
=======
def build_adjacency(m, voltages):
    """Build adjacency list for fast girth checking."""
    n_vertices = 8 * m
    adj = [[] for _ in range(n_vertices)]
    
    for a in range(4):
        for b in range(4):
            v = voltages[(a, b)]
            base_a = a
            base_b = 4 + b
            for i in range(m):
                u_idx = base_a * m + i
                j_idx = (i + v) % m
                v_idx = base_b * m + j_idx
                adj[u_idx].append(v_idx)
                adj[v_idx].append(u_idx)
    
    return adj

def update_adjacency(adj, m, a, b, new_v):
    """Update adjacency when voltage (a,b) changes to new_v.
    
    For the K_{4,4} base graph, each vertex on side a connects to exactly
    one vertex on side b (the one determined by the voltage). So when we change
    voltage (a,b), we need to:
    1. Remove the old edge from each vertex on side a to its old partner on side b
    2. Add the new edge from each vertex on side a to its new partner on side b
    """
    base_a = a
    base_b = 4 + b
    
    # For each vertex on side a, remove old connection and add new
    for i in range(m):
        u_idx = base_a * m + i
        
        # The new partner on side b
        new_partner_idx = base_b * m + ((i + new_v) % m)
        
        # We need to know the old partner to remove it
        # But we don't have it stored. Let's just rebuild this pair.
        pass
    
    # Simpler: just rebuild all edges for this (a,b) pair
    # Clear all edges between side a and side b
    for i in range(m):
        u_idx = base_a * m + i
        # Remove all neighbors on side b
        adj[u_idx] = [v for v in adj[u_idx] if not (base_b * m <= v < (base_b + 1) * m)]
    
    for j in range(m):
        v_idx = base_b * m + j
        # Remove all neighbors on side a
        adj[v_idx] = [u for u in adj[v_idx] if not (base_a * m <= u < (base_a + 1) * m)]
    
    # Add new edges
    for i in range(m):
        u_idx = base_a * m + i
        new_partner_idx = base_b * m + ((i + new_v) % m)
        adj[u_idx].append(new_partner_idx)
        adj[new_partner_idx].append(u_idx)
>>>>>>> REPLACE

Now let me implement the fast girth check using non-backtracking walks:

