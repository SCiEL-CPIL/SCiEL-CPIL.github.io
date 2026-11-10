import random
from itertools import combinations

def construct():
    n = 20
    random.seed(42)
    
    # Use bitmask representation for adjacency
    # adj_mask[i] is a bitmask where bit j is set if edge (i,j) exists
    adj_mask = [0] * n
    
    # Start with a complete graph, then remove edges
    for i in range(n):
        for j in range(i+1, n):
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
    
    def common_neighbors(a, b, c):
        """Return bitmask of common neighbors of a, b, c"""
        return adj_mask[a] & adj_mask[b] & adj_mask[c]
    
    def find_violation():
        """Find a triple with >= 3 common neighbors. Returns (a,b,c, common_neighbors_mask) or None"""
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    cn = common_neighbors(a, b, c)
                    # Count bits
                    if bin(cn).count('1') >= 3:
                        return (a, b, c, cn)
        return None
    
    def remove_edge(u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    
    # Iteratively remove edges to fix violations
    # Use a strategy: for each violation, pick an edge to remove that minimizes future violations
    # Simple approach: remove a random edge from the common neighbors
    # Better: remove an edge that is part of many violations
    
    # First pass: simple removal
    max_iters = 5000
    for it in range(max_iters):
        viol = find_violation()
        if viol is None:
            break
        a, b, c, cn = viol
        # Get list of common neighbors
        neighbors = []
        for x in range(n):
            if (cn >> x) & 1:
                neighbors.append(x)
        # Pick a common neighbor x, and remove edge (x, a) or (x, b) or (x, c)
        # To maximize edge retention, remove the edge that is in fewest other violations
        # For now, pick randomly
        x = random.choice(neighbors)
        # Remove one of the edges (x,a), (x,b), or (x,c)
        # Try removing the one that breaks the most violations
        # Simple: just remove (x, a)
        remove_edge(x, a)
    
    # Second pass: try to add back edges that were removed
    # Try adding each non-edge and check if it creates a violation
    for u in range(n):
        for v in range(u+1, n):
            if not (adj_mask[u] >> v) & 1:
                # Try adding edge (u, v)
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                # Check if this creates any violation
                valid = True
                # Check all triples that include u or v
                # Actually, just check all triples
                for a in range(n):
                    for b in range(a+1, n):
                        for c in range(b+1, n):
                            cn = common_neighbors(a, b, c)
                            if bin(cn).count('1') >= 3:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    # Revert
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_mask[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
