"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # Start with 8 disjoint K5s (80 edges)
    edges = []
    for k in range(8):
        edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
    
    # Try to add edges between the K5 components
    # First, let's try to add all possible cross-edges and remove those that create C6
    
    # Build adjacency structure
    n = 40
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Helper to check if adding edge (u,v) creates a C6
    def creates_c6(u, v):
        """Check if adding edge (u,v) creates a 6-cycle."""
        # A 6-cycle containing (u,v) means there's a path of length 5 from u to v
        # that doesn't use edge (u,v) directly
        # Equivalently, there exist vertices a,b,c,d such that:
        # u-a-b-c-d-v is a path (all edges exist, all distinct vertices)
        # Or u-a-b-c-d-v where u~a, a~b, b~c, c~d, d~v
        
        # More efficiently: for each common neighbor approach
        # Path u - x1 - x2 - x3 - x4 - v
        # We need to find if there's a path of length 5 from u to v in G (without using u-v edge)
        
        # BFS/DFS from u up to depth 5, checking if we can reach v
        # But we need simple paths (no repeated vertices)
        
        # Alternative: enumerate pairs
        # For each pair (x1, x4) where u~x1 and x4~v:
        #   check if there's a path of length 3 from x1 to x4 (x1-x2-x3-x4)
        #   with x1, x2, x3, x4 all distinct and not equal to u, v
        
        neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
        neighbors_v = [x for x in range(n) if adj[v][x] and x != u]
        
        for x1 in neighbors_u:
            for x4 in neighbors_v:
                if x1 == x4:
                    continue
                # Check if there's a path x1-x2-x3-x4 of length 3
                # with all vertices distinct from u, v, x1, x4
                for x2 in range(n):
                    if x2 in (u, v, x1, x4):
                        continue
                    if not adj[x1][x2]:
                        continue
                    for x3 in range(n):
                        if x3 in (u, v, x1, x2, x4):
                            continue
                        if not adj[x2][x3]:
                            continue
                        if adj[x3][x4]:
                            return True
        return False
    
    # Try adding all possible cross-edges
    # Group vertices by their K5 component
    component = [v // 5 for v in range(n)]
    
    # Sort candidate edges by some heuristic - try edges between components
    # that might be "safe"
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                candidates.append((u, v))
    
    # Shuffle for variety
    random.shuffle(candidates)
    
    # Greedily add edges that don't create C6
    best_edges = list(edges)
    best_count = len(edges)
    
    for u, v in candidates:
        if time.time() > deadline - 5.0:
            break
        if not creates_c6(u, v):
            adj[u][v] = True
            adj[v][u] = True
            best_edges.append([u, v])
            best_count += 1
    
    # Local search: try to swap edges to increase count
    # This is complex, so let's just return what we have
    
    return best_edges
# EVOLVE-BLOCK-END
