import itertools
import time
import random

def _build_grotzsch_edges():
    """Build the Grötzsch graph (11 vertices, triangle-free, not 3-colorable)."""
    edges = []
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, w in cycle:
        edges.append((u, w))
        edges.append((u, 5 + w))
        edges.append((w, 5 + u))
    for u in range(5):
        edges.append((5 + u, 10))
    return list(set(edges))

def _make_bitmask_adj(edges, n):
    """Create adjacency bitmask list from edge list."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_valid_triangle_free(edges, n):
    """Check if graph is triangle-free."""
    adj = _make_bitmask_adj(edges, n)
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                # Check if any common neighbor creates triangle
                common = adj[u] & adj[v]
                if common & (1 << u) or common & (1 << v):
                    # Remove u and v from common
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking."""
    color = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        # Try colors 0, 1, 2
        for c in range(3):
            # Check if color c is valid for vertex v
            valid = True
            for u in range(v):
                if adj[v] & (1 << u):
                    if color[u] == c:
                        valid = False
                        break
            if valid:
                color[v] = c
                if backtrack(v + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def _local_search(initial_edges, n, deadline):
    """Perform local search to maximize edges while maintaining triangle-free and not 3-colorable."""
    best_edges = list(initial_edges)
    best_count = len(best_edges)
    
    # Build adjacency
    adj = _make_bitmask_adj(best_edges, n)
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Set of current edges for O(1) lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in best_edges)
    
    # Current temperature
    temp = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Try to add a random edge
        candidate_edges = [e for e in all_edges if e not in edge_set]
        if not candidate_edges:
            break
            
        u, v = random.choice(candidate_edges)
        
        # Check if adding edge creates triangle
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        if common:
            # Would create triangle, skip
            continue
        
        # Add edge
        new_edges = best_edges + [(u, v)]
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
        
        # Check if still not 3-colorable
        if not _is_3_colorable(new_adj, n):
            # Good, keep it
            best_edges = new_edges
            adj = new_adj
            best_count = len(best_edges)
        else:
            # Bad, revert
            edge_set.remove((min(u, v), max(u, v)))
            
        # Try to remove an edge (to escape local optima)
        if best_edges and random.random() < 0.1:
            idx = random.randint(0, len(best_edges) - 1)
            u, v = best_edges[idx]
            new_edges = best_edges[:idx] + best_edges[idx+1:]
            new_adj = _make_bitmask_adj(new_edges, n)
            edge_set.remove((min(u, v), max(u, v)))
            
            if not _is_3_colorable(new_adj, n):
                # Keep the removal
                best_edges = new_edges
                adj = new_adj
                best_count = len(best_edges)
            else:
                # Revert
                best_edges.insert(idx, (u, v))
                adj = _make_bitmask_adj(best_edges, n)
                edge_set.add((min(u, v), max(u, v)))
        
        # Cool down
        temp *= cooling
    
    return best_edges, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    n = 16
    
    # Start with Grötzsch graph (11 vertices)
    grotzsch_edges = _build_grotzsch_edges()
    
    # Initialize remaining 5 vertices with no edges
    initial_edges = list(grotzsch_edges)
    
    # Run local search
    best_edges, best_count = _local_search(initial_edges, n, deadline)
    
    # Final validation
    if not _is_valid_triangle_free(best_edges, n):
        # Fallback to valid Grötzsch graph
        best_edges = list(grotzsch_edges)
    
    return [list(e) for e in best_edges]
