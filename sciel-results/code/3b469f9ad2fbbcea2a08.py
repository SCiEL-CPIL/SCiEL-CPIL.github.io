"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_3_colorable(adj_masks, n):
    """Check if graph with adjacency bitmasks is 3-colorable using backtracking."""
    # Try to color vertices 0..n-1 with colors 0,1,2
    # Use bitmask: adj_masks[i] is a bitmask of neighbors of i
    color = [-1] * n
    
    def bt(v):
        if v == n:
            return True
        # Try colors 0, 1, 2
        for c in range(3):
            # Check if any neighbor already has color c
            conflict = False
            nb = adj_masks[v]
            # For each neighbor, check its color
            # We can precompute: for color c, which vertices have color c
            # But simpler: just iterate neighbors
            # Actually with bitmask, we need to check neighbors' colors
            # Let's use a different approach: track color class bitmasks
            # For now, just check directly
            for u in range(n):
                if (nb >> u) & 1:
                    if color[u] == c:
                        conflict = True
                        break
            if not conflict:
                color[v] = c
                if bt(v + 1):
                    return True
                color[v] = -1
        return False
    
    return bt(0)


def _is_3_colorable_fast(adj_masks, n):
    """Faster 3-colorability check using color-class bitmasks."""
    # color_class[c] = bitmask of vertices colored c
    color = [-1] * n
    cc = [0, 0, 0]  # color class bitmasks
    
    def bt(v):
        if v == n:
            return True
        nb = adj_masks[v]
        for c in range(3):
            # If any neighbor is in color class c, conflict
            if nb & cc[c]:
                continue
            color[v] = c
            cc[c] |= (1 << v)
            if bt(v + 1):
                return True
            cc[c] &= ~(1 << v)
            color[v] = -1
        return False
    
    return bt(0)


def _triangle_free(adj_masks, n):
    """Check if graph is triangle-free using bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if (adj_masks[u] >> v) & 1:
                # Check if u and v share a common neighbor
                common = adj_masks[u] & adj_masks[v]
                # Remove u and v themselves
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common:
                    return False
    return True


def _grötzsch_edges():
    """Return edges of the Grötzsch graph (Mycielski of C5) on vertices 0..10."""
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges


def _edges_to_adj(edges, n):
    """Convert edge list to adjacency bitmasks."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj


def _adj_to_edges(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 16
    
    # Start with Grötzsch graph on vertices 0..10, vertices 11..15 isolated
    base_edges = _grötzsch_edges()
    adj = _edges_to_adj(base_edges, n)
    
    # Verify initial state
    assert _triangle_free(adj, n), "Initial graph not triangle-free"
    assert not _is_3_colorable_fast(adj, n), "Initial graph is 3-colorable"
    
    best_edges = _adj_to_edges(adj, n)
    best_count = len(best_edges)
    
    # Try to add edges greedily: for each non-edge, try adding it
    # if it maintains triangle-free and non-3-colorable
    def try_add_edge(adj, u, v, n):
        """Try to add edge (u,v). Return new adj if valid, else None."""
        # Check triangle-free: u and v must not share a common neighbor
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        if common:
            return None  # Would create a triangle
        
        # Check non-3-colorable: new graph must not be 3-colorable
        new_adj = adj[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        if _is_3_colorable_fast(new_adj, n):
            return None  # Would make it 3-colorable
        
        return new_adj
    
    # Greedy edge addition
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue  # Edge already exists
                new_adj = try_add_edge(adj, u, v, n)
                if new_adj is not None:
                    adj = new_adj
                    changed = True
                    best_edges = _adj_to_edges(adj, n)
                    best_count = len(best_edges)
    
    # Local search: try edge swaps to improve
    # Random restarts with perturbation
    rng = random.Random(42)
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Start from current best
        current_adj = adj[:]
        
        # Try random local moves
        for _ in range(500):
            if time.time() > deadline:
                break
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try adding a random non-edge
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (current_adj[u] >> v) & 1:
                    continue
                new_adj = try_add_edge(current_adj, u, v, n)
                if new_adj is not None:
                    current_adj = new_adj
            else:
                # Try removing an edge and adding two others (swap)
                # Find a random edge
                edges = _adj_to_edges(current_adj, n)
                if not edges:
                    continue
                ru, rv = rng.choice(edges)
                
                # Remove edge
                new_adj = current_adj[:]
                new_adj[ru] &= ~(1 << rv)
                new_adj[rv] &= ~(1 << ru)
                
                # Try to add two random non-edges
                added = 0
                for _ in range(10):
                    u = rng.randint(0, n - 1)
                    v = rng.randint(0, n - 1)
                    if u == v or u > v:
                        continue
                    if (new_adj[u] >> v) & 1:
                        continue
                    # Check triangle-free
                    common = new_adj[u] & new_adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        continue
                    # Check non-3-colorable
                    test_adj = new_adj[:]
                    test_adj[u] |= (1 << v)
                    test_adj[v] |= (1 << u)
                    if not _is_3_colorable_fast(test_adj, n):
                        new_adj = test_adj
                        added += 1
                        break
                
                if added > 0:
                    current_adj = new_adj
        
        # Check if we improved
        cur_edges = _adj_to_edges(current_adj, n)
        if len(cur_edges) > best_count:
            best_count = len(cur_edges)
            best_edges = cur_edges
            adj = current_adj[:]
    
    return best_edges

# EVOLVE-BLOCK-END
