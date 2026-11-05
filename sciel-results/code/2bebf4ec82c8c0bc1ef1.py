import numpy as np
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Try to build a C4-free graph on n vertices (n up to 47) with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145  # safety margin before 150s
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple strategies
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            result = _construct_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add_fast(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would be created.
    Uses bitset operations for speed."""
    # Adding (u,v) creates C4 if there exists x in N(u)\{v} and y in N(v)\{u} 
    # such that x~y and x!=y, x!=u, x!=v, y!=u, y!=v
    # Equivalently: exists x in N(u)\{v} such that (N(x) & N(v)) has an element y 
    # with y not in {u, x, v}
    
    neighbors_u = np.where(adj[u])[0]
    neighbors_v = np.where(adj[v])[0]
    
    for x in neighbors_u:
        if x == v:
            continue
        # Check if x has any neighbor y that is also a neighbor of v (y != u, x, v)
        common = np.where(adj[x] & adj[v])[0]
        for y in common:
            if y != u and y != x and y != v:
                return True
    return False


def _get_degree(adj, v):
    return int(adj[v].sum())


def _construct_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req."""
    rng = random.Random(seed_offset * 12345 + 67890)
    
    # Strategy: Greedy construction with intelligent edge ordering
    adj = np.zeros((n, n), dtype=bool)
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Multiple passes with different strategies
    for _pass in range(8):
        if time.time() > deadline:
            break
        
        # Sort edges by a dynamic score: prefer edges that help low-degree vertices
        # Score = sum of degrees of endpoints (lower = more urgent)
        remaining = [(u, v) for (u, v) in all_edges if not adj[u, v]]
        
        if not remaining:
            break
        
        # Use a weighted random selection biased toward low-degree vertices
        def edge_priority(u, v):
            du = int(adj[u].sum())
            dv = int(adj[v].sum())
            # Lower priority value = more urgent (we process these first)
            return (du + dv, rng.random())
        
        remaining.sort(key=lambda e: edge_priority(e[0], e[1]))
        
        improved = False
        for (u, v) in remaining:
            if time.time() > deadline:
                break
            if adj[u, v]:
                continue
            if not _check_c4_add_fast(adj, n, u, v):
                adj[u, v] = True
                adj[v, u] = True
                improved = True
        
        if not improved:
            break
    
    # Check if we meet the requirement
    degrees = adj.sum(axis=1)
    if int(degrees.min()) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    # Strategy 2: Local search - try to fix low-degree vertices
    # Find vertices with degree < min_deg_req and try to add edges to them
    for _iter in range(150):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        low_verts = [v for v in range(n) if degrees[v] < min_deg_req]
        
        if not low_verts:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        improved = False
        # Process lowest-degree vertices first
        low_verts.sort(key=lambda v: int(degrees[v]))
        
        for v in low_verts:
            if time.time() > deadline:
                break
            # Try adding edges from v to other vertices
            candidates = [u for u in range(n) if u != v and not adj[u, v]]
            # Sort candidates by degree (lower degree first)
            candidates.sort(key=lambda u: int(adj[u].sum()))
            for u in candidates:
                if not _check_c4_add_fast(adj, n, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    improved = True
                    if degrees[v] >= min_deg_req:
                        break
        
        if not improved:
            break
    
    # Final check
    degrees = adj.sum(axis=1)
    if int(degrees.min()) >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    return None


def _verify_c4_free(adj, n):
    """Verify graph is C4-free. Returns True if no C4 exists."""
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                continue
            # Check if there's a path u-x-y-v of length 3
            neighbors_u = set(np.where(adj[u])[0])
            neighbors_v = set(np.where(adj[v])[0])
            for x in neighbors_u:
                if x == v:
                    continue
                neighbors_x = set(np.where(adj[x])[0])
                common = neighbors_x & neighbors_v
                for y in common:
                    if y != u and y != v and y != x:
                        return False
    return True

# EVOLVE-BLOCK-END
