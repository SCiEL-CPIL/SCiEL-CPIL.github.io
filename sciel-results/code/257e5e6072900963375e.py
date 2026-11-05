import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Build a C4-free graph on n vertices with min degree >= n - 39.
    Return {'n': n, 'edges': [[u, v], ...]}.
    Maximize n.
    """
    deadline = time.time() + 145
    
    best_result = None
    best_n = 0
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        required_min_deg = n - 39
        
        # Try multiple random restarts with local search
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            result = _local_search_c4_free(n, required_min_deg, deadline, attempt)
            if result is not None:
                if n > best_n:
                    best_n = n
                    best_result = result
                    return result
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a valid small graph
    return {'n': 1, 'edges': []}


def _check_c4_add(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if C4 would be created."""
    # Adding (u,v) creates C4 if exists x in N(u)\{v}, y in N(v)\{u} with x~y, x!=y, x,y not in {u,v}
    # i.e., exists x in N(u)\{v} such that N(x) ∩ N(v) contains y not in {u,x,v}
    
    neighbors_u = np.where(adj[u])[0]
    neighbors_v = np.where(adj[v])[0]
    
    for x in neighbors_u:
        if x == v:
            continue
        # Check if x has any neighbor y that is also a neighbor of v
        common = np.where(adj[x] & adj[v])[0]
        for y in common:
            if y != u and y != x and y != v:
                return True
    return False


def _local_search_c4_free(n, min_deg_req, deadline, seed_offset):
    """Construct a C4-free graph on n vertices with min degree >= min_deg_req using local search."""
    rng = random.Random(seed_offset * 7919 + 31)
    
    # Start with empty graph
    adj = np.zeros((n, n), dtype=bool)
    
    # Phase 1: Greedy build - add edges in random order, skipping those that create C4
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        if not _check_c4_add(adj, n, u, v):
            adj[u, v] = True
            adj[v, u] = True
    
    # Phase 2: Local search to improve minimum degree
    # We want to maximize min degree. Use simulated annealing style.
    degrees = adj.sum(axis=1)
    
    for _iter in range(5000):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        min_deg = int(degrees.min())
        
        if min_deg >= min_deg_req:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
        
        # Find the lowest-degree vertex
        v = int(np.argmin(degrees))
        
        # Try to add an edge from v to some other vertex
        candidates = [u for u in range(n) if u != v and not adj[u, v]]
        rng.shuffle(candidates)
        
        added = False
        for u in candidates:
            if time.time() > deadline:
                break
            if not _check_c4_add(adj, n, u, v):
                adj[u, v] = True
                adj[v, u] = True
                added = True
                break
        
        if not added:
            # Try edge swap: remove edge (a,b) and add edges that help
            # Or try to find a different vertex to connect to
            # Try removing a random edge and see if we can add a better one
            existing_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if existing_edges:
                # Try to swap: remove a random edge, then try to add edges to low-degree vertices
                a, b = rng.choice(existing_edges)
                # Check if removing (a,b) is safe (it always is for C4)
                adj[a, b] = False
                adj[b, a] = False
                
                # Now try to add edges to low-degree vertices
                degrees = adj.sum(axis=1)
                low_verts = sorted(range(n), key=lambda x: int(degrees[x]))
                
                for w in low_verts[:5]:
                    if time.time() > deadline:
                        break
                    candidates2 = [x for x in range(n) if x != w and not adj[w, x]]
                    rng.shuffle(candidates2)
                    for x in candidates2:
                        if not _check_c4_add(adj, n, w, x):
                            adj[w, x] = True
                            adj[x, w] = True
                            break
                
                # Check if we improved
                new_degrees = adj.sum(axis=1)
                if int(new_degrees.min()) < min_deg:
                    # Revert the swap
                    adj[a, b] = True
                    adj[b, a] = True
                    # Try to undo the additions we made - this is complex, so just break
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
