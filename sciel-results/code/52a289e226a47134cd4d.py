import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    
    # Try n=47 first
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline:
            break
        
        min_deg_req = target_n - 39
        
        # Strategy 1: Polarity graph construction (provably C4-free)
        # For prime power q, polarity graph has q^2+q+1 vertices, degree q
        # We can use subgraphs or augment
        result = _polarity_based_construct(target_n, min_deg_req, deadline)
        if result is not None:
            return result
        
        # Strategy 2: Greedy construction with C4 checking
        for attempt in range(30):
            if time.time() > deadline:
                break
            result = _greedy_construct(target_n, min_deg_req, deadline, attempt)
            if result is not None:
                return result
        
        # Strategy 3: Local search from random dense graph
        for attempt in range(10):
            if time.time() > deadline:
                break
            result = _local_search_construct(target_n, min_deg_req, deadline, attempt)
            if result is not None:
                return result
    
    # Fallback
    return {'n': 39, 'edges': []}


def _check_add_edge_c4(adj, u, v, n):
    """Check if adding edge (u,v) creates a C4 in a graph on n vertices."""
    # A C4 u-x-y-v-u is created if there exists x in N(u)\{v} and y in N(x)\{u,v} with y in N(v)
    # Equivalent: count paths of length 3 from u to v not using edge (u,v)
    # For each x neighbor of u (x != v), for each y neighbor of x (y != u, y != v), check if y neighbor of v
    nu = adj[u]
    nv = adj[v]
    for x in range(n):
        if x == v:
            continue
        if not nu[x]:
            continue
        nx = adj[x]
        for y in range(n):
            if y == u or y == v:
                continue
            if nx[y] and nv[y]:
                return True
    return False


def _count_c4_violations(adj, n):
    """Count number of C4s in the graph. A C4 is detected if any pair has >= 2 common neighbors."""
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            common = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common += 1
                    if common >= 2:
                        count += 1
                        break
    return count


def _polarity_based_construct(n, min_deg_req, deadline):
    """
    Construct a C4-free graph using finite field polarity graph ideas.
    For prime p, the graph on Z_p^2 with edges based on quadratic form is C4-free.
    """
    # Use the standard polarity graph: vertices are points of PG(2,q)
    # For simplicity, use the "norm graph" construction
    # Norm graph: vertices (x,y) in F_q^2, edge if x*y' - y*x' = 1 (or similar)
    
    # Actually, let's use a simpler approach: 
    # The graph where vertices are elements of F_q and edges based on quadratic residues
    # This gives a C4-free graph when q is prime
    
    # For our purposes, let's try a direct construction:
    # Vertices 0..n-1. We want min degree >= min_deg_req.
    # Use a greedy approach seeded with a structured initial graph.
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Initial structure: use a bipartite-like base
    # Split into two groups, connect based on a C4-free pattern
    mid = n // 2
    
    # Try quadratic residue construction for the base
    # For a prime p, the graph on Z_p where i~j iff (i-j) is a quadratic residue is C4-free
    # when p ≡ 3 mod 4 (Paley graph complement properties)
    
    # Let's try: for each vertex, connect to vertices at "quadratic residue" distances
    # This is complex, so let's use a simpler structured seed
    
    # Seed with a random C4-free subgraph, then greedily add edges
    rng = random.Random(42)
    
    # Start with a sparse random graph
    p = min_deg_req / n * 1.2
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = True
                adj[v][u] = True
    
    # Remove any C4s
    for _ in range(100):
        if time.time() > deadline:
            break
        c4_found = False
        for u in range(n):
            if c4_found:
                break
            for v in range(u + 1, n):
                common = []
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        common.append(w)
                if len(common) >= 2:
                    # Remove edge (u, common[0]) to break C4
                    w = common[0]
                    adj[u][w] = False
                    adj[w][u] = False
                    c4_found = True
                    break
        if not c4_found:
            break
    
    # Now greedily add edges to increase minimum degree
    for _ in range(500):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        if degrees.min() >= min_deg_req:
            break
        
        # Find vertices with lowest degree
        low_vertices = np.argsort(degrees)[:min_deg_req + 2]
        
        added = False
        for u in low_vertices:
            if time.time() > deadline:
                break
            if degrees[u] >= min_deg_req:
                continue
            
            # Try to add edges from u to non-neighbors
            candidates = []
            for v in range(n):
                if v == u or adj[u][v]:
                    continue
                candidates.append(v)
            
            rng.shuffle(candidates)
            for v in candidates:
                if _check_add_edge_c4(adj, u, v, n):
                    continue
                # Check this doesn't create a C4
                adj[u][v] = True
                adj[v][u] = True
                degrees[u] += 1
                degrees[v] += 1
                added = True
                break
            if degrees[u] >= min_deg_req:
                continue
        
        if not added:
            break
    
    # Final check
    degrees = adj.sum(axis=1)
    if degrees.min() >= min_deg_req:
        # Verify no C4
        c4_count = _count_c4_violations(adj, n)
        if c4_count == 0:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None


def _greedy_construct(n, min_deg_req, deadline, seed):
    """Greedy edge addition with C4 checking."""
    rng = random.Random(seed)
    adj = np.zeros((n, n), dtype=bool)
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Priority: add edges that help low-degree vertices
    while time.time() < deadline:
        degrees = adj.sum(axis=1)
        if degrees.min() >= min_deg_req:
            break
        
        # Sort edges by sum of endpoint degrees (add to low-degree vertices first)
        # But also consider C4 constraint
        candidates = []
        for (u, v) in all_edges:
            if adj[u][v]:
                continue
            # Only consider if it helps a vertex below required degree
            if degrees[u] >= min_deg_req and degrees[v] >= min_deg_req:
                continue
            candidates.append((u, v, degrees[u] + degrees[v]))
        
        if not candidates:
            break
        
        # Sort by priority (lower degree sum first)
        candidates.sort(key=lambda x: x[2])
        
        added = False
        for (u, v, _) in candidates:
            if time.time() > deadline:
                break
            if adj[u][v]:
                continue
            if _check_add_edge_c4(adj, u, v, n):
                continue
            adj[u][v] = True
            adj[v][u] = True
            added = True
            break
        
        if not added:
            break
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= min_deg_req:
        c4_count = _count_c4_violations(adj, n)
        if c4_count == 0:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None


def _local_search_construct(n, min_deg_req, deadline, seed):
    """Local search: start from a valid sparse graph, try edge swaps."""
    rng = random.Random(seed)
    
    # Start with a valid C4-free graph (empty is valid)
    adj = np.zeros((n, n), dtype=bool)
    
    # First, build a valid base using greedy
    base_result = _greedy_construct(n, min_deg_req, deadline, seed)
    if base_result is not None:
        return base_result
    
    # If greedy failed, try a different approach:
    # Start with a complete graph, remove edges to eliminate C4s while maintaining degree
    adj = np.ones((n, n), dtype=bool)
    np.fill_diagonal(adj, False)
    
    # Remove edges to eliminate all C4s
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Find a C4
        c4_found = False
        for u in range(n):
            if c4_found:
                break
            for v in range(u + 1, n):
                if not adj[u][v]:
                    continue
                common = []
                for w in range(n):
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        common.append(w)
                if len(common) >= 2:
                    # Remove one edge to break this C4
                    # Remove (u, common[0])
                    adj[u][common[0]] = False
                    adj[common[0]][u] = False
                    c4_found = True
                    break
        if not c4_found:
            break
    
    # Check if min degree is satisfied
    degrees = adj.sum(axis=1)
    if degrees.min() >= min_deg_req:
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return {'n': n, 'edges': edges}
    
    # Try to add edges back where possible
    for _ in range(100):
        if time.time() > deadline:
            break
        
        degrees = adj.sum(axis=1)
        if degrees.min() >= min_deg_req:
            break
        
        # Try adding edges to low-degree vertices
        low_v = np.argmin(degrees)
        if degrees[low_v] >= min_deg_req:
            break
        
        added = False
        for v in range(n):
            if v == low_v or adj[low_v][v]:
                continue
            if _check_add_edge_c4(adj, low_v, v, n):
                continue
            adj[low_v][v] = True
            adj[v][low_v] = True
            degrees[low_v] += 1
            degrees[v] += 1
            added = True
            break
        
        if not added:
            break
    
    degrees = adj.sum(axis=1)
    if degrees.min() >= min_deg_req:
        c4_count = _count_c4_violations(adj, n)
        if c4_count == 0:
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u][v]:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    return None
