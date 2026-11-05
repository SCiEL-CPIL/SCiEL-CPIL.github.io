import time
import random
from itertools import combinations

def _has_c5(adj, u, v, n):
    """Check if adding edge (u,v) to graph with adjacency matrix adj creates a 5-cycle."""
    # A 5-cycle through u-v would be: u - v - a - b - c - u
    # where a,b,c are distinct vertices not u or v
    # Need: v~a, a~b, b~c, c~u, and {a,b,c} disjoint from {u,v}
    # Also need a != c (otherwise it's a 4-cycle with a chord or triangle)
    # Actually for a simple 5-cycle: u-v-a-b-c-u where all 5 vertices distinct
    
    # Get neighbors of u (excluding v) and neighbors of v (excluding u)
    # We need: exists a in N(v)\{u}, b in N(a)\{v}, c in N(b)\{a} such that c in N(u)\{v}
    # and all of u,v,a,b,c distinct
    
    # More efficient: iterate over pairs (a,c) where a~v, c~u, a!=c, a!=u, c!=v
    # Then check if there exists b such that b~a, b~c, b not in {u,v,a,c}
    
    neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
    neighbors_v = [x for x in range(n) if adj[v][x] and x != u]
    
    for a in neighbors_v:
        if a == u:
            continue
        for c in neighbors_u:
            if c == v or c == a:
                continue
            # Now check if there's a b adjacent to both a and c, b not in {u,v,a,c}
            # b must be in intersection of N(a) and N(c), excluding {u,v,a,c}
            na = adj[a]
            nc = adj[c]
            for b in range(n):
                if b in (u, v, a, c):
                    continue
                if na[b] and nc[b]:
                    return True
    return False

def _calc_alpha(adj, n):
    """Calculate independence number of graph using branch and bound."""
    # Use a simple backtracking approach with pruning
    # Convert to adjacency sets for faster lookup
    adj_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                adj_sets[i].add(j)
                adj_sets[j].add(i)
    
    best = [0]
    
    def dfs(candidates, size):
        if size + len(candidates) <= best[0]:
            return
        if not candidates:
            best[0] = max(best[0], size)
            return
        
        # Branch and bound: try each vertex in candidates
        cands = list(candidates)
        # Sort by degree (descending) for better pruning
        cands.sort(key=lambda x: len(adj_sets[x]))
        
        for i, v in enumerate(cands):
            # Include v
            new_cands = set(cands[i+1:])
            # Remove all neighbors of v from new_cands
            for nb in adj_sets[v]:
                new_cands.discard(nb)
            dfs(new_cands, size + 1)
            
            # Exclude v: continue with remaining candidates
            # But we need to be careful - let's just iterate
        # Actually, let me use a cleaner approach
        # Pick a vertex, branch on include/exclude
        if not cands:
            return
        v = cands[0]
        remaining = set(cands[1:])
        
        # Include v
        new_cands = set(remaining)
        for nb in adj_sets[v]:
            new_cands.discard(nb)
        dfs(new_cands, size + 1)
        
        # Exclude v
        dfs(remaining, size)
    
    dfs(set(range(n)), 0)
    return best[0]

def _try_greedy(n, max_time=100.0, seed=None):
    """Try to construct a C5-free graph on n vertices with alpha < 9 using greedy edge addition."""
    start_time = time.time()
    rng = random.Random(seed)
    
    # Initialize empty graph
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    
    # We want to add edges to reduce independence number
    # Start with a structure that has high independence number, then add edges
    # Actually, let's start from a known good structure and try to extend
    
    # Strategy: Start with 8 K4s (n=32) and try to add 3 more vertices with edges
    # Or: start from scratch with random edge ordering
    
    # Let's try a different approach: start with a Turan-like graph and add edges
    # to break C5 while maintaining low alpha
    
    # Better: use the 8 K4 construction as base, add 3 vertices, try to connect them
    # such that no C5 is formed and alpha stays < 9
    
    # First, create the 8 K4 base
    adj_base = [[False] * n for _ in range(n)]
    for block_idx in range(8):
        start = block_idx * 4
        for i in range(4):
            for j in range(i+1, 4):
                u = start + i
                v = start + j
                adj_base[u][v] = True
                adj_base[v][u] = True
                edge_set.add((u, v) if u < v else (v, u))
    
    # Now try to add edges involving vertices 32, 33, 34
    # These vertices initially have no edges, so they're isolated
    # We need to connect them to reduce alpha
    
    # The alpha of the base is 8 (one from each K4)
    # Adding 3 isolated vertices makes alpha = 8 + 3 = 11
    # We need to reduce this to < 9, so we need to connect the 3 new vertices
    # to the existing graph in a way that reduces alpha by at least 3
    
    # For each new vertex, if we connect it to at least one vertex in each K4,
    # it can't be in an independent set with the chosen vertex from that K4
    # But we also need to avoid creating C5s
    
    # Let's try: connect each new vertex to multiple vertices in the K4s
    # but carefully to avoid C5
    
    # Actually, let's use a more systematic approach:
    # Try random subsets of edges for the 3 new vertices
    
    base_edges = list(edge_set)
    
    # Generate candidate edges for the 3 new vertices
    new_verts = [32, 33, 34]
    candidate_edges = []
    for nv in new_verts:
        for old_v in range(32):
            candidate_edges.append((min(nv, old_v), max(nv, old_v)))
    
    # Also edges between new vertices
    for i in range(len(new_verts)):
        for j in range(i+1, len(new_verts)):
            candidate_edges.append((min(new_verts[i], new_verts[j]), max(new_verts[i], new_verts[j])))
    
    # Try random subsets of candidate edges
    # We want to find a subset that:
    # 1. Doesn't create C5
    # 2. Results in alpha < 9
    
    # Use greedy: randomly add edges, check constraints
    
    best_result = None
    
    for attempt in range(5000):
        if time.time() - start_time > max_time - 5:
            break
        
        # Start with base
        adj = [[False] * n for _ in range(n)]
        edges = []
        for (u, v) in edge_set:
            adj[u][v] = True
            adj[v][u] = True
            edges.append([u, v])
        
        # Randomly try to add edges involving new vertices
        shuffled = candidate_edges[:]
        rng.shuffle(shuffled)
        
        success = True
        for (u, v) in shuffled:
            if time.time() - start_time > max_time - 5:
                success = False
                break
            
            # Check if adding this edge creates a C5
            if _has_c5(adj, u, v, n):
                continue
            
            # Tentatively add edge
            adj[u][v] = True
            adj[v][u] = True
            
            # Check alpha (expensive, so do it less frequently)
            # For now, just add it
            edges.append([u, v])
        
        if not success:
            continue
        
        # Check if alpha < 9
        alpha = _calc_alpha(adj, n)
        if alpha < 9:
            # Verify no C5 exists in the entire graph
            # The _has_c5 check was done incrementally, but let's verify
            # Actually, since we checked each edge addition, it should be C5-free
            # But let's be safe and do a full check
            
            # Full C5 check: for each edge (u,v), check if there's a 3-path from v to u
            # avoiding u and v internally
            has_c5 = False
            for (u, v) in edges:
                if _has_c5(adj, u, v, n):
                    has_c5 = True
                    break
            
            if not has_c5:
                best_result = {'n': n, 'edges': edges}
                break
    
    return best_result

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Tries to achieve n=35 via random greedy search, falls back to n=32.
    """
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Try to construct n=35 graph
    try:
        result = _try_greedy(35, max_time=140.0, seed=42)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Fallback: 8 disjoint K4s, n=32
    n = 32
    edges = []
    for block_idx in range(8):
        start = block_idx * 4
        verts = [start, start + 1, start + 2, start + 3]
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                edges.append([verts[i], verts[j]])
    
    return {'n': n, 'edges': edges}
