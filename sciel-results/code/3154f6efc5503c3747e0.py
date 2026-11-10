"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_k4_free(edge_set, n=18):
    """Check if graph with given edge_set is K4-free."""
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in adj[u]:
            for w in adj[v]:
                if w > v and w in adj[u]:
                    # u,v,w form a triangle; check if any common neighbor
                    common = adj[u] & adj[v] & adj[w]
                    if common:
                        return False
    return True

def _max_triangle_free_subgraph_size(edge_set, n=18):
    """Find the size of the largest triangle-free induced subgraph."""
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build triangle set: list of frozensets of 3 vertices
    triangles = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                for w in common:
                    if w > v:
                        triangles.append(frozenset([u, v, w]))
    
    # For each subset of vertices, check if it's triangle-free
    # We want the max size of a vertex set with no triangle fully contained
    # Use branch and bound: try to find a large triangle-free set
    
    best_size = 0
    
    # Greedy: start with all vertices, remove vertices to break all triangles
    # This is equivalent to finding the minimum vertex cover of the 3-uniform hypergraph of triangles
    # Then max triangle-free = n - min vertex cover of triangles
    
    # Use a greedy + local search approach
    # Greedy: repeatedly pick the vertex in the most triangles, remove it
    def greedy_min_vertex_cover(triangles, n):
        if not triangles:
            return 0
        tri_list = list(triangles)
        removed = set()
        while tri_list:
            # Count how many triangles each vertex appears in (among remaining)
            counts = {}
            for tri in tri_list:
                for v in tri:
                    if v not in removed:
                        counts[v] = counts.get(v, 0) + 1
            if not counts:
                break
            # Pick vertex with max count
            max_v = max(counts, key=counts.get)
            removed.add(max_v)
            tri_list = [t for t in tri_list if max_v not in t]
        return len(removed)
    
    # Try multiple random orderings for greedy
    best_vc = greedy_min_vertex_cover(triangles, n)
    
    # Local search: try to improve by moving vertices
    # Start from greedy solution
    removed = set()
    tri_list = list(triangles)
    # Reconstruct greedy solution
    while tri_list:
        counts = {}
        for tri in tri_list:
            for v in tri:
                if v not in removed:
                    counts[v] = counts.get(v, 0) + 1
        if not counts:
            break
        max_v = max(counts, key=counts.get)
        removed.add(max_v)
        tri_list = [t for t in tri_list if max_v not in t]
    
    best_size = n - len(removed)
    
    # Try local search: for each vertex in removed, try removing it and see if we can add another
    # Or: for each vertex NOT in removed, try adding it to removed and removing another
    for _ in range(200):
        improved = False
        # Try to remove a vertex from the cover (make it available)
        for v in list(removed):
            new_removed = removed - {v}
            # Check if all triangles are still covered
            all_covered = True
            for tri in triangles:
                if not (tri & new_removed):
                    all_covered = False
                    break
            if all_covered:
                removed = new_removed
                best_size = n - len(removed)
                improved = True
                break
        
        if not improved:
            # Try to swap: remove v from cover, add w to cover
            for v in list(removed):
                for w in range(n):
                    if w in removed:
                        continue
                    new_removed = removed - {v} | {w}
                    all_covered = True
                    for tri in triangles:
                        if not (tri & new_removed):
                            all_covered = False
                            break
                    if all_covered and len(new_removed) < len(removed):
                        removed = new_removed
                        best_size = n - len(removed)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_size

def _random_greedy_k4_free(n=18, seed=None):
    """Build a random K4-free graph using greedy edge addition."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    # Random edge order
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        # Check if adding (u,v) creates a K4
        # A K4 would be formed if there exist w, x such that
        # u-v, u-w, u-x, v-w, v-x, w-x all exist
        # After adding u-v, check common neighbors of u and v
        common = adj[u] & adj[v]
        # For each pair in common, check if they're connected
        can_add = True
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                w, x = common_list[i], common_list[j]
                if x in adj[w]:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
    
    return list(edge_set)

def _local_search(edges, n=18, max_iters=500, seed=None):
    """Local search to minimize max triangle-free subgraph size."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    edge_set = set(tuple(sorted(e)) for e in edges)
    current_score = _max_triangle_free_subgraph_size(edge_set, n)
    best_edges = list(edge_set)
    best_score = current_score
    
    for it in range(max_iters):
        if it % 50 == 0:
            if time.time() > 140:
                break
        
        # Try a random edge swap: remove one edge, add one
        if len(edge_set) > 0:
            rem_edge = rng.choice(list(edge_set))
        else:
            break
        
        # Find a candidate edge to add
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        cand = rng.choice(all_edges)
        
        new_set = edge_set - {rem_edge} | {cand}
        
        if not _is_k4_free(new_set, n):
            continue
        
        new_score = _max_triangle_free_subgraph_size(new_set, n)
        
        if new_score < current_score:
            current_score = new_score
            edge_set = new_set
            if current_score < best_score:
                best_score = current_score
                best_edges = list(edge_set)
    
    return best_edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n = 18
    
    # Start with the known good construction
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    base_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    base_score = _max_triangle_free_subgraph_size(base_set, n)
    
    best_edges = base_edges
    best_score = base_score
    
    # Try random constructions + local search
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Build random K4-free graph
        edges = _random_greedy_k4_free(n, seed=trial)
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Check it's valid
        if not _is_k4_free(edge_set, n):
            continue
        
        score = _max_triangle_free_subgraph_size(edge_set, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search
        ls_edges, ls_score = _local_search(edges, n, max_iters=100, seed=trial)
        
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
        
        # Also try local search on the base
        if trial == 0:
            ls_base, ls_base_score = _local_search(base_edges, n, max_iters=200, seed=999)
            if ls_base_score < best_score:
                best_score = ls_base_score
                best_edges = ls_base
    
    return [[u, v] for (u, v) in best_edges]
# EVOLVE-BLOCK-END
