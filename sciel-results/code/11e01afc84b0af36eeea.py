"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_polarity_graph_pg2_5():
    """
    Build the polarity graph of PG(2,5).
    Points: 31 points in PG(2,5).
    A point is adjacent to all points on its polar line.
    This graph is C4-free and has independence number 8.
    """
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                p = [x, y, z]
                for i in range(3):
                    if p[i] != 0:
                        inv = pow(p[i], 3, 5)
                        p = [(c * inv) % 5 for c in p]
                        break
                if p not in points:
                    points.append(p)
    
    n = len(points)
    assert n == 31, f"Expected 31 points, got {n}"
    
    edges = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p, q = points[i], points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
    
    return edges, adj, n

def _to_bitmask(adj, n):
    """Convert adjacency list of sets to list of bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    return masks

def _max_independent_set_size_bitmask(adj_mask, n, time_limit=1.0):
    """
    Find max independent set size using bitmask branch and bound with greedy coloring bound.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is Clique in complement.
    """
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    # Precompute vertex ordering by degree in complement graph
    comp_deg = [comp_mask[i].bit_count() for i in range(n)]
    
    # Greedy coloring of the complement graph for upper bound - optimized
    def _color_bound(candidates):
        """Return an upper bound on clique size in the subgraph induced by candidates."""
        if not candidates:
            return 0
        
        cands_list = []
        temp = candidates
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cands_list.append(idx)
            temp ^= lsb
        
        # Order by degree in induced subgraph (descending)
        cands_list.sort(key=lambda v: -(comp_mask[v] & candidates).bit_count())
        
        # Greedy coloring: for each vertex, assign smallest available color
        color_of = [-1] * len(cands_list)
        color_count = 0
        # Map vertex -> index in cands_list for O(1) lookup
        v_to_idx = {}
        for idx, v in enumerate(cands_list):
            v_to_idx[v] = idx
        
        for idx, v in enumerate(cands_list):
            # Collect colors of already-colored neighbors
            used_colors = set()
            v_comp = comp_mask[v]
            for jdx in range(idx):
                w = cands_list[jdx]
                if (v_comp >> w) & 1:
                    used_colors.add(color_of[jdx])
            c = 0
            while c in used_colors:
                c += 1
            color_of[idx] = c
            if c + 1 > color_count:
                color_count = c + 1
        return color_count

    def _search(candidates, current_size):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Bound using greedy coloring
        color_bound = _color_bound(candidates)
        if current_size + color_bound <= best[0]:
            return
        
        # Pick a vertex with high degree in the candidate subgraph
        max_deg = -1
        v = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            idx = lsb.bit_length() - 1
            deg = (comp_mask[idx] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            temp_cands ^= lsb
            
        if v == -1:
            return

        # Include v
        new_cands = candidates & comp_mask[v]
        _search(new_cands, current_size + 1)
        
        # Exclude v
        new_cands2 = candidates & ~(1 << v)
        _search(new_cands2, current_size)

    _search((1 << n) - 1, 0)
    return best[0]

def _check_c4_free(adj, n):
    """Check if graph is C4-free."""
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i] & adj[j]
            if len(common) > 1:
                return False
    return True

def _find_valid_neighbor_set(adj, n, time_limit=5.0):
    """
    Systematically find a valid neighbor set for a new vertex.
    A valid neighbor set S must satisfy:
    1. No two vertices in S share a common neighbor in G (C4-free constraint)
    2. The non-neighbors U = V\S must have alpha(U) <= 9
    
    Returns the neighbor set S if found, else None.
    """
    start_time = time.time()
    
    # Precompute: for each vertex u, the set of vertices incompatible with u
    # (i.e., share at least one common neighbor)
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & adj[j]:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # Build compatibility graph: u ~ v if u and v can both be neighbors of new vertex
    # i.e., N(u) and N(v) are disjoint
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    # Find all maximal cliques in the compatibility graph using Bron-Kerbosch
    # These are candidate neighbor sets
    cliques = []
    
    def bronk_kerbosch(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            cliques.append(frozenset(R))
            return
        
        # Pivot selection
        if P or X:
            P_X = P | X
            pivot = max(P_X, key=lambda u: len(P & comp_adj[u]))
            candidates = [v for v in P if v not in comp_adj[pivot]]
        else:
            candidates = list(P)
        
        for v in candidates:
            bronk_kerbosch(R | {v}, P & comp_adj[v], X & comp_adj[v])
            P.discard(v)
            X.add(v)
    
    bronk_kerbosch(set(), set(range(n)), set())
    
    # Sort cliques by size (larger S means smaller U, easier to satisfy alpha constraint)
    cliques.sort(key=lambda s: -len(s))
    
    for S in cliques:
        if time.time() - start_time > time_limit:
            break
        
        U = set(range(n)) - S
        u_list = sorted(U)
        n_u = len(u_list)
        
        if n_u == 0:
            alpha_u = 0
        elif n_u <= 20:
            # Use bitmask solver for small U
            sub_adj_mask = [0] * n_u
            u_idx = {u: i for i, u in enumerate(u_list)}
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj_mask[i] |= (1 << j)
                        sub_adj_mask[j] |= (1 << i)
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=0.3)
        else:
            # For large U, use a quick greedy upper bound first
            # If greedy finds alpha > 9, skip; otherwise do exact
            # Greedy: repeatedly pick vertex with min degree in U
            remaining = set(u_list)
            greedy_size = 0
            temp_remaining = set(remaining)
            while temp_remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                chosen = None
                for v in temp_remaining:
                    deg = len(adj[v] & temp_remaining)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                if chosen is not None:
                    greedy_size += 1
                    temp_remaining.discard(chosen)
                    temp_remaining -= adj[chosen]
            
            if greedy_size > 9:
                continue  # Skip this S, likely alpha > 9
            
            # Do exact computation
            sub_adj_mask = [0] * n_u
            u_idx = {u: i for i, u in enumerate(u_list)}
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj_mask[i] |= (1 << j)
                        sub_adj_mask[j] |= (1 << i)
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
        
        if alpha_u <= 9:
            return set(S)
    
    return None

def _try_add_vertex(adj, n, edges, S):
    """
    Add a new vertex connected to all vertices in S.
    Returns (new_edges, new_adj, new_n) if valid, else None.
    """
    new_v = n
    new_n = n + 1
    new_adj = [set(s) for s in adj]
    new_adj.append(set(S))
    for u in S:
        new_adj[u].add(new_v)
    
    new_edges = list(edges)
    for u in S:
        new_edges.append([u, new_v])
    
    return new_edges, new_adj, new_n

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Build Base Graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
    # Verify base
    if not _check_c4_free(adj, n):
        return {'n': n, 'edges': edges}
    
    # Check alpha of base
    adj_mask = _to_bitmask(adj, n)
    alpha = _max_independent_set_size_bitmask(adj_mask, n, time_limit=2.0)
    
    if alpha >= 11:
        return {'n': n, 'edges': edges}

    best_n = n
    best_edges = edges
    best_adj = adj

    # 2. Extension Loop
    while time.time() < deadline - 20:
        found_extension = False
        
        # Systematic search for valid neighbor set
        S = _find_valid_neighbor_set(adj, n, time_limit=min(15.0, deadline - time.time() - 25))
        
        if S is not None:
            result = _try_add_vertex(adj, n, edges, S)
            if result:
                edges, adj, n = result
                best_n = n
                best_edges = edges
                best_adj = adj
                found_extension = True
        
        if not found_extension:
            break

    # Final Validation
    if best_n > 1:
        if not _check_c4_free(best_adj, best_n):
            return {'n': n, 'edges': edges}
        
        adj_mask_final = _to_bitmask(best_adj, best_n)
        alpha_final = _max_independent_set_size_bitmask(adj_mask_final, best_n, time_limit=5.0)
        
        if alpha_final >= 11:
            return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
