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
    Find max independent set size using bitmask branch and bound with bit-level greedy coloring.
    adj_mask[i] has bit j set if i and j are adjacent.
    Independent set in G is Clique in complement.
    """
    start_time = time.time()
    best = [0]
    
    all_mask = (1 << n) - 1
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = all_mask ^ adj_mask[i] ^ (1 << i)

    # Precompute vertex ordering by degree in complement graph for tie-breaking
    comp_deg = [comp_mask[i].bit_count() for i in range(n)]
    
    def _color_bound(candidates):
        """Fast bit-level greedy coloring for upper bound on clique size."""
        if not candidates:
            return 0
        color_classes = []  # list of masks, one per color
        remaining = candidates
        while remaining:
            # Pick vertex with max degree in remaining subgraph
            max_deg = -1
            v = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                deg = (comp_mask[idx] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp ^= lsb
            if v == -1:
                break
            # Assign smallest available color
            placed = False
            for ci, cm in enumerate(color_classes):
                if not (comp_mask[v] & cm):
                    color_classes[ci] = cm | (1 << v)
                    placed = True
                    break
            if not placed:
                color_classes.append(1 << v)
            remaining &= ~(1 << v)
        return len(color_classes)

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

def _get_compatible_cliques(adj, n, time_limit=5.0):
    """
    Find cliques in the compatibility graph H.
    H has vertices 0..n-1.
    u ~_H v if N_G(u) and N_G(v) are disjoint.
    We want cliques in H, which are sets S of vertices in G that can be neighbors of a new vertex.
    """
    start_time = time.time()
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & adj[j]):
                comp_adj[i].add(j)
                comp_adj[j].add(i)
    
    cliques = []
    
    def bronk(R, P, X):
        if time.time() - start_time > time_limit:
            return
        if not P and not X:
            cliques.append(set(R))
            return
        
        u = -1
        max_deg = -1
        P_list = list(P)
        for v in P_list:
            deg = len(P & comp_adj[v])
            if deg > max_deg:
                max_deg = deg
                u = v
        
        if u == -1:
            if not X:
                cliques.append(set(R))
            return

        P_copy = set(P)
        for v in list(P_copy):
            if v in comp_adj[u]:
                continue
            
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bronk(new_R, new_P, new_X)
            
            P.remove(v)
            X.add(v)

    bronk(set(), set(range(n)), set())
    return cliques

def _random_extension_attempt(adj, n, edges, time_limit=5.0):
    """
    Try to add a new vertex using random sampling of neighbor sets.
    Returns (new_edges, new_adj, new_n) if successful, else None.
    """
    start_time = time.time()
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & adj[j]:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    for trial in range(300):
        if time.time() - start_time > time_limit:
            break
        
        size = random.randint(3, max(3, n - 3))
        if size > n:
            size = n
        
        neighbors = set(random.sample(range(n), size))
        
        valid = True
        for u in neighbors:
            if neighbors & incompatible[u]:
                valid = False
                break
        
        if not valid:
            continue
        
        U = set(range(n)) - neighbors
        u_list = sorted(U)
        n_u = len(u_list)
        
        if n_u == 0:
            alpha_u = 0
        else:
            sub_adj_mask = [0] * n_u
            for i in range(n_u):
                for j in range(i+1, n_u):
                    u, v = u_list[i], u_list[j]
                    if v in adj[u]:
                        sub_adj_mask[i] |= (1 << j)
                        sub_adj_mask[j] |= (1 << i)
            alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=0.3)
        
        if alpha_u <= 9:
            new_v = n
            new_n = n + 1
            new_adj = [set(s) for s in adj]
            new_adj.append(set(neighbors))
            for u in neighbors:
                new_adj[u].add(new_v)
            
            new_edges = list(edges)
            for u in neighbors:
                new_edges.append([u, new_v])
            
            return new_edges, new_adj, new_n
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # 1. Build Base Graph
    edges, adj, n = _build_polarity_graph_pg2_5()
    
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
    rounds = 0
    while time.time() < deadline - 15:
        rounds += 1
        found_extension = False
        
        # Try systematic clique-based extension
        remaining = deadline - time.time() - 15
        if remaining > 5:
            cliques = _get_compatible_cliques(adj, n, time_limit=min(remaining - 10, 20.0))
            
            if cliques:
                cliques.sort(key=lambda s: -len(s))
                
                for S in cliques:
                    if time.time() > deadline - 15:
                        break
                    
                    U = set(range(n)) - S
                    u_list = sorted(U)
                    n_u = len(u_list)
                    
                    if n_u == 0:
                        alpha_u = 0
                    else:
                        sub_adj = [set() for _ in range(n_u)]
                        for i in range(n_u):
                            for j in range(i+1, n_u):
                                u, v = u_list[i], u_list[j]
                                if v in adj[u]:
                                    sub_adj[i].add(v)
                                    sub_adj[j].add(u)
                        sub_adj_mask = _to_bitmask(sub_adj, n_u)
                        alpha_u = _max_independent_set_size_bitmask(sub_adj_mask, n_u, time_limit=1.0)
                    
                    if alpha_u <= 9:
                        new_v = n
                        new_n = n + 1
                        new_adj = [set(s) for s in adj]
                        new_adj.append(set(S))
                        for u in S:
                            new_adj[u].add(new_v)
                        
                        new_edges = list(edges)
                        for u in S:
                            new_edges.append([u, new_v])
                        
                        edges = new_edges
                        adj = new_adj
                        n = new_n
                        best_n = n
                        best_edges = new_edges
                        best_adj = new_adj
                        
                        found_extension = True
                        break
        
        # If systematic failed, try random extension
        if not found_extension and time.time() < deadline - 15:
            result = _random_extension_attempt(adj, n, edges, time_limit=min(deadline - time.time() - 15, 10.0))
            if result:
                new_edges, new_adj, new_n = result
                edges = new_edges
                adj = new_adj
                n = new_n
                best_n = n
                best_edges = new_edges
                best_adj = new_adj
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
