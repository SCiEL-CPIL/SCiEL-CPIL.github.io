import random
import time

def _popcount(x):
    return bin(x).count('1')

def _is_k4_free_bitmask(adj):
    """Check if graph with adjacency bitmasks is K4-free."""
    n = 18
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                common = adj[u] & adj[v] & ~((1 << u) | (1 << v))
                m = common
                while m:
                    lsb = m & -m
                    a = lsb.bit_length() - 1
                    m ^= lsb
                    if adj[a] & common:
                        return False
    return True

def _max_triangle_free_bitmask(adj):
    """
    Find the maximum number of vertices inducing a triangle-free subgraph.
    Uses branch and bound with bitmasks.
    """
    n = 18
    best = [0]
    
    def branch(candidates, current_set, current_size):
        # Pruning
        if current_size + _popcount(candidates) <= best[0]:
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Heuristic: pick a vertex with high degree in the induced subgraph of candidates
        v_idx = -1
        max_deg = -1
        m = candidates
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = _popcount(adj[v] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = v
        
        v = v_idx
        v_bit = 1 << v
        rest_candidates = candidates & ~v_bit
        
        # Option 1: Exclude v
        branch(rest_candidates, current_set, current_size)
        
        # Option 2: Include v
        N_v = adj[v] & current_set
        if N_v:
            m2 = N_v
            conflict = False
            while m2:
                lsb2 = m2 & -m2
                u = lsb2.bit_length() - 1
                m2 ^= lsb2
                if adj[u] & N_v:
                    conflict = True
                    break
            if conflict:
                return
        
        new_set = current_set | v_bit
        new_size = current_size + 1
        
        forbidden = 0
        m3 = N_v
        while m3:
            lsb3 = m3 & -m3
            u = lsb3.bit_length() - 1
            m3 ^= lsb3
            forbidden |= (adj[v] & adj[u])
        
        new_candidates = rest_candidates & ~forbidden
        
        branch(new_candidates, new_set, new_size)
        
    branch((1 << n) - 1, 0, 0)
    return best[0]

def _line_graph_of_bipartite(bip_adj):
    """
    Construct the line graph of a bipartite graph.
    bip_adj: list of bitmasks for the bipartite graph (left vertices 0..8, right vertices 0..8)
    The line graph has 18 vertices (edges of the bipartite graph).
    Two vertices in the line graph are adjacent iff the corresponding edges in the bipartite graph share an endpoint.
    """
    n_left = 9
    n_right = 9
    # Map each edge (i, j) to a vertex index in the line graph
    # vertex index = i * 9 + j
    line_adj = [0] * 18
    
    for i in range(n_left):
        for j in range(n_right):
            if bip_adj[i] & (1 << j):
                u = i * 9 + j
                # Edges sharing left endpoint i
                for j2 in range(n_right):
                    if j2 != j and (bip_adj[i] & (1 << j2)):
                        v = i * 9 + j2
                        line_adj[u] |= (1 << v)
                        line_adj[v] |= (1 << u)
                # Edges sharing right endpoint j
                for i2 in range(n_left):
                    if i2 != i and (bip_adj[i2] & (1 << j)):
                        v = i2 * 9 + j
                        line_adj[u] |= (1 << v)
                        line_adj[v] |= (1 << u)
    
    return line_adj

def _is_line_graph_k4_free(line_adj):
    """
    Line graphs of bipartite graphs are K4-free iff the bipartite graph has no K_{2,2} (i.e., no 4-cycle).
    Actually, a line graph contains a K4 iff the original graph contains a K4 or a K_{1,4} (star with 4 leaves) or a triangle with a pendant edge... 
    Wait, for bipartite graphs: The line graph of a bipartite graph is K4-free iff the bipartite graph has no vertex of degree >= 4 AND no K_{2,2}.
    Actually, a K4 in the line graph corresponds to 4 edges in the original graph that are pairwise adjacent. In a bipartite graph, 4 pairwise adjacent edges must either:
    1. All share a common vertex (a star K_{1,4}), or
    2. Form a K_{2,2} (a 4-cycle).
    So we need to check: no vertex has degree >= 4, and no K_{2,2}.
    """
    n_left = 9
    n_right = 9
    # Check degrees
    for i in range(n_left):
        if _popcount(line_adj[i * 9]) >= 4:  # This isn't right, line_adj is for the line graph
            pass
    
    # Better: check the bipartite graph directly
    # Reconstruct bipartite adjacency from line graph? No, we should pass bip_adj.
    # Let me restructure.
    return True  # Placeholder

def _bipartite_has_k22(bip_adj):
    """Check if bipartite graph has a K_{2,2} (4-cycle)."""
    n_left = 9
    n_right = 9
    for i1 in range(n_left):
        for i2 in range(i1 + 1, n_left):
            common = bip_adj[i1] & bip_adj[i2]
            if _popcount(common) >= 2:
                return True
    return False

def _bipartite_max_degree(bip_adj):
    """Check max degree in bipartite graph."""
    n_left = 9
    n_right = 9
    max_deg = 0
    for i in range(n_left):
        max_deg = max(max_deg, _popcount(bip_adj[i]))
    for j in range(n_right):
        deg = 0
        for i in range(n_left):
            if bip_adj[i] & (1 << j):
                deg += 1
        max_deg = max(max_deg, deg)
    return max_deg

def _is_line_graph_k4_free_from_bip(bip_adj):
    """
    Line graph of a bipartite graph is K4-free iff:
    1. No vertex has degree >= 4 (no K_{1,4} star)
    2. No K_{2,2} (no 4-cycle)
    """
    if _bipartite_max_degree(bip_adj) >= 4:
        return False
    if _bipartite_has_k22(bip_adj):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 18
    n_left = 9
    n_right = 9
    full_mask = (1 << n) - 1
    
    # Initial graph: Turan T(18,3)
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(n):
        for j in range(i + 1, n):
            part_i = 0 if i < 6 else (1 if i < 12 else 2)
            part_j = 0 if j < 6 else (1 if j < 12 else 2)
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_score = _max_triangle_free_bitmask(adj)
    best_adj = adj[:]
    
    # Also try line graph approach
    # Initialize bipartite graph with random edges
    best_bip_adj = None
    best_bip_score = 999
    
    # Generate random bipartite graph and check
    for trial in range(50):
        bip_adj = [0] * n_left
        for i in range(n_left):
            for j in range(n_right):
                if random.random() < 0.3:
                    bip_adj[i] |= (1 << j)
        
        if _is_line_graph_k4_free_from_bip(bip_adj):
            line_adj = _line_graph_of_bipartite(bip_adj)
            score = _max_triangle_free_bitmask(line_adj)
            if score < best_bip_score:
                best_bip_score = score
                best_bip_adj = bip_adj[:]
    
    if best_bip_adj is not None:
        line_adj = _line_graph_of_bipartite(best_bip_adj)
        if best_bip_score < best_score:
            best_score = best_bip_score
            best_adj = line_adj[:]
    
    # Local Search on Turan graph
    def get_edges_within_parts(adj):
        edges = []
        for p in range(3):
            for i in range(len(parts[p])):
                for j in range(i + 1, len(parts[p])):
                    u = parts[p][i]
                    v = parts[p][j]
                    if not (adj[u] & (1 << v)):
                        edges.append((u, v))
        return edges

    def get_edges_between_parts(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                part_i = 0 if i < 6 else (1 if i < 12 else 2)
                part_j = 0 if j < 6 else (1 if j < 12 else 2)
                if part_i != part_j and (adj[i] & (1 << j)):
                    edges.append((i, j))
        return edges

    # Simulated annealing on bipartite graph
    def _bipartite_score(bip_adj):
        if not _is_line_graph_k4_free_from_bip(bip_adj):
            return 999
        line_adj = _line_graph_of_bipartite(bip_adj)
        return _max_triangle_free_bitmask(line_adj)
    
    # Start SA from best_bip_adj if available, else random
    if best_bip_adj is not None:
        sa_bip_adj = best_bip_adj[:]
        sa_score = best_bip_score
    else:
        sa_bip_adj = [0] * n_left
        for i in range(n_left):
            for j in range(n_right):
                if random.random() < 0.3:
                    sa_bip_adj[i] |= (1 << j)
        sa_score = _bipartite_score(sa_bip_adj)
    
    sa_best_bip_adj = sa_bip_adj[:]
    sa_best_score = sa_score
    
    temperature = 1.0
    cooling = 0.9995
    
    attempts = 0
    while time.time() < deadline - 5:
        # Alternate between Turan local search and bipartite SA
        if random.random() < 0.5:
            # Turan local search
            if random.random() < 0.7:
                intra_edges = get_edges_within_parts(adj)
                if intra_edges:
                    u, v = random.choice(intra_edges)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if _is_k4_free_bitmask(adj):
                        score = _max_triangle_free_bitmask(adj)
                        if score < best_score:
                            best_score = score
                            best_adj = adj[:]
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            else:
                inter_edges = get_edges_between_parts(adj)
                if inter_edges:
                    u, v = random.choice(inter_edges)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    score = _max_triangle_free_bitmask(adj)
                    if score < best_score:
                        best_score = score
                        best_adj = adj[:]
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Bipartite SA: try an edge swap
            if sa_score == 999:
                # Reset to random
                sa_bip_adj = [0] * n_left
                for i in range(n_left):
                    for j in range(n_right):
                        if random.random() < 0.3:
                            sa_bip_adj[i] |= (1 << j)
                sa_score = _bipartite_score(sa_bip_adj)
                if sa_score < sa_best_score:
                    sa_best_score = sa_score
                    sa_best_bip_adj = sa_bip_adj[:]
                continue
            
            # Pick a random edge to flip
            i = random.randint(0, n_left - 1)
            j = random.randint(0, n_right - 1)
            bit = 1 << j
            old_val = sa_bip_adj[i] & bit
            sa_bip_adj[i] ^= bit
            
            new_score = _bipartite_score(sa_bip_adj)
            
            if new_score < sa_score:
                sa_score = new_score
                if new_score < sa_best_score:
                    sa_best_score = new_score
                    sa_best_bip_adj = sa_bip_adj[:]
            else:
                # Accept with probability exp(-(new_score - sa_score) / temperature)
                if new_score < 999 and random.random() < pow(2.718281828, -(new_score - sa_score) / max(temperature, 0.01)):
                    sa_score = new_score
                    if new_score < sa_best_score:
                        sa_best_score = new_score
                        sa_best_bip_adj = sa_bip_adj[:]
                else:
                    sa_bip_adj[i] ^= bit  # Revert
        
        temperature *= cooling
        attempts += 1
    
    # Check if bipartite SA found something better
    if sa_best_score < best_score:
        best_score = sa_best_score
        best_adj = _line_graph_of_bipartite(sa_best_bip_adj)[:]
    
    # Convert best_adj back to list of edges
    final_edges = []
    for i in range(n):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                final_edges.append([i, j])
            m ^= lsb
            
    return final_edges
