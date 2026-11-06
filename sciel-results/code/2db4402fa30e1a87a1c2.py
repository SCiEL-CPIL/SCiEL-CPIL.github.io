import time
import random
from itertools import combinations

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
        
    # Copies 5-9 correspond to 0-4.
    # Center 10.
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in orig_edges:
        edges.append([v, 5+u])
        edges.append([u, 5+v])
        
    for i in range(5):
        edges.append([10, 5+i])
        
    return edges

def _mycielski_c6_edges():
    """
    Mycielski of C6.
    13 vertices.
    C6: 0-1-2-3-4-5-0
    Copies: 6-11
    Center: 12
    """
    edges = []
    for i in range(6):
        edges.append([i, (i+1)%6])
        
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0]]
    for u, v in orig_edges:
        edges.append([v, 6+u])
        edges.append([u, 6+v])
        
    for i in range(6):
        edges.append([12, 6+i])
        
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    return (adj[u] & adj[v]) == 0

def _is_3_colorable(adj):
    """
    Fast backtracking to check if graph is 3-colorable using DSATUR heuristic.
    """
    colors = [-1] * N
    # Precompute degrees
    degrees = [bin(adj[i]).count('1') for i in range(N)]
    
    # DSATUR: Order vertices by saturation (number of distinct colors in neighbors)
    # and then by degree.
    # We will dynamically update saturation during backtracking.
    
    # For speed, we use a static ordering based on initial degrees, 
    # but dynamic reordering is better. Let's try a simple static order first for speed,
    # and if it fails, we can try dynamic.
    # Actually, dynamic DSATUR is much more effective for 3-colorability.
    
    def get_saturation(v, colors):
        sat = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if colors[u] != -1:
                sat |= (1 << colors[u])
        return bin(sat).count('1')
    
    def backtrack(idx, colored_count):
        if idx == N:
            return True
        
        # Select next vertex: max saturation, then max degree
        best_v = -1
        best_sat = -1
        best_deg = -1
        for v in range(N):
            if colors[v] == -1:
                sat = get_saturation(v, colors)
                deg = degrees[v]
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_v = v
        
        if best_v == -1:
            return True
            
        v = best_v
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if colors[u] != -1:
                forbidden |= (1 << colors[u])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1, colored_count + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0, 0)

def _greedy_add_all(adj):
    """Greedily add all possible triangle-free edges."""
    changed = True
    while changed:
        changed = False
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1) and _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
    return adj

def _local_search(adj, deadline, rng, min_edges=0):
    """
    Local search to maximize edges while maintaining 4-chromaticity.
    """
    best_cnt = min_edges
    best_adj = None
    
    current_adj = [a for a in adj]
    _greedy_add_all(current_adj)
    current_cnt = _edge_count(current_adj)
    
    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
        best_cnt = current_cnt
        best_adj = [a for a in current_adj]
    
    stall_count = 0
    max_stall = 15
    
    while time.time() < deadline:
        improved = False
        stall_count += 1
        
        if stall_count > max_stall:
            # Restart with perturbation
            current_adj = [a for a in best_adj] if best_adj else [a for a in adj]
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (current_adj[u] >> v) & 1]
            if existing:
                for _ in range(rng.randint(2, 5)):
                    if not existing:
                        break
                    idx = rng.randint(0, len(existing) - 1)
                    u, v = existing[idx]
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    existing = [(uu, vv) for uu in range(N) for vv in range(uu + 1, N) 
                               if (current_adj[uu] >> vv) & 1]
            _greedy_add_all(current_adj)
            current_cnt = _edge_count(current_adj)
            stall_count = 0
            continue
        
        # Try to add single edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if not ((current_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_cnt += 1
                improved = True
                
                if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
                break
        
        if improved:
            stall_count = 0
            continue
        
        # Try remove-1-add-2 swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if (current_adj[u] >> v) & 1]
        rng.shuffle(existing)
        
        found_swap = False
        for rem_u, rem_v in existing:
            if time.time() > deadline or found_swap:
                break
            
            adj_copy = [a for a in current_adj]
            adj_copy[rem_u] &= ~(1 << rem_v)
            adj_copy[rem_v] &= ~(1 << rem_u)
            
            # Find addable edges
            addable = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not ((adj_copy[u] >> v) & 1) and _is_triangle_free_add(adj_copy, u, v):
                        addable.append((u, v))
            
            if len(addable) >= 2:
                rng.shuffle(addable)
                # Try pairs
                max_pairs = min(len(addable) * (len(addable) - 1) // 2, 20)
                count = 0
                for i in range(min(len(addable), 10)):
                    for j in range(i + 1, min(len(addable), 10)):
                        if count >= max_pairs or time.time() > deadline:
                            break
                        count += 1
                        e1, e2 = addable[i], addable[j]
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        test_adj = [a for a in adj_copy]
                        test_adj[u1] |= (1 << v1)
                        test_adj[v1] |= (1 << u1)
                        if _is_triangle_free_add(test_adj, u2, v2):
                            test_adj[u2] |= (1 << v2)
                            test_adj[v2] |= (1 << u2)
                            cnt = _edge_count(test_adj)
                            if cnt > current_cnt and not _is_3_colorable(test_adj):
                                current_adj = test_adj
                                current_cnt = cnt
                                improved = True
                                found_swap = True
                                if current_cnt > best_cnt:
                                    best_cnt = current_cnt
                                    best_adj = [a for a in current_adj]
                                break
                    if found_swap or time.time() > deadline:
                        break
            elif len(addable) == 1:
                u, v = addable[0]
                adj_copy[u] |= (1 << v)
                adj_copy[v] |= (1 << u)
                cnt = _edge_count(adj_copy)
                if cnt > current_cnt and not _is_3_colorable(adj_copy):
                    current_adj = adj_copy
                    current_cnt = cnt
                    improved = True
                    found_swap = True
                    if current_cnt > best_cnt:
                        best_cnt = current_cnt
                        best_adj = [a for a in current_adj]
            
        if improved:
            stall_count = 0
            continue
        
        if not improved:
            stall_count += 1
            
    return best_adj, best_cnt

def construct():
    start = time.time()
    deadline = start + 115.0
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    seeds = []
    
    # Seed 1: Grötzsch + 5 random vertices
    grotzsch_edges = _groetzsch_edges()
    grotzsch_adj = _adj_from_edges(grotzsch_edges)
    seeds.append(grotzsch_adj)
    
    # Seed 2: Mycielski C6 + 3 random vertices
    myc_c6_edges = _mycielski_c6_edges()
    myc_c6_adj = _adj_from_edges(myc_c6_edges)
    seeds.append(myc_c6_adj)
    
    # Seed 3: Random triangle-free graph
    random_adj = [0] * N
    for u in range(N):
        for v in range(u + 1, N):
            if rng.random() < 0.3:
                if _is_triangle_free_add(random_adj, u, v):
                    random_adj[u] |= (1 << v)
                    random_adj[v] |= (1 << u)
    seeds.append(random_adj)
    
    # Seed 4: Complete bipartite K8,8 (not 4-chromatic, but good structure)
    # Actually K8,8 is 2-chromatic. We need to add edges to make it 4-chromatic.
    # Let's try a random dense graph and filter.
    
    max_trials = 30
    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a seed
        seed = seeds[trial % len(seeds)]
        adj = [a for a in seed]
        
        # Perturb
        if trial > 0:
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (adj[u] >> v) & 1]
            for _ in range(rng.randint(1, 3)):
                if existing:
                    idx = rng.randint(0, len(existing) - 1)
                    u, v = existing[idx]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    existing = [(uu, vv) for uu in range(N) for vv in range(uu + 1, N) 
                               if (adj[uu] >> vv) & 1]
        
        result_adj, result_cnt = _local_search(adj, deadline, rng, best_cnt)
        
        if result_adj and result_cnt > best_cnt:
            best_cnt = result_cnt
            best_adj = result_adj
    
    if best_adj is None:
        best_adj = grotzsch_adj
        best_cnt = _edge_count(best_adj)
    else:
        _greedy_add_all(best_adj)
        best_cnt = _edge_count(best_adj)
    
    if _is_3_colorable(best_adj):
        best_adj = grotzsch_adj
        best_cnt = _edge_count(best_adj)
    
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
