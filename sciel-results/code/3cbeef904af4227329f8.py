import time
import random
from itertools import combinations

random.seed(42)
START_TIME = time.time()
TIME_LIMIT = 115

def build_adj(edges, n=16):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def is_triangle_free_bitmask(adj, n=16):
    for u in range(n):
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            common = adj[v] & adj[u]
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_exact(adj, n=16):
    """Exact 3-colorability check using backtracking with stronger heuristics."""
    colors = [-1] * n
    uncolored = (1 << n) - 1
    node_order = list(range(n))
    node_order.sort(key=lambda x: adj[x].bit_count(), reverse=True)
    
    def dfs():
        nonlocal uncolored
        if uncolored == 0:
            return False
        
        # Pick vertex with maximum degree in remaining uncolored subgraph
        best_v = -1
        best_deg = -1
        m = uncolored
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (adj[v] & uncolored).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        v = best_v
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        # Try colors in order of least constraint (fewest conflicts with uncolored neighbors)
        color_scores = []
        for c in range(3):
            if forbidden & (1 << c):
                continue
            # Count how many uncolored neighbors already have this color forbidden
            score = 0
            nn = adj[v] & uncolored
            while nn:
                lsb_nn = nn & -nn
                nb2 = lsb_nn.bit_length() - 1
                nn ^= lsb_nn
                if colors[nb2] == c:
                    score += 1
            color_scores.append((score, c))
        
        color_scores.sort(reverse=True)
        
        for score, c in color_scores:
            colors[v] = c
            uncolored &= ~(1 << v)
            if not dfs():
                return False
            uncolored |= 1 << v
            colors[v] = -1
        
        return True
    
    return dfs()

def greedy_3_colorable(adj, n=16, max_trials=8):
    for _ in range(max_trials):
        order = list(range(n))
        random.shuffle(order)
        colors = [-1] * n
        for node in order:
            used = 0
            nb_mask = adj[node]
            mm = nb_mask
            while mm:
                lsb = mm & -mm
                nb = lsb.bit_length() - 1
                mm ^= lsb
                if colors[nb] != -1:
                    used |= 1 << colors[nb]
            # Try colors in order of least usage among uncolored neighbors
            color_usage = [0, 0, 0]
            nn = adj[node]
            while nn:
                lsb_nn = nn & -nn
                nb2 = lsb_nn.bit_length() - 1
                nn ^= lsb_nn
                if colors[nb2] != -1:
                    color_usage[colors[nb2]] += 1
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[node] = c
                    break
            else:
                break
        else:
            return True
    return False

def is_not_3_colorable_fast(adj, n=16):
    if greedy_3_colorable(adj, n, max_trials=6):
        return False
    return is_not_3_colorable_exact(adj, n)

def build_groetzsch_edges():
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def get_groetzsch_adj(n=11):
    edges = build_groetzsch_edges()
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def generate_blowup_distributions(num_extra, num_original):
    dists = []
    def rec(rem, idx, current):
        if idx == num_original - 1:
            current.append(rem)
            dists.append(tuple(current))
            current.pop()
            return
        for i in range(rem + 1):
            current.append(i)
            rec(rem - i, idx + 1, current)
            current.pop()
    rec(num_extra, 0, [])
    return dists

def local_search_add_edges(adj, edges, n=16, deadline=None):
    """Systematically try to add edges while maintaining triangle-free and not 3-colorable."""
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try all potential edges, prioritizing those between high-degree vertices
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    continue
                if adj[u] & adj[v]:
                    continue
                # Score: prefer adding edges that connect vertices with fewer current edges
                # (more "room" to add) and maintain balance
                deg_u = adj[u].bit_count()
                deg_v = adj[v].bit_count()
                score = (n - 1 - deg_u) + (n - 1 - deg_v)
                candidates.append((score, u, v))
        
        candidates.sort(reverse=True)
        
        for score, u, v in candidates:
            if deadline and time.time() > deadline:
                break
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edges.append([u, v])
            
            if not is_not_3_colorable_fast(adj, n):
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edges.pop()
            else:
                improved = True
    return edges

def local_search_swaps(adj, edges, n=16, deadline=None):
    """Try swapping edges to escape local optima: remove one edge, add another."""
    improved = True
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try removing each edge and see if we can add a better one
        for idx in range(len(edges) - 1, -1, -1):
            if deadline and time.time() > deadline:
                break
            u, v = edges[idx]
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            edges.pop(idx)
            
            found = False
            candidates = []
            for a in range(n):
                for b in range(a+1, n):
                    if adj[a] & (1 << b):
                        continue
                    if adj[a] & adj[b]:
                        continue
                    # Prefer edges that increase density balance
                    deg_a = adj[a].bit_count()
                    deg_b = adj[b].bit_count()
                    # Penalize if this would create imbalance
                    max_deg = max(deg_a, deg_b)
                    min_deg = min(deg_a, deg_b)
                    score = (n - 1 - deg_a) + (n - 1 - deg_b) - 0.5 * (max_deg - min_deg)
                    candidates.append((score, a, b))
            
            candidates.sort(reverse=True)
            
            for score, a, b in candidates:
                if deadline and time.time() > deadline:
                    break
                adj[a] |= 1 << b
                adj[b] |= 1 << a
                edges.append([a, b])
                
                if is_not_3_colorable_fast(adj, n):
                    found = True
                    improved = True
                    break
                else:
                    adj[a] ^= 1 << b
                    adj[b] ^= 1 << a
                    edges.pop()
            
            if not found:
                edges.append([u, v])
                adj[u] |= 1 << v
                adj[v] |= 1 << u
    return edges

def subtractive_search(n=16, deadline=None):
    """Start from complete graph, remove edges to satisfy triangle-free,
    then try to add back edges that maintain non-3-colorability."""
    # Start with complete graph
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edges.append([u, v])
    
    # Remove edges to make triangle-free: for each triangle, remove one edge
    # Strategy: greedily remove edges that are in the most triangles
    changed = True
    while changed and (deadline is None or time.time() < deadline):
        changed = False
        # Find all triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    continue
                common = adj[u] & adj[v]
                mm = common
                while mm:
                    lsb = mm & -mm
                    w = lsb.bit_length() - 1
                    mm ^= lsb
                    if w > v:
                        triangles.append((u, v, w))
        
        if not triangles:
            break
        
        # Count edge frequencies in triangles
        edge_count = {}
        for u, v, w in triangles:
            for e in [(u,v), (v,w), (u,w)]:
                key = (min(e), max(e))
                edge_count[key] = edge_count.get(key, 0) + 1
        
        # Remove the edge that's in the most triangles
        max_count = max(edge_count.values())
        for (u, v), cnt in edge_count.items():
            if cnt == max_count and (adj[u] & (1 << v)):
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edges.remove([u, v])
                changed = True
                break
    
    # Now try to add back edges that maintain triangle-free and non-3-colorability
    if not is_not_3_colorable_fast(adj, n):
        return None, 0
    
    current_edges = local_search_add_edges(adj, list(edges), n, deadline)
    if current_edges:
        return current_edges, len(current_edges)
    return None, 0

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    n = 16
    num_original = 11
    num_extra = n - num_original
    
    g_adj = get_groetzsch_adj(num_original)
    
    best_edges = []
    best_count = 0
    deadline = START_TIME + TIME_LIMIT
    
    # Phase A: Blow-up search with saturation
    dists = generate_blowup_distributions(num_extra, num_original)
    
    def expected_edges(dist):
        sizes = [1 + d for d in dist]
        total = 0
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    total += sizes[i] * sizes[j]
        return total
    
    dists.sort(key=expected_edges, reverse=True)
    
    for dist in dists:
        if time.time() > deadline:
            break
            
        sizes = [1 + d for d in dist]
        
        orig_map = [0] * n
        current_idx = 0
        for i in range(num_original):
            for _ in range(sizes[i]):
                orig_map[current_idx] = i
                current_idx += 1
        
        edges = []
        adj = [0] * n
        
        for i in range(num_original):
            for j in range(i+1, num_original):
                if g_adj[i] & (1 << j):
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            edges.append([u, v])
                            adj[u] |= 1 << v
                            adj[v] |= 1 << u
        
        # Saturate: add all triangle-safe non-edges
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        edges.append([u, v])
        
        if not is_triangle_free_bitmask(adj, n):
            continue
        
        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
        
        # Local search on this candidate
        current_edges = local_search_add_edges(adj, list(edges), n, deadline)
        if current_edges and len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
    
    # Phase B: Subtractive search from K16
    if time.time() < deadline:
        sub_edges, sub_count = subtractive_search(n, deadline)
        if sub_edges and sub_count > best_count:
            best_count = sub_count
            best_edges = sub_edges
    
    # Phase C: Polish the best solution with multiple passes
    if best_edges and time.time() < deadline:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        
        # Multiple refinement passes to converge to a local optimum
        for _ in range(4):
            if time.time() > deadline - 5:
                break
            current_edges = local_search_add_edges(adj, current_edges, n, deadline)
            if time.time() > deadline - 5:
                break
            current_edges = local_search_swaps(adj, current_edges, n, deadline)
            if time.time() > deadline - 5:
                break
            current_edges = local_search_add_edges(adj, current_edges, n, deadline)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
    
    if not best_edges:
        best_edges = build_groetzsch_edges()
    
    return best_edges
