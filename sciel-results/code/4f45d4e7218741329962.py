import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

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
    """Exact backtracking 3-colorability check with MRV heuristic."""
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False
        
        # Select vertex with most colored neighbors (MRV heuristic)
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            score = 0
            nb_mask = adj[v]
            mm = nb_mask
            while mm:
                lsb_nb = mm & -mm
                nb = lsb_nb.bit_length() - 1
                mm ^= lsb_nb
                if colors[nb] != -1:
                    score += 1
            if score > best_score:
                best_score = score
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
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if not dfs(uncolored_mask & ~(1 << v)):
                    return False
                colors[v] = -1
        
        return True
    
    initial_mask = (1 << n) - 1
    return dfs(initial_mask)

def greedy_3_colorable(adj, n=16, max_trials=20):
    """Quick greedy check - if it succeeds, graph is 3-colorable."""
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
    """Fast check: greedy first, then exact if needed."""
    if greedy_3_colorable(adj, n, max_trials=15):
        return False
    return is_not_3_colorable_exact(adj, n)

def build_groetzsch_edges():
    """Build edges of the Grötzsch graph (11 vertices, 20 edges)."""
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
    """Generate all distributions of num_extra extra vertices among num_original groups."""
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

def construct_blowup_graph(dist, base_adj, base_n, total_n):
    """Construct a blowup graph from a distribution and base graph."""
    sizes = [1 + d for d in dist]
    
    # Map each vertex to its base vertex
    orig_map = [0] * total_n
    current_idx = 0
    for i in range(base_n):
        for _ in range(sizes[i]):
            orig_map[current_idx] = i
            current_idx += 1
    
    edges = []
    adj = [0] * total_n
    
    # Add edges between groups that are connected in base graph
    for i in range(base_n):
        for j in range(i + 1, base_n):
            if base_adj[i] & (1 << j):
                verts_i = [v for v in range(total_n) if orig_map[v] == i]
                verts_j = [v for v in range(total_n) if orig_map[v] == j]
                for u in verts_i:
                    for v in verts_j:
                        edges.append([u, v])
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
    
    return adj, edges

def local_search_maximize(adj, current_edges, n=16, deadline=None, max_iterations=200):
    """Local search: try to add edges while maintaining triangle-free and non-3-colorable."""
    improved = True
    iterations = 0
    while improved and iterations < max_iterations and (deadline is None or time.time() < deadline):
        improved = False
        iterations += 1
        
        # Find all candidate edges that would maintain triangle-free
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if deadline and time.time() > deadline:
                return current_edges
            
            # Try adding edge
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            
            if is_not_3_colorable_fast(adj, n):
                improved = True
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
                
        if not improved:
            break
    
    return current_edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 140
    
    n = 16
    deadline = START_TIME + TIME_LIMIT
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Blowup of Grötzsch graph (11 vertices) to 16 vertices
    base_n = 11
    num_extra = n - base_n
    g_adj = get_groetzsch_adj(base_n)
    
    # Precompute which non-edges in base graph can be saturated without creating triangles
    can_saturate = [[False] * base_n for _ in range(base_n)]
    for i in range(base_n):
        for j in range(i + 1, base_n):
            if not (g_adj[i] & (1 << j)):
                if (g_adj[i] & g_adj[j]) == 0:
                    can_saturate[i][j] = True
                    can_saturate[j][i] = True
    
    # Generate all distributions
    dists = generate_blowup_distributions(num_extra, base_n)
    
    # Evaluate each distribution
    for dist in dists:
        if time.time() > deadline:
            break
        
        adj, edges = construct_blowup_graph(dist, g_adj, base_n, n)
        
        # Try to saturate non-edges that can be safely added
        for i in range(base_n):
            for j in range(i + 1, base_n):
                if can_saturate[i][j]:
                    verts_i = [v for v in range(n) if (v // (1 + dist[i])) == i and v % (1 + dist[i]) < (1 + dist[i])]
                    # More robust: use orig_map
                    sizes = [1 + d for d in dist]
                    orig_map = [0] * n
                    current_idx = 0
                    for k in range(base_n):
                        for _ in range(sizes[k]):
                            orig_map[current_idx] = k
                            current_idx += 1
                    
                    verts_i = [v for v in range(n) if orig_map[v] == i]
                    verts_j = [v for v in range(n) if orig_map[v] == j]
                    for u in verts_i:
                        for v in verts_j:
                            if not (adj[u] & (1 << v)):
                                if (adj[u] & adj[v]) == 0:
                                    edges.append([u, v])
                                    adj[u] |= 1 << v
                                    adj[v] |= 1 << u
        
        if not is_triangle_free_bitmask(adj, n):
            continue
            
        if is_not_3_colorable_fast(adj, n):
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
    
    # Strategy 2: Intensive local search from best blowup graph
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        
        # Multiple rounds of local search with different random seeds
        for trial in range(30):
            if time.time() > deadline:
                break
            random.seed(1000 + trial * 777)
            current_edges = local_search_maximize(adj, current_edges, n, deadline, max_iterations=100)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                adj = build_adj(best_edges, n)
    
    # Strategy 3: If still not good enough, try blowup of other base graphs
    if best_count < 45:
        # Try different base graphs or parameters
        for trial in range(20):
            if time.time() > deadline:
                break
            
            random.seed(5000 + trial * 13)
            # Generate random triangle-free base and blowup
            base_adj, base_edges = generate_random_triangle_free_base(11, 0.4, seed=5000 + trial * 13)
            if base_adj is None:
                continue
            
            # Check if base is non-3-colorable
            if not is_not_3_colorable_fast(base_adj, 11):
                continue
            
            # Blowup to 16
            dists_small = generate_blowup_distributions(5, 11)
            for dist in dists_small[:50]:  # Limit to first 50 for time
                if time.time() > deadline:
                    break
                adj, edges = construct_blowup_graph(dist, base_adj, 11, 16)
                if is_triangle_free_bitmask(adj, 16) and is_not_3_colorable_fast(adj, 16):
                    if len(edges) > best_count:
                        best_count = len(edges)
                        best_edges = list(edges)
    
    # Strategy 4: Final intensive local search
    if best_edges:
        adj = build_adj(best_edges, n)
        current_edges = list(best_edges)
        for trial in range(20):
            if time.time() > deadline:
                break
            random.seed(9000 + trial * 31)
            current_edges = local_search_maximize(adj, current_edges, n, deadline, max_iterations=80)
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
                adj = build_adj(best_edges, n)
    
    # Final safety check
    if best_edges:
        adj = build_adj(best_edges, n)
        if not is_triangle_free_bitmask(adj, n):
            # Fallback to Grötzsch blowup
            dist = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5)
            adj, best_edges = construct_blowup_graph(dist, g_adj, base_n, n)
    
    return best_edges

def generate_random_triangle_free_base(n, edge_prob, seed=None):
    """Generate a random triangle-free graph on n vertices."""
    if seed is not None:
        random.seed(seed)
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < edge_prob:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])
    return adj, edges
