import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

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
    """Exact backtracking 3-colorability check."""
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

def greedy_3_colorable(adj, n=16, max_trials=30):
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

def build_clebsch_edges(n=16):
    """Build edges of the Clebsch graph (16 vertices, 40 edges)."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist in (1, 4):
                edges.append([u, v])
    return edges

def get_clebsch_adj(n=16):
    edges = build_clebsch_edges(n)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def branch_and_bound(n=16, deadline=None, initial_adj=None, initial_edges=None):
    """
    Branch-and-bound to find maximum edge triangle-free non-3-colorable graph.
    Orders edges, branches include/exclude, prunes with constraints.
    """
    if deadline is None:
        deadline = START_TIME + TIME_LIMIT
    
    # Generate all possible edges in a strategic order
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Order edges by some heuristic: prioritize edges that are in Clebsch graph
    clebsch_set = set()
    for u in range(n):
        for v in range(u+1, n):
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist in (1, 4):
                clebsch_set.add((u, v))
    
    # Sort: Clebsch edges first, then others
    all_edges.sort(key=lambda e: (0 if e in clebsch_set else 1, e[0], e[1]))
    
    if initial_adj is None:
        initial_adj = [0] * n
    if initial_edges is None:
        initial_edges = []
    
    best_edges = list(initial_edges)
    best_count = len(initial_edges)
    current_adj = list(initial_adj)
    current_edge_set = set(tuple(e) for e in initial_edges)
    
    # For each edge in all_edges, decide include or exclude
    # We'll do a recursive DFS with pruning
    
    node_count = [0]
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) maintains triangle-free."""
        if adj[u] & (1 << v):
            return False  # already exists
        # Check no common neighbor
        if adj[u] & adj[v]:
            return False
        return True
    
    def dfs(edge_idx, adj, edge_set, edge_count):
        if time.time() > deadline:
            return
        
        node_count[0] += 1
        
        if edge_idx == len(all_edges):
            # Check if this is a valid solution
            if edge_count > best_count:
                if is_not_3_colorable_exact(adj, n):
                    nonlocal best_edges, best_count
                    # Reconstruct edges
                    edges = []
                    for u in range(n):
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            v = lsb.bit_length() - 1
                            mask ^= lsb
                            if u < v:
                                edges.append([u, v])
                    best_edges = edges
                    best_count = edge_count
            return
        
        u, v = all_edges[edge_idx]
        
        # Option 1: Include this edge (if valid)
        if can_add_edge(adj, u, v):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_set.add((u, v))
            
            # Pruning: if remaining edges + current can't beat best, skip
            remaining = len(all_edges) - edge_idx - 1
            if edge_count + 1 + remaining <= best_count:
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                edge_set.remove((u, v))
            else:
                dfs(edge_idx + 1, adj, edge_set, edge_count + 1)
            
            adj[u] ^= 1 << v
            adj[v] ^= 1 << u
            edge_set.remove((u, v))
        
        # Option 2: Exclude this edge
        # Pruning: if current + remaining can't beat best, skip
        remaining = len(all_edges) - edge_idx - 1
        if edge_count + remaining > best_count:
            dfs(edge_idx + 1, adj, edge_set, edge_count)
    
    dfs(0, current_adj, current_edge_set, best_count)
    
    return best_edges, best_count

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 145
    
    n = 16
    deadline = START_TIME + TIME_LIMIT
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start from Clebsch graph and do branch-and-bound
    clebsch_adj = get_clebsch_adj(n)
    clebsch_edges = build_clebsch_edges(n)
    
    if is_triangle_free_bitmask(clebsch_adj, n) and is_not_3_colorable_fast(clebsch_adj, n):
        current_edges = list(clebsch_edges)
        current_adj = list(clebsch_adj)
        
        # Run branch-and-bound from this starting point
        if time.time() < deadline:
            bb_edges, bb_count = branch_and_bound(n, deadline, current_adj, current_edges)
            if bb_count > best_count:
                best_count = bb_count
                best_edges = bb_edges
    
    # Strategy 2: If B&B didn't find anything better, try local search from Clebsch
    if best_count <= len(clebsch_edges):
        for trial in range(20):
            if time.time() > deadline:
                break
            random.seed(100 + trial * 13)
            
            # Local search: try to add edges
            adj = list(clebsch_adj)
            current_edges = list(clebsch_edges)
            improved = True
            while improved and time.time() < deadline:
                improved = False
                candidates = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            if (adj[u] & adj[v]) == 0:
                                candidates.append((u, v))
                
                random.shuffle(candidates)
                
                for u, v in candidates:
                    if time.time() > deadline:
                        break
                    
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    current_edges.append([u, v])
                    
                    if is_not_3_colorable_fast(adj, n):
                        improved = True
                    else:
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
                        current_edges.pop()
                
                if not improved:
                    break
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = list(current_edges)
    
    # Strategy 3: Try different starting points
    for trial in range(5):
        if time.time() > deadline:
            break
        
        random.seed(5000 + trial * 13)
        # Generate a random dense triangle-free graph
        adj = [0] * n
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.4:
                    if (adj[u] & adj[v]) == 0:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        edges.append([u, v])
        
        if is_triangle_free_bitmask(adj, n) and is_not_3_colorable_fast(adj, n):
            # Try to improve with local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                candidates = []
                for u in range(n):
                    for v in range(u+1, n):
                        if not (adj[u] & (1 << v)):
                            if (adj[u] & adj[v]) == 0:
                                candidates.append((u, v))
                
                random.shuffle(candidates)
                
                for u, v in candidates:
                    if time.time() > deadline:
                        break
                    
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    edges.append([u, v])
                    
                    if is_not_3_colorable_fast(adj, n):
                        improved = True
                    else:
                        adj[u] ^= 1 << v
                        adj[v] ^= 1 << u
                        edges.pop()
                
                if not improved:
                    break
            
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = list(edges)
    
    # Final safety check
    if best_edges:
        adj = build_adj(best_edges, n)
        if not is_triangle_free_bitmask(adj, n):
            # Fallback to Clebsch
            best_edges = build_clebsch_edges(n)
    
    return best_edges
