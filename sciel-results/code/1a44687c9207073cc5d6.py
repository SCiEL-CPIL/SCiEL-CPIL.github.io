import time
import random

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
        # Check all pairs of neighbors
        # If u is connected to v and w, and v is connected to w, triangle.
        # Iterate neighbors of u
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            # Check if v's neighbors intersect with u's neighbors (excluding u and v)
            # Actually, just check if adj[v] & adj[u] has any bit set that is not u or v?
            # Simpler: for each neighbor v of u, check if adj[v] & adj[u] has any common neighbor w.
            # If yes, triangle u-v-w.
            common = adj[v] & adj[u]
            # Remove u and v from common?
            # u is in adj[v], v is in adj[u].
            # If common has any bit other than u and v, it's a triangle.
            # common & ~(1<<u) & ~(1<<v)
            if common & ~(1 << u) & ~(1 << v):
                return False
    return True

def is_not_3_colorable_bitmask(adj, n=16):
    """
    Returns True if the graph is NOT 3-colorable.
    Uses backtracking with bitmasks for speed.
    """
    # colors: -1 uncolored, 0, 1, 2
    # We can represent colors as an array of ints, or use a more complex state.
    # For n=16, simple array is fine, but bitmask operations on neighbors are fast.
    
    colors = [-1] * n
    
    def dfs(uncolored_mask):
        if uncolored_mask == 0:
            return False # Successfully colored, so it IS 3-colorable
        
        # DSATUR heuristic: pick vertex with most colored neighbors
        best_v = -1
        best_score = -1
        m = uncolored_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Count colored neighbors
            score = 0
            nb_mask = adj[v]
            # Iterate neighbors
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
        
        # Determine available colors
        # A color c is forbidden if any neighbor has color c.
        forbidden = 0
        nb_mask = adj[v]
        mm = nb_mask
        while mm:
            lsb_nb = mm & -mm
            nb = lsb_nb.bit_length() - 1
            mm ^= lsb_nb
            if colors[nb] != -1:
                forbidden |= 1 << colors[nb]
        
        # Try available colors
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
    """Fast check. If it finds a coloring, graph IS 3-colorable."""
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
    if greedy_3_colorable(adj, n, max_trials=10):
        return False
    return is_not_3_colorable_bitmask(adj, n)

def build_groetzsch_edges():
    """Build Grötzsch graph on 11 vertices."""
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    for u, v in [[0,1],[1,2],[2,3],[3,4],[4,0]]:
        edges.append([5+u, v])
        edges.append([5+v, u])
    for u in range(5):
        edges.append([10, 5+u])
    return edges

def construct():
    global START_TIME, TIME_LIMIT
    START_TIME = time.time()
    TIME_LIMIT = 115
    
    # Start with Grötzsch graph (11 vertices)
    base_edges = build_groetzsch_edges()
    n = 16
    adj = build_adj(base_edges, n)
    current_edges = list(base_edges)
    
    best_edges = list(current_edges)
    best_count = len(current_edges)
    
    # Add vertices 11-15
    # Greedily connect them to existing structure
    for v in range(11, 16):
        # Connect to as many as possible without triangle
        candidates = list(range(11))
        random.shuffle(candidates)
        for c in candidates:
            if time.time() - START_TIME > TIME_LIMIT:
                return best_edges
            # Check triangle
            if adj[v] & adj[c] == 0:
                # Safe to add
                current_edges.append([v, c])
                adj[v] |= 1 << c
                adj[c] |= 1 << v
    
    # Local Search Phase
    deadline = START_TIME + TIME_LIMIT
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Generate candidate edges
        candidates = []
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)):
                    # Check triangle
                    if (adj[u] & adj[v]) == 0:
                        candidates.append((u, v))
        
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline:
                break
            
            # Tentatively add
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            current_edges.append([u, v])
            
            # Check colorability
            if is_not_3_colorable_fast(adj, n):
                # Keep edge
                best_count = len(current_edges)
                best_edges = list(current_edges)
                improved = True
            else:
                # Revert
                adj[u] ^= 1 << v
                adj[v] ^= 1 << u
                current_edges.pop()
        
        # If no improvement in a full pass, we might be stuck.
        # But since we shuffle, we try different orders.
        # If we truly can't add any edge, we break.
        if not improved:
            # Try a random restart? Or just break?
            # For now, break to avoid infinite loop if no edges can be added.
            break
            
    return best_edges
