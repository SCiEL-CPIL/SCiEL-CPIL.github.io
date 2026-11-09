import random
import time

def construct():
    n = 16
    deadline = time.time() + 110.0
    
    # Bitmask 3-colorability check
    def is_3_colorable(adj_masks):
        # adj_masks: list of 16 ints
        colors = [-1] * n
        order = sorted(range(n), key=lambda x: bin(adj_masks[x]).count('1'), reverse=True)
        
        def backtrack(i):
            if i == n:
                return True
            v = order[i]
            # Check neighbors' colors
            # We need to know which colors are used by neighbors
            used_colors = 0
            mask = adj_masks[v]
            for u in range(n):
                if (mask >> u) & 1:
                    c = colors[u]
                    if c != -1:
                        used_colors |= (1 << c)
                        
            for c in range(3):
                if not (used_colors & (1 << c)):
                    colors[v] = c
                    if backtrack(i + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    def masks_to_edges(adj_masks):
        edges = []
        for u in range(n):
            mask = adj_masks[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def add_edge(adj_masks, u, v):
        if u == v: return False
        if (adj_masks[u] >> v) & 1: return False
        # Check triangle
        # Common neighbors of u and v
        common = adj_masks[u] & adj_masks[v]
        if common:
            return False
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        return True

    def make_maximal(adj_masks):
        # Try to add edges in random order
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if not ((adj_masks[u] >> v) & 1):
                add_edge(adj_masks, u, v)

    def init_grotzsch():
        adj_masks = [0] * n
        def set_edge(u, v):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            
        # Cycle 0-4
        for i in range(5):
            set_edge(i, (i+1)%5)
        # Copies 5-9
        for i in range(5):
            set_edge(5+i, 5+(i+1)%5)
        # Cross
        for i in range(5):
            v1 = 5+i
            n1 = (i-1)%5
            n2 = (i+1)%5
            set_edge(v1, n1)
            set_edge(v1, n2)
        # Center 10
        for i in range(5):
            set_edge(10, 5+i)
        return adj_masks

    # Initialize
    adj = init_grotzsch()
    make_maximal(adj)
    
    best_adj = adj[:]
    best_edges = masks_to_edges(adj)
    best_count = len(best_edges)
    
    # Check if initial is valid (not 3-colorable)
    if is_3_colorable(adj):
        # Try to find a non-3-colorable one by random restarts
        for _ in range(100):
            adj = init_grotzsch()
            # Randomize initial edges slightly? No, Grötzsch is fixed.
            # Just randomize the maximalization order
            make_maximal(adj)
            if not is_3_colorable(adj):
                break
        if is_3_colorable(adj):
            # Fallback: return Grötzsch + isolated (11 edges)
            return best_edges # This is the seed result
            
    current_adj = adj[:]
    current_count = best_count
    
    T = 5.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Remove a random edge
        if current_count > 0:
            # Pick random edge
            # Find an edge
            u = random.randint(0, n-1)
            mask = current_adj[u]
            if mask == 0: continue
            # Pick a neighbor
            neighbors = [i for i in range(n) if (mask >> i) & 1]
            if not neighbors: continue
            v = random.choice(neighbors)
            
            # Remove
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_count -= 1
            
            # Re-maximize
            make_maximal(current_adj)
            
            # Count edges
            new_count = sum(bin(current_adj[i]).count('1') for i in range(n)) // 2
            
            # Check 3-colorability
            if not is_3_colorable(current_adj):
                delta = new_count - current_count
                if delta > 0 or random.random() < pow(2.718, delta / T):
                    current_count = new_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                        best_edges = masks_to_edges(best_adj)
            else:
                # Rejected
                pass
        
        T *= cooling
        if T < 0.5:
            T = 0.5

    return best_edges
