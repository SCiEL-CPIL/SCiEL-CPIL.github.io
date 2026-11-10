import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    adj = [0] * (N + 1)
    
    for u in range(1, N + 1):
        mask = 0
        for v in range(1, N + 1):
            if u == v:
                continue
            s = u + v
            prod = u * v
            if prod % s != 0:
                mask |= (1 << (v - 1))
        adj[u] = mask
        
    degrees = [0] * (N + 1)
    for u in range(1, N + 1):
        degrees[u] = adj[u].bit_count()
        
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    best_size = 0
    best_clique = []
    
    def expand(R, P_mask):
        nonlocal best_size, best_clique
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Get active vertices indices
        # We need to iterate in reverse order of 'vertices' for coloring
        # and forward for branching?
        # Tomita: Color in reverse order. Branch on vertex with highest color (which is last in reverse order, so first in forward order of the colored set? No.)
        # Let's stick to:
        # 1. Identify active vertices.
        # 2. Color them in reverse order of the global sorted list.
        # 3. The max color count is the bound.
        # 4. Branch on the vertex that received the highest color?
        #    Actually, branching on the vertex with the highest degree in P is a good heuristic.
        
        active_indices = []
        for i, v in enumerate(vertices):
            if (P_mask >> (v-1)) & 1:
                active_indices.append(i)
                
        if not active_indices:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return

        color_masks = []
        
        # Color in reverse order of active_indices
        for idx in reversed(active_indices):
            v = vertices[idx]
            n_mask = adj[v] & P_mask
            
            assigned = False
            for i, cm in enumerate(color_masks):
                if (n_mask & cm) == 0:
                    color_masks[i] = cm | (1 << (v-1))
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(1 << (v-1))
                
        max_color = len(color_masks)
        
        if len(R) + max_color <= best_size:
            return
            
        # Select best_v
        # Heuristic: Highest degree in P
        best_v = -1
        best_deg = -1
        for idx in active_indices:
            v = vertices[idx]
            deg = (adj[v] & P_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                
        # Branch 1: Include best_v
        R.append(best_v)
        new_P = P_mask & adj[best_v]
        new_P &= ~(1 << (best_v - 1))
        expand(R, new_P)
        R.pop()
        
        # Branch 2: Exclude best_v
        P_mask &= ~(1 << (best_v - 1))
        expand(R, P_mask)

    init_P = (1 << 256) - 1
    expand([], init_P)
    
    if not best_clique:
        best_clique = [1]
        
    return best_clique
