import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    # Precompute edge indices for CW and CCW for each demand
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        cw_e = []
        curr = s
        while curr != t:
            cw_e.append(curr)
            curr = (curr + 1) % N
        cw_edges.append(cw_e)
        
        ccw_e = []
        curr = s
        while curr != t:
            curr = (curr - 1) % N
            ccw_e.append(curr)
        ccw_edges.append(ccw_e)
        weights.append(w)

    def calculate_loads(mask):
        loads = np.zeros(N)
        for i in range(D):
            w = weights[i]
            if (mask >> i) & 1:
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                for e in cw_edges[i]:
                    loads[e] += w
        return loads

    def get_max_load(mask):
        return np.max(calculate_loads(mask))

    best_max_load = float('inf')
    best_mask = 0
    
    # Local Search with Random Restarts
    # Since 2^15 is small, we can do many restarts quickly.
    # Or we can just do a full local search from a few good starts.
    
    # Start 1: All CW
    current_mask = 0
    current_max = get_max_load(current_mask)
    
    # Start 2: All CCW
    current_mask = (1 << D) - 1
    if get_max_load(current_mask) < current_max:
        best_max_load = current_max # Update best
        best_mask = current_mask
    # Actually, let's just run local search from a few random starts and a greedy start.
    
    def local_search(start_mask):
        mask = start_mask
        current_max = get_max_load(mask)
        improved = True
        while improved:
            improved = False
            for i in range(D):
                new_mask = mask ^ (1 << i)
                new_max = get_max_load(new_mask)
                if new_max < current_max:
                    mask = new_mask
                    current_max = new_max
                    improved = True
        return mask, current_max

    # Greedy initialization
    greedy_mask = 0
    loads = np.zeros(N)
    # Process demands in order of weight descending to make greedy more effective? 
    # Or just standard order. Let's stick to standard order for simplicity, 
    # but the local search will fix it.
    for i in range(D):
        w = weights[i]
        # Try CW
        loads_cw = loads.copy()
        for e in cw_edges[i]:
            loads_cw[e] += w
        max_cw = np.max(loads_cw)
        
        # Try CCW
        loads_ccw = loads.copy()
        for e in ccw_edges[i]:
            loads_ccw[e] += w
        max_ccw = np.max(loads_ccw)
        
        if max_cw <= max_ccw:
            # Choose CW
            loads = loads_cw
            # bit i remains 0
        else:
            # Choose CCW
            loads = loads_ccw
            greedy_mask |= (1 << i)
            
    # Run local search from greedy start
    mask_ls, max_ls = local_search(greedy_mask)
    if max_ls < best_max_load:
        best_max_load = max_ls
        best_mask = mask_ls
        
    # Run local search from a few random starts to escape local minima
    import random
    for _ in range(100): # 100 random restarts is very fast
        rand_mask = random.getrandbits(D)
        mask_ls, max_ls = local_search(rand_mask)
        if max_ls < best_max_load:
            best_max_load = max_ls
            best_mask = mask_ls

    # Construct result
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0 # CCW
        else:
            routing[i] = 0.0 # CW
            
    return routing
