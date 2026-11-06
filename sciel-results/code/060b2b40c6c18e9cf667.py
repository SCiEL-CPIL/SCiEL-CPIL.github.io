import time
import random

# ---------- Bitmask-based helpers ----------

def _edge_mask_to_adj_masks(edge_mask, n=16):
    """
    Convert a single integer representing the edge set to a list of neighbor masks.
    edge_mask: int where bit (u*n + v) is 1 if edge (u,v) exists.
    Returns: list of n integers, where adj_mask[u] has bits set for neighbors of u.
    """
    adj_masks = [0] * n
    # Iterate through all possible edges
    for u in range(n):
        for v in range(u + 1, n):
            bit = 1 << (u * n + v)
            if edge_mask & bit:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
    return adj_masks

def _is_3_colorable(adj_masks, n=16):
    """
    Check if graph defined by adj_masks is 3-colorable.
    Uses backtracking with bitmasks for speed.
    """
    # Order vertices by descending degree
    order = sorted(range(n), key=lambda v: -bin(adj_masks[v]).count('1'))
    
    color = [-1] * n
    color_mask = [0, 0, 0]
    
    # Precompute forbidden masks? No, dynamic is better for backtracking.
    
    def backtrack(i):
        v = order[i]
        # For each color, check if it's valid
        for c in range(3):
            if adj_masks[v] & color_mask[c]:
                continue
            # Assign color c
            color[v] = c
            color_mask[c] |= (1 << v)
            
            if i == n - 1:
                return True
            
            if backtrack(i + 1):
                return True
            
            # Undo
            color_mask[c] &= ~(1 << v)
        
        color[v] = -1
        return False

    return backtrack(0)

def _groetzsch_edge_mask():
    """
    Return the edge mask for the Grötzsch graph (Mycielski of C5) on 11 vertices.
    Vertices 0-10 are used. 11-15 are isolated initially.
    """
    n = 16
    mask = 0
    # C5: 0-1, 1-2, 2-3, 3-4, 4-0
    cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    for u, v in cycle:
        mask |= 1 << (u * n + v)
    
    # Mycielski construction:
    # For each u in C5, add edge (u, 5+u) and (v, 5+u) where v is neighbor of u in C5?
    # Actually, standard Mycielski:
    # New vertices 5..9 corresponding to 0..4.
    # Edges: (u, 5+u) for all u in 0..4.
    # Edges: (u, 5+v) if (u,v) in E(C5).
    
    for u in range(5):
        mask |= 1 << (u * n + 5 + u)
        for v in range(5):
            if (u, v) in cycle or (v, u) in cycle:
                mask |= 1 << (u * n + 5 + v)
    
    # New vertex 10 connected to all 5..9
    for u in range(5):
        mask |= 1 << (10 * n + 5 + u)
        
    return mask

def construct():
    start = time.time()
    deadline = start + 115.0
    n = 16
    
    best_mask = 0
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def get_edge_count(mask):
        return bin(mask).count('1')

    def try_add_edge(mask, u, v):
        if u == v: return None
        if u > v: u, v = v, u
        bit = 1 << (u * n + v)
        if mask & bit: return None
        
        new_mask = mask | bit
        
        # Check triangle-free
        # Neighbors of u and v must be disjoint
        adj_u = 0
        adj_v = 0
        for w in range(n):
            if w != u and w != v:
                if mask & (1 << (min(u,w)*n + max(u,w))):
                    adj_u |= (1 << w)
                if mask & (1 << (min(v,w)*n + max(v,w))):
                    adj_v |= (1 << w)
        
        if adj_u & adj_v:
            return None
            
        # Check 3-colorability
        adj_masks = _edge_mask_to_adj_masks(new_mask, n)
        if _is_3_colorable(adj_masks, n):
            return None
            
        return new_mask

    def run_greedy(seed):
        random.seed(seed)
        mask = _groetzsch_edge_mask()
        
        # Precompute adj masks for current state to speed up triangle check?
        # The triangle check above is O(N). We can optimize it if we maintain adj masks.
        # Let's maintain adj masks in the loop.
        
        adj_masks = _edge_mask_to_adj_masks(mask, n)
        
        # List of all pairs
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in pairs:
                if not elapsed_ok(): break
                bit = 1 << (u * n + v)
                if mask & bit: continue
                
                # Triangle check using adj_masks
                if adj_masks[u] & adj_masks[v]:
                    continue
                
                # Tentative add
                new_mask = mask | bit
                new_adj_masks = adj_masks[:] # Copy
                new_adj_masks[u] |= (1 << v)
                new_adj_masks[v] |= (1 << u)
                
                if _is_3_colorable(new_adj_masks, n):
                    continue
                
                # Accept
                mask = new_mask
                adj_masks = new_adj_masks
                changed = True
                
        # Local Search: 1-add, 1-swap
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Add
            for u, v in pairs:
                if not elapsed_ok(): break
                bit = 1 << (u * n + v)
                if mask & bit: continue
                if adj_masks[u] & adj_masks[v]: continue
                
                new_mask = mask | bit
                new_adj_masks = adj_masks[:]
                new_adj_masks[u] |= (1 << v)
                new_adj_masks[v] |= (1 << u)
                
                if _is_3_colorable(new_adj_masks, n): continue
                
                mask = new_mask
                adj_masks = new_adj_masks
                improved = True
                break # Restart scan
            
            if improved or not elapsed_ok(): continue
            
            # Phase 2: Swap
            # Pick a random edge to remove
            # Get list of edges
            edges_in_mask = []
            for u in range(n):
                for v in range(u + 1, n):
                    if mask & (1 << (u * n + v)):
                        edges_in_mask.append((u, v))
                        
            if not edges_in_mask: continue
            
            u_rem, v_rem = random.choice(edges_in_mask)
            
            # Remove it
            mask_rem = mask & ~(1 << (u_rem * n + v_rem))
            adj_rem = adj_masks[:]
            adj_rem[u_rem] &= ~(1 << v_rem)
            adj_rem[v_rem] &= ~(1 << u_rem)
            
            # Try to add a random non-edge
            # Generate candidates
            candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not (mask_rem & (1 << (u * n + v)))]
            random.shuffle(candidates)
            
            swapped = False
            for u_add, v_add in candidates[:50]: # Limit attempts per swap
                if not elapsed_ok(): break
                if adj_rem[u_add] & adj_rem[v_add]: continue
                
                mask_new = mask_rem | (1 << (u_add * n + v_add))
                adj_new = adj_rem[:]
                adj_new[u_add] |= (1 << v_add)
                adj_new[v_add] |= (1 << u_add)
                
                if _is_3_colorable(adj_new, n): continue
                
                # Accept swap
                mask = mask_new
                adj_masks = adj_new
                swapped = True
                break
            
            if swapped:
                improved = True
                
        return mask, get_edge_count(mask)

    # Run multiple instances
    for i in range(100): # Will break early if deadline
        if not elapsed_ok(): break
        
        seed = i * 12345 + 67890
        mask, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_mask = mask
            
    # Convert best_mask to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_mask & (1 << (u * n + v)):
                edges.append([u, v])
                
    # Fallback if something went wrong
    if not edges:
        # Return Groetzsch
        mask = _groetzsch_edge_mask()
        for u in range(n):
            for v in range(u + 1, n):
                if mask & (1 << (u * n + v)):
                    edges.append([u, v])
                    
    return edges
