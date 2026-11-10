"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (90s limit, leave 35s buffer)
    
    n = 256
    nums = list(range(1, n + 1))
    
    # Build compatibility masks (complement of conflict)
    # compat_masks[i] is a bitset of j such that (i*j) % (i+j) != 0
    compat_masks = [0] * (n + 1)
    conflict_masks = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if (i * j) % (i + j) != 0:
                compat_masks[i] |= (1 << (j - 1))
                compat_masks[j] |= (1 << (i - 1))
            else:
                conflict_masks[i] |= (1 << (j - 1))
                conflict_masks[j] |= (1 << (i - 1))
                
    # 1. Strong Heuristic Initialization
    # All odd numbers form a valid set because for distinct odd a,b,
    # a+b is even, ab is odd, so ab % (a+b) != 0.
    S_mask = 0
    for i in range(1, n + 1, 2):
        S_mask |= (1 << (i - 1))
        
    # Greedily add even numbers
    # Sort evens by degree in conflict graph (ascending) to pick "safe" ones
    evens = [i for i in range(2, n + 1, 2)]
    even_degrees = {i: bin(conflict_masks[i]).count('1') for i in evens}
    evens.sort(key=lambda x: even_degrees[x])
    
    for i in evens:
        # Check if i is compatible with all in S
        if (conflict_masks[i] & S_mask) == 0:
            S_mask |= (1 << (i - 1))
            
    best_size = bin(S_mask).count('1')
    best_set = set()
    tmp = S_mask
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        best_set.add(idx + 1)
        tmp ^= lsb

    # 2. Local Search (1-for-1 swap to improve size if possible? No, 1-for-1 keeps size same.
    # We need 1-for-2 or just rely on BK.
    # Let's do a simple "try to add" pass again in random order?
    # Actually, let's just run BK with this strong lower bound.
    
    # Helper: Extract vertices from mask
    def get_vertices(mask):
        verts = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            verts.append(idx + 1)
            mask ^= lsb
        return verts

    # Helper: Fast Greedy Coloring Count
    # Uses bitsets for color classes
    def greedy_color_count(P_mask):
        if P_mask == 0:
            return 0
        
        # Order vertices by degree in P (descending) might be better, but expensive.
        # Let's use ascending index for speed, it's a bound.
        verts = get_vertices(P_mask)
        
        color_classes = [] # list of bitsets
        
        for v in verts:
            placed = False
            v_bit = 1 << (v - 1)
            for c_idx, class_mask in enumerate(color_classes):
                # v can join class if no edge to any node in class
                # Edge exists in compat graph if (compat_masks[v] & class_mask) != 0
                # We need NO edge, so (compat_masks[v] & class_mask) == 0
                if (compat_masks[v] & class_mask) == 0:
                    color_classes[c_idx] |= v_bit
                    placed = True
                    break
            
            if not placed:
                color_classes.append(v_bit)
                
        return len(color_classes)

    # Global state for BK
    sys.setrecursionlimit(10000)
    
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if P_mask == 0 and X_mask == 0:
            # Maximal clique found
            size = bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best_set = set(get_vertices(R_mask))
            return
        
        # Weak bound: popcount
        if bin(R_mask).count('1') + bin(P_mask).count('1') <= best_size:
            return
            
        # Coloring Bound
        num_colors = greedy_color_count(P_mask)
        if bin(R_mask).count('1') + num_colors <= best_size:
            return
        
        # Pivot Selection
        PX_mask = P_mask | X_mask
        if PX_mask == 0:
            return
            
        # Choose u in P | X that maximizes |N(u) & P|
        pivot_val = 0
        pivot_mask = 0
        px_iter = PX_mask
        while px_iter:
            lsb = px_iter & -px_iter
            u_idx = lsb.bit_length() - 1
            u = u_idx + 1
            count = bin(compat_masks[u] & P_mask).count('1')
            if count > pivot_val:
                pivot_val = count
                pivot_mask = lsb
            px_iter ^= lsb
            
        if pivot_mask == 0:
            return
            
        u_idx = pivot_mask.bit_length() - 1
        u = u_idx + 1
        
        # Candidates are P \ N(pivot)
        # Nodes in P that are NOT neighbors of u
        candidates_mask = P_mask & ~compat_masks[u]
        
        cm = candidates_mask
        while cm:
            lsb = cm & -cm
            v_idx = lsb.bit_length() - 1
            v = v_idx + 1
            cm ^= lsb
            
            new_P = P_mask & compat_masks[v]
            new_X = X_mask & compat_masks[v]
            new_R = R_mask | lsb
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P_mask ^= lsb
            X_mask |= lsb
            
            if time.time() > deadline:
                return

    # Start BK
    init_P = 0
    for i in range(1, n + 1):
        init_P |= (1 << (i - 1))
        
    bron_kerbosch(0, init_P, 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
