import time
import random
import sys

# Increase recursion limit just in case, though we use iterative backtracking
sys.setrecursionlimit(10000)

def _has_c6_bitset(adj, n):
    """
    Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
    adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
    """
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            
            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                
                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    
                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        
                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        
                        if mask_v5:
                            return True
    return False

def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6)."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True

def _algebraic_coloring(n):
    """
    Construct a 4-coloring of K_n using finite field arithmetic over GF(19).
    This is a standard construction for multicolor Ramsey numbers.
    Colors are assigned based on the ratio (j-i) in GF(p).
    We map the non-zero elements of GF(19) to 4 colors.
    Since 19 = 4*4 + 3, we can partition the 18 non-zero elements into 4 sets of 4 or 5.
    However, a simpler mapping that works for R_4(C6) > 17 is:
    color(i,j) = floor( ( (j-i) % 19 ) / 5 ) if we adjust properly?
    
    Actually, let's use the known construction for R_4(C6) >= 17.
    A common construction uses GF(q) where q is a prime power.
    For R_4(C6), n=17 is known to be possible.
    
    Let's use a specific mapping for GF(19):
    The non-zero elements are 1..18.
    We want to color edges (i,j) with c in {0,1,2,3}.
    Let d = (j - i) % 19.
    If d == 0, they are the same vertex, but i < j so d != 0 in K_n subset.
    
    Partition {1, ..., 18} into 4 sets S0, S1, S2, S3 such that no set contains a "C6 structure".
    In GF(p), a 6-cycle corresponds to a specific additive/multiplicative structure.
    Actually, the condition "no monochromatic C6" in the complete graph coloring is equivalent to:
    For any 6 distinct vertices v0..v5, the colors of edges (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0) are not all the same.
    
    A known construction for R_4(C6) > 17 uses the Paley graph or similar.
    Let's try:
    color(i, j) = ( (j - i) % 19 ) // 5  ? No, 18/4 = 4.5.
    
    Let's use the following partition of GF(19)^*:
    S0 = {1, 7, 8, 12}
    S1 = {2, 3, 13, 18}
    S2 = {4, 5, 10, 16}
    S3 = {6, 9, 11, 14, 15, 17} -- Wait, sizes should be balanced or specific.
    
    Actually, let's just use a simple heuristic that is known to work for small n or verify a specific one.
    The prompt says "best known colouring has 17 vertices".
    I will use a deterministic search to find a valid coloring for n=17 if the algebraic one fails, 
    but first I'll try a standard GF(19) construction.
    
    Standard construction for R_4(C6):
    Use GF(19). Map x in GF(19) to color c if x in S_c.
    The sets S_c must be such that no 6-term arithmetic progression or similar cyclic structure is monochromatic?
    No, it's about C6 in the graph sense.
    
    Let's try this specific mapping which is known to avoid small cycles in some contexts:
    color = ( (j - i) % 19 ) % 4 ? 
    Let's test n=17 with random restarts if algebraic fails, but we need a fast baseline.
    """
    colors = []
    # Try a few simple algebraic mappings
    mappings = [
        lambda d: d % 4,
        lambda d: (d // 5) % 4,
        lambda d: (d % 5) % 4,
    ]
    
    for m in mappings:
        c_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % 19
                c_list.append(m(d))
        if _is_valid(c_list, n):
            return c_list
            
    return None

def _find_valid_coloring_backtrack(n, time_limit=10.0):
    """
    Backtracking search for a valid 4-coloring of K_n.
    We assign colors to edges (i,j) in lexicographic order.
    We maintain adjacency masks for each color.
    After each assignment, we check if any color class now contains a C6.
    To speed up, we can check only C6s involving the newly added edge.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    
    # adj[c][i] is bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(4)]
    
    # Precompute edge indices
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1
            
    def check_c6_involving_edge(c, u, v):
        """
        Check if adding edge (u,v) to color c creates a C6.
        A C6 containing (u,v) has the form u-v-a-b-c-d-u.
        So we need a path of length 4 from v to u in the graph of color c,
        using vertices distinct from u, v.
        Path: v - x1 - x2 - x3 - x4 - u
        """
        # x1 in N(v)
        mask_x1 = adj[c][v]
        while mask_x1:
            x1 = (mask_x1 & -mask_x1).bit_length() - 1
            mask_x1 &= mask_x1 - 1
            if x1 == u: continue # length 1, not 4
            
            # x2 in N(x1) \ {v}
            mask_x2 = adj[c][x1] & ~(1 << v)
            while mask_x2:
                x2 = (mask_x2 & -mask_x2).bit_length() - 1
                mask_x2 &= mask_x2 - 1
                if x2 == u or x2 == x1: continue
                
                # x3 in N(x2) \ {v, x1}
                mask_x3 = adj[c][x2] & ~(1 << v) & ~(1 << x1)
                while mask_x3:
                    x3 = (mask_x3 & -mask_x3).bit_length() - 1
                    mask_x3 &= mask_x3 - 1
                    if x3 == u or x3 == x1 or x3 == x2: continue
                    
                    # x4 in N(x3) \ {v, x1, x2}
                    mask_x4 = adj[c][x3] & ~(1 << v) & ~(1 << x1) & ~(1 << x2)
                    while mask_x4:
                        x4 = (mask_x4 & -mask_x4).bit_length() - 1
                        mask_x4 &= mask_x4 - 1
                        if x4 == u or x4 == x1 or x4 == x2 or x4 == x3: continue
                        
                        # Check if x4 is connected to u
                        if adj[c][x4] & (1 << u):
                            return True
        return False

    def backtrack(edge_pos):
        if edge_pos == num_edges:
            return True
        if time.time() > deadline:
            return False
            
        i = 0
        j = 0
        # Find i, j for edge_pos
        # This is slow. Precompute list of (i,j)
        # We'll pass the list instead.
        pass

    # Precompute edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
            
    def backtrack(edge_pos):
        if edge_pos == num_edges:
            return True
        if time.time() > deadline:
            return False
            
        i, j = edge_list[edge_pos]
        
        # Try each color
        for c in range(4):
            # Check if adding (i,j) with color c creates a C6
            if check_c6_involving_edge(c, i, j):
                continue
                
            # Assign
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            colors[edge_pos] = c
            
            if backtrack(edge_pos + 1):
                return True
                
            # Unassign
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            colors[edge_pos] = -1
            
        return False

    # Symmetry breaking: First edge (0,1) must be color 0
    # We can force this to reduce search space by factor 4
    # But we need to be careful. Let's just try all 4 for the first edge if we don't do it in the loop.
    # Actually, let's force (0,1) to be 0, (0,2) to be 0 or 1, etc.
    # For simplicity, we'll just run the standard backtrack.
    
    # Optimization: Force (0,1) = 0
    c0 = 0
    adj[c0][0] |= (1 << 1)
    adj[c0][1] |= (1 << 0)
    colors[0] = c0
    
    # Force (0,2) = 0 or 1?
    # If (0,1)=0, (0,2) can be 0,1,2,3.
    # If (0,2)=0, then 0-1-2 is a path.
    
    # Let's just start from edge 1
    if backtrack(1):
        return colors
    return None

def _construct_known_17():
    """
    Find a valid 4-coloring of K_17.
    """
    # Try algebraic first (fast)
    res = _algebraic_coloring(17)
    if res:
        return {'n': 17, 'colors': res}
        
    # Try backtracking with a short time limit
    res = _find_valid_coloring_backtrack(17, time_limit=5.0)
    if res:
        return {'n': 17, 'colors': res}
        
    # Fallback to smaller n if all else fails
    for n in range(16, 4, -1):
        res = _algebraic_coloring(n)
        if res:
            return {'n': n, 'colors': res}
        res = _find_valid_coloring_backtrack(n, time_limit=2.0)
        if res:
            return {'n': n, 'colors': res}
            
    # Random fallback for very small n
    random.seed(42)
    n = 5
    colors = [random.randint(0, 3) for _ in range(n*(n-1)//2)]
    return {'n': n, 'colors': colors}

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Get a valid baseline (n=17)
    baseline = _construct_known_17()
    best_n = baseline['n']
    best_colors = baseline['colors']
    
    # 2. Try to improve to n=18 using remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Use backtracking for n=18
        # This is likely to be very slow, but we try with the remaining budget
        # We might need a smarter search for n=18, but let's try the basic one first.
        # If it's too slow, it will return None when deadline is hit.
        res = _find_valid_coloring_backtrack(18, time_limit=remaining_time - 5.0)
        if res:
            best_n = 18
            best_colors = res
            
    # 3. If we still have time and didn't get 18, maybe try n=19? 
    # Unlikely to succeed with this simple backtracker, but we can check time.
    remaining_time = deadline - time.time()
    if remaining_time > 20.0 and best_n < 19:
        res = _find_valid_coloring_backtrack(19, time_limit=remaining_time - 5.0)
        if res:
            best_n = 19
            best_colors = res

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    result = construct()
    print(f"Found n={result['n']}")
    # Verify
    if _is_valid(result['colors'], result['n']):
        print("Valid")
    else:
        print("INVALID")
