import time
from math import gcd

def construct():
    """
    Finds a large subset of [1, 512] such that lcm(a, b) <= 512 for all pairs.
    Uses a Greedy initialization followed by Tomita Maximum Clique with bitset optimization.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 512
    nodes = list(range(1, N + 1))
    
    # 1. Build adjacency bitmasks
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            l = (i * j) // g
            if l <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Compute global degrees and order
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = adj[i].bit_count()
        
    # Sort vertices by degree descending for Greedy and Coloring
    order = sorted(range(1, N + 1), key=lambda x: degrees[x], reverse=True)
    
    # 3. Greedy Clique Initialization
    # This provides a strong initial lower bound for B&B
    P_mask = 0
    for i in range(1, N + 1):
        P_mask |= (1 << i)
        
    greedy_clique = []
    current_P = P_mask
    while current_P:
        # Pick the first vertex in 'order' that is in current_P
        v = None
        for u in order:
            if current_P & (1 << u):
                v = u
                break
        if v is None:
            break
        greedy_clique.append(v)
        current_P &= adj[v]
        
    best_clique = greedy_clique
    best_size = len(greedy_clique)
    
    # Helper to get list of vertices from bitmask
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        return verts

    # Helper to greedily color vertices in P
    # Uses the global 'order' to iterate, which is faster than sorting every time
    def color_vertices(P_mask):
        if P_mask == 0:
            return []
            
        # We iterate through 'order' and pick vertices in P_mask
        # This ensures a consistent ordering
        verts_in_P = [v for v in order if P_mask & (1 << v)]
        
        if not verts_in_P:
            return []
            
        colors = [] # list of (vertex, color)
        uncolored = verts_in_P[:]
        current_color = 0
        
        while uncolored:
            current_color += 1
            colored_in_round = []
            new_uncolored = []
            for v in uncolored:
                # Check if v is adjacent to any in colored_in_round
                is_adjacent = False
                for u in colored_in_round:
                    if adj[v] & (1 << u):
                        is_adjacent = True
                        break
                if not is_adjacent:
                    colors.append((v, current_color))
                    colored_in_round.append(v)
                else:
                    new_uncolored.append(v)
            uncolored = new_uncolored
            
        # Sort colors by color descending
        colors.sort(key=lambda x: x[1], reverse=True)
        return colors

    def expand(clique, P_mask):
        nonlocal best_clique, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique[:]
            return
            
        # Pruning: if even taking all candidates doesn't beat best
        # Use coloring bound
        colored = color_vertices(P_mask)
        
        if not colored:
            return
            
        # Iterate through vertices in the order determined by coloring
        for v, c in colored:
            # Pruning bound
            if len(clique) + c <= best_size:
                break
                
            # Include v
            new_P = P_mask & adj[v]
            
            # Recurse
            expand(clique + [v], new_P)
            
            # Remove v from P_mask
            P_mask ^= (1 << v)
            
    # Initial call
    # We can start with the full set, but the greedy clique is already a valid clique.
    # The B&B will explore alternatives.
    expand([], P_mask)
    
    if not best_clique:
        return [1]
        
    return sorted(best_clique)

# EVOLVE-BLOCK-END
