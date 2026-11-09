import time
import math
import sys

def construct():
    start_time = time.time()
    DEADLINE = start_time + 87.0  # Safety margin for 90s limit
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    # Note: Bit k in compat[i] corresponds to node k+1.
    # So node 1 is bit 0, node 512 is bit 511.
    compat = [0] * (N + 1)
    
    # Precompute gcd/lcm checks
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # j is compatible with i
                compat[i] |= (1 << (j - 1))
                # i is compatible with j
                compat[j] |= (1 << (i - 1))
    
    # Precompute popcount for all possible masks up to 2^N is too large,
    # but we can use bit_count() which is much faster than bin(x).count('1')
    # Also precompute neighbor masks for faster access
    
    # 2. Tomita's Algorithm with Bitmasks and Dynamic Coloring
    
    best_clique = []
    
    # Greedy Lower Bound
    def greedy_clique(order=None):
        if order is None:
            # Sort by degree descending
            degrees = [(i, compat[i].bit_count()) for i in range(1, N+1)]
            degrees.sort(key=lambda x: -x[1])
            order = [d[0] for d in degrees]
        
        clique = []
        clique_mask = 0
        for node in order:
            if time.time() > DEADLINE:
                break
            # Check if node is compatible with all nodes in clique using bitmask
            if clique_mask == 0 or (compat[node] & clique_mask) == clique_mask:
                clique.append(node)
                clique_mask |= (1 << (node - 1))
        return clique

    # Try a couple of different orderings for the initial bound
    init_clique1 = greedy_clique()
    if init_clique1:
        best_clique = init_clique1
        
    # Try ordering by value ascending? (Small numbers are more compatible)
    init_clique2 = greedy_clique(order=list(range(1, N+1)))
    if len(init_clique2) > len(best_clique):
        best_clique = init_clique2

    sys.setrecursionlimit(10000)
    
    def expand(P_mask, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        # Simple bound
        if depth + P_mask.bit_count() <= len(best_clique):
            return
            
        # Extract vertices from P_mask
        P = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            P.append(idx + 1)
            temp ^= lsb
            
        # Calculate dynamic degree for each vertex in P
        # degree(v) = popcount(compat[v] & P_mask)
        P_with_deg = []
        for v in P:
            deg = (compat[v] & P_mask).bit_count()
            P_with_deg.append((deg, v))
            
        # Sort by degree descending (standard Tomita improvement)
        P_with_deg.sort(key=lambda x: -x[0])
        P = [v for _, v in P_with_deg]
        
        # Greedy Coloring
        # We assign colors to vertices in P such that no two adjacent vertices have the same color.
        # The number of colors used is an upper bound on the clique size in P.
        color_list = []  # List of lists, color_list[k] contains vertices with color k
        color_masks = []  # Bitmask of vertices in each color class for fast conflict checking
        vertex_color = {}
        
        for v in P:
            assigned = False
            v_bit = 1 << (v - 1)
            # Try to assign the lowest possible color
            for k in range(len(color_list)):
                # Check if v is compatible with any vertex in color_list[k]
                # If v is adjacent (compatible) with any u in this color class, conflict
                if (compat[v] & color_masks[k]) == 0:
                    color_list[k].append(v)
                    color_masks[k] |= v_bit
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_list.append([v])
                color_masks.append(v_bit)
                vertex_color[v] = len(color_list) - 1
        
        max_color = len(color_list)
        
        if depth + max_color <= len(best_clique):
            return
        
        # Build a list of (vertex, color) sorted by color descending, then by original order
        # This helps us process higher colors first
        vertex_order = []
        for k in range(max_color - 1, -1, -1):
            for v in color_list[k]:
                vertex_order.append((v, k))
        
        # Branching
        # Process vertices in order of decreasing color
        for v, k in vertex_order:
            # Check if v is still in P_mask
            v_bit = 1 << (v - 1)
            if not (P_mask & v_bit):
                continue
            
            # Upper bound check: if we can't beat best_clique even with this vertex, skip
            if depth + 1 + (P_mask & compat[v]).bit_count() <= len(best_clique):
                # Exclude v
                P_mask &= ~v_bit
                continue
            
            # Include v
            C.append(v)
            
            # New P: neighbors of v in P_mask \ {v}
            new_P_mask = P_mask & compat[v] & ~v_bit
            
            expand(new_P_mask, C, depth + 1)
            
            C.pop()
            
            # Exclude v
            P_mask &= ~v_bit
            
            if time.time() > DEADLINE:
                return

    initial_P_mask = (1 << N) - 1
    # Note: Bit 0 is node 1. Bit N-1 is node N.
    # (1 << N) - 1 has bits 0..N-1 set.
    # This corresponds to nodes 1..N.
    
    expand(initial_P_mask, [], 0)
    
    if not best_clique:
        return [1]
        
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
