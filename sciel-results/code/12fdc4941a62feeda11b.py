import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a maximum cardinality subset of [1, 256] such that for all distinct a, b in the subset,
    (a * b) % (a + b) != 0.
    Uses Tomita Max Clique algorithm with improved coloring bound and vectorized setup.
    """
    N = 256
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit

    # 1. Build the graph of "good" pairs - vectorized with numpy
    import numpy as np
    
    nums = np.arange(1, N + 1)
    # Compute all pair products and sums
    # For each pair (i,j) with i<j, check if (i*j) % (i+j) != 0
    # Build adjacency masks
    
    good_adj = [0] * (N + 1)
    
    # Vectorized construction: for each u, compute which v are "good"
    for u in range(1, N + 1):
        # v ranges from 1 to N, v != u
        v_arr = np.arange(1, N + 1)
        mask_arr = (v_arr != u) & ((u * v_arr) % (u + v_arr) != 0)
        # Convert to bitmask
        # For each bit position j (0-indexed), if mask_arr[j] is True, set bit j
        # v_arr[j] = j+1, so bit position = j = v-1
        mask = 0
        for j in range(N):
            if mask_arr[j]:
                mask |= (1 << j)
        good_adj[u] = mask

    # Precompute degree for each vertex
    degrees = [good_adj[i].bit_count() for i in range(1, N + 1)]

    best_clique_mask = 0
    best_size = 0

    # Multiple greedy restarts to get a better initial lower bound
    full_mask = (1 << N) - 1
    
    # Greedy 1: max degree ordering
    P = full_mask
    R = 0
    while P:
        max_deg = -1
        best_v = -1
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            deg = (good_adj[v + 1] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v)
    
    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    # Greedy 2: min degree ordering (sometimes finds different cliques)
    P = full_mask
    R = 0
    while P:
        min_deg = N + 1
        best_v = -1
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            deg = (good_adj[v + 1] & P).bit_count()
            if deg < min_deg:
                min_deg = deg
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v)
    
    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    # Greedy 3: random-ish ordering using hash of vertex index
    P = full_mask
    R = 0
    while P:
        best_v = -1
        best_score = -1
        p_temp = P
        while p_temp:
            lsb = p_temp & -p_temp
            v = lsb.bit_length() - 1
            # Use a pseudo-random score based on vertex and current P
            score = (good_adj[v + 1] & P).bit_count() * 31 + ((v * 2654435761) & 0xFFFF)
            if score > best_score:
                best_score = score
                best_v = v
            p_temp ^= lsb
        if best_v == -1:
            break
        R |= (1 << best_v)
        P &= good_adj[best_v + 1]
        P &= ~(1 << best_v)
    
    if R.bit_count() > best_size:
        best_size = R.bit_count()
        best_clique_mask = R

    def expand(P, R):
        nonlocal best_size, best_clique_mask
        
        if time.time() > deadline:
            return

        if not P:
            if R.bit_count() > best_size:
                best_size = R.bit_count()
                best_clique_mask = R
            return

        # Improved greedy coloring with degree-sorted ordering
        # Use a more efficient approach: sort vertices by degree in P, then greedy color
        verts_in_P = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            verts_in_P.append(v)
            temp ^= lsb
        
        # Sort by degree in P descending for tighter coloring
        verts_in_P.sort(key=lambda v: (good_adj[v + 1] & P).bit_count(), reverse=True)
        
        # Assign colors greedily
        color_of = [-1] * N
        next_color = 0
        for v in verts_in_P:
            # Find the smallest color not used by neighbors of v in P
            neighbor_colors = set()
            adj_mask = good_adj[v + 1] & P
            temp2 = adj_mask
            while temp2:
                lsb2 = temp2 & -temp2
                u = lsb2.bit_length() - 1
                if color_of[u] >= 0:
                    neighbor_colors.add(color_of[u])
                temp2 ^= lsb2
            
            # Find smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            next_color = max(next_color, c + 1)
        
        num_colors = next_color
        
        if R.bit_count() + num_colors <= best_size:
            return

        # Branch on vertices in reverse color order (highest color first)
        # Group vertices by color
        color_groups = [[] for _ in range(num_colors)]
        for v in verts_in_P:
            color_groups[color_of[v]].append(v)
        
        ordered_vertices = []
        for i in range(num_colors - 1, -1, -1):
            ordered_vertices.extend(color_groups[i])
        
        k = num_colors
        for v in ordered_vertices:
            if time.time() > deadline:
                return
            
            if not (P & (1 << v)):
                continue
                
            new_R = R | (1 << v)
            new_P = P & good_adj[v + 1]
            
            expand(new_P, new_R)
            
            P &= ~(1 << v)
            k -= 1
            
            if R.bit_count() + k <= best_size:
                break

    # Start search
    P = full_mask
    R = 0
    expand(P, R)

    result = []
    for i in range(N):
        if best_clique_mask & (1 << i):
            result.append(i + 1)
    
    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
