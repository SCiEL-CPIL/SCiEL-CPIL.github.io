import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    deadline = start_time + 87.0  # Safety margin before 90s kill
    
    # 1. Build Compatibility Graph
    # Two numbers a, b are compatible if lcm(a,b) <= limit.
    # We want the Maximum Clique in this Compatibility Graph.
    
    # neighbors[i] is bitmask of j such that lcm(i+1, j+1) <= limit
    neighbors = [0] * n
    
    for i in range(n):
        num_i = i + 1
        mask = 0
        for j in range(n):
            if i == j:
                continue
            num_j = j + 1
            g = gcd(num_i, num_j)
            lcm_val = (num_i * num_j) // g
            if lcm_val <= limit:
                mask |= (1 << j)
        neighbors[i] = mask
        
    # Degrees
    degrees = [bin(neighbors[i]).count('1') for i in range(n)]
    
    # 2. Maximum Clique via Branch and Bound with greedy coloring
    best_size = 0
    best_mask = 0
    
    # Greedy coloring: assign colors 1..k to vertices in P such that adjacent vertices get different colors.
    # Returns (max_color, color_dict) where color_dict[v] = color of v.
    # Vertices are processed in order of decreasing degree within P for better coloring quality.
    
    def greedy_color(P_mask):
        # Extract vertices from P_mask
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        
        # Sort by degree in P (descending) - use global degree as proxy for speed
        # For small P, compute exact degree in P
        if len(verts) <= 32:
            deg_in_P = {}
            for v in verts:
                deg_in_P[v] = bin(neighbors[v] & P_mask).count('1')
            verts.sort(key=lambda x: -deg_in_P[x])
        else:
            verts.sort(key=lambda x: -degrees[x])
        
        colors = {}
        max_c = 0
        
        for v in verts:
            # Find min color not used by already-colored neighbors in P
            used_colors = set()
            nb_mask = neighbors[v] & P_mask
            m_nb = nb_mask
            while m_nb:
                lsb = m_nb & -m_nb
                u = lsb.bit_length() - 1
                m_nb ^= lsb
                if u in colors:
                    used_colors.add(colors[u])
                    
            # Find min available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_c:
                max_c = c
        
        return max_c, colors, verts

    # Branch and Bound
    def expand(R_mask, P_mask):
        nonlocal best_size, best_mask
        
        if time.time() > deadline:
            return
        
        r_size = bin(R_mask).count('1')
        
        if P_mask == 0:
            if r_size > best_size:
                best_size = r_size
                best_mask = R_mask
            return
        
        # Get coloring bound
        max_c, colors, verts = greedy_color(P_mask)
        
        # Prune if current clique size + max possible extension <= best
        if r_size + max_c <= best_size:
            return
        
        # Process vertices in reverse color order (highest color first) for better pruning
        # Vertices with higher colors are less likely to be pruned, so try them first
        verts_sorted = sorted(verts, key=lambda x: -colors[x])
        
        for v in verts_sorted:
            if time.time() > deadline:
                return
            
            # Pruning: if r_size + color[v] <= best_size, skip
            if r_size + colors[v] <= best_size:
                continue
            
            # Include v in clique
            new_R = R_mask | (1 << v)
            
            # New candidate set: vertices in P compatible with v
            new_P = P_mask & neighbors[v]
            
            expand(new_R, new_P)
            
            # Remove v from P for subsequent iterations
            P_mask &= ~(1 << v)
            
            # Update r_size is constant within this loop, but P shrinks
            # Re-check if we can prune remaining vertices
            # The coloring was for the original P, so it's an upper bound still valid
            
    # Initial candidate set is all nodes
    initial_P = (1 << n) - 1
    
    # First, get a good initial solution using greedy to improve pruning
    R_greedy = 0
    P_greedy = initial_P
    order = sorted(range(n), key=lambda x: -degrees[x])
    for v in order:
        if P_greedy & (1 << v):
            if (neighbors[v] & R_greedy) == R_greedy:
                R_greedy |= (1 << v)
    
    best_size = bin(R_greedy).count('1')
    best_mask = R_greedy
    
    # Now run B&B
    expand(0, initial_P)
    
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return sorted(result)

construct()
