import time
import sys
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    limit = 512
    n = limit
    deadline = start_time + 85.0  # Safety margin before 90s kill

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
    
    # 3. Local Search: 1-for-2 and 2-for-3 moves to improve clique
    def try_add(v, R_mask):
        if (neighbors[v] & R_mask) == R_mask:
            return R_mask | (1 << v)
        return None

    def local_search(R_mask, P_mask):
        nonlocal best_size, best_mask
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            
            # 1-for-2 move: remove one vertex, add two
            verts_in_R = []
            m = R_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                verts_in_R.append(idx)
                m ^= lsb
            
            for v_out in verts_in_R:
                if time.time() > deadline:
                    break
                new_R = R_mask & ~(1 << v_out)
                # Find candidates to add
                # Candidates must be in P (or just not in R? In MIS/MC context, P is usually the set of vertices compatible with R)
                # Here P is passed as compatible vertices. But for 1-for-2, we can consider any vertex not in R.
                # Let's consider all vertices not in R.
                candidates = [v for v in range(n) if not (R_mask & (1 << v))]
                
                # Sort candidates by degree in new_R? Or just iterate?
                # To be efficient, we can check pairs of candidates that are compatible with each other and with new_R.
                # This is O(N^2) per removal. 512^2 * 512 is too slow.
                # We need a smarter way.
                # Let's just try adding vertices one by one from the current P (compatible with R).
                # But 1-for-2 implies the new vertices might not have been in P originally?
                # Actually, if v_out is removed, the set of compatible vertices expands.
                # Let's compute new_P = {v | v compatible with new_R}
                new_P_mask = 0
                for v in range(n):
                    if not (new_R & (1 << v)):
                        if (neighbors[v] & new_R) == new_R:
                            new_P_mask |= (1 << v)
                
                # Now try to add two vertices from new_P_mask that are compatible with each other
                # We can just run a quick greedy or check pairs?
                # Given time constraints, let's stick to 1-for-1 (add) and maybe simple 2-for-1 (remove two, add one? No, cardinality decreases).
                # The inspiration says "1-for-2 and 2-for-3".
                # 1-for-2: Remove 1, Add 2. Net gain +1.
                # 2-for-3: Remove 2, Add 3. Net gain +1.
                
                # Efficient 1-for-2:
                # Pick a vertex u from new_P_mask.
                # If we add u, can we add another v from new_P_mask & neighbors[u]?
                # We want to maximize gain.
                # Let's try to find a clique of size 2 in new_P_mask that can be added.
                # This is equivalent to finding an edge in the subgraph induced by new_P_mask.
                # If we find an edge (u, v), we can add both.
                
                # Optimization: Just pick the highest degree vertex in new_P_mask, add it, then pick highest degree in neighbors of that vertex in new_P_mask.
                # This is a greedy 2-clique.
                
                if new_P_mask:
                    # Greedy pick 2
                    # Find vertex with max degree in new_P_mask
                    best_u = -1
                    max_deg = -1
                    m_np = new_P_mask
                    while m_np:
                        lsb = m_np & -m_np
                        u = lsb.bit_length() - 1
                        m_np ^= lsb
                        deg = bin(neighbors[u] & new_P_mask).count('1')
                        if deg > max_deg:
                            max_deg = deg
                            best_u = u
                    
                    if best_u != -1:
                        # Add best_u
                        R_temp = new_R | (1 << best_u)
                        P_temp = new_P_mask & neighbors[best_u] & ~(1 << best_u)
                        
                        # Find best_v in P_temp
                        best_v = -1
                        max_deg_v = -1
                        m_pt = P_temp
                        while m_pt:
                            lsb = m_pt & -m_pt
                            v = lsb.bit_length() - 1
                            m_pt ^= lsb
                            deg = bin(neighbors[v] & P_temp).count('1')
                            if deg > max_deg_v:
                                max_deg_v = deg
                                best_v = v
                            
                        if best_v != -1:
                            new_R_candidate = R_temp | (1 << best_v)
                            # Size check
                            if bin(new_R_candidate).count('1') > bin(R_mask).count('1'):
                                R_mask = new_R_candidate
                                improved = True
                                if bin(R_mask).count('1') > best_size:
                                    best_size = bin(R_mask).count('1')
                                    best_mask = R_mask
                                break # Restart loop with new R_mask
            
            # 1-for-1 move: Just add if possible (handled by B&B usually, but for local search on a fixed set)
            # Actually, the B&B already does this. The local search is to escape local optima of the greedy seed or B&B pruning?
            # If B&B is exact, it finds the global optimum. But if it times out, we have a partial solution.
            # The local search should operate on the current best_mask or a candidate.
            # Let's apply local search to the initial greedy solution to improve the seed for B&B.
            
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
    
    # Apply local search to improve the greedy seed
    # We need a P_mask for local search. Let's use all vertices compatible with R_greedy.
    P_for_ls = 0
    for v in range(n):
        if not (R_greedy & (1 << v)):
            if (neighbors[v] & R_greedy) == R_greedy:
                P_for_ls |= (1 << v)
    
    R_ls = R_greedy
    P_ls = P_for_ls
    local_search(R_ls, P_ls)
    
    if bin(R_ls).count('1') > best_size:
        best_size = bin(R_ls).count('1')
        best_mask = R_ls
    
    # Now run B&B with the improved initial bound
    expand(0, initial_P)
    
    # If B&B didn't find anything better, keep the local search result
    # (expand updates best_size/best_mask if it finds something better)
    
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
