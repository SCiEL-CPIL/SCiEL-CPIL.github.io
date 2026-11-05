import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()

    def edge_idx(i, j, n):
        # Index for edge (i, j) in lexicographic order for vertices 0..n-1
        # Edges are (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Number of edges before row i is i * (n - 1) - i * (i - 1) // 2
        # Position in row i is j - i - 1
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    # 1. Deterministic Construction for n=26 (Lazebnik-Woldar)
    # Vertices are pairs (x, y) in F_13 x F_13
    # Colors are determined by slope (y2-y1)/(x2-x1) mapped to {0, 1, 3, 4, 9}
    # These are the quadratic residues mod 13 plus 0.
    
    n_lw = 26
    num_edges_lw = n_lw * (n_lw - 1) // 2
    colors_lw = [0] * num_edges_lw
    
    # Map vertex index to (x, y)
    def get_coords(idx):
        return idx % 13, idx // 13
        
    # Map slope to color index
    # Quadratic residues mod 13: 1^2=1, 2^2=4, 3^2=9, 4^2=16=3, 5^2=25=12, 6^2=36=10
    # Wait, 1, 4, 9, 3, 12, 10.
    # The set {0, 1, 3, 4, 9} is not the set of all quadratic residues.
    # Let's check the property.
    # Lazebnik-Woldar construction for R_5(K_{2,2}) uses q=13.
    # The colors correspond to the quadratic residues of F_q.
    # For q=13, QRs are {1, 3, 4, 9, 10, 12}. There are 6 QRs.
    # We need 5 colors.
    # Actually, the construction for R_5(C4) with n=26 uses a specific subset or a different mapping.
    # Let's look at the standard result: R_5(C4) >= 26.
    # The construction is based on the affine plane AG(2, 13).
    # A valid coloring can be obtained by partitioning the slopes into 5 sets such that no 4-cycle is formed.
    # A 4-cycle in a color class corresponds to two vertices having two common neighbors in that color.
    # In the affine plane, if we color edges by slope, a color class is a set of parallel lines (or lines of a specific slope).
    # Two vertices u, v have a common neighbor w in color c if the slope (u,w) is in C and (v,w) is in C.
    # If C is a set of slopes, this forms a C4 if there are two distinct w1, w2 such that slopes (u,w1), (v,w1), (u,w2), (v,w2) are all in C.
    # This implies that the quadrilateral u-w1-v-w2 is a parallelogram in the affine plane if the slopes match appropriately?
    # Actually, if we use all QRs as colors, we have 6 colors. We need 5.
    # It turns out that for q=13, we can merge two colors or use a subset.
    # However, it is known that n=26 is possible.
    # Let's try using the 5 smallest QRs? No, that might not work.
    # Let's use the standard construction:
    # Colors are assigned based on the difference vector.
    # For (u,v), let d = v - u. If d=(0,0) skip.
    # If d_x != 0, slope m = d_y * d_x^{-1}.
    # If d_x == 0, slope is infinity.
    # We need to map slopes to 5 colors.
    # The set of slopes has size 14 (13 finite + 1 infinity).
    # We can partition the 14 slopes into 5 sets such that each set is "sum-free" or similar?
    # Actually, the condition for no C4 is that the set of slopes assigned to a color does not contain a "harmonic" structure?
    # A simpler approach: The known solution for n=26 uses the fact that 13 is a prime.
    # Let's hardcode the colors for n=26 by generating them via a verified method.
    # One such method:
    # Map the 14 slopes to 5 colors.
    # Let QRs = {1, 3, 4, 9, 10, 12}.
    # Let NQRs = {2, 5, 6, 7, 8, 11}.
    # Infinity = 13.
    # If we use 5 colors, we must merge some.
    # It is known that merging QRs and NQRs carefully works.
    # Specifically, for R_5(C4), we can use the following coloring:
    # Color 0: Slope 0 (horizontal)
    # Color 1: Slope infinity (vertical)
    # Colors 2-4: Partition the 12 non-zero finite slopes into 3 sets of 4?
    # Or partition the 13 non-zero slopes (including infinity) into 4 sets?
    
    # Let's try a different known construction for n=26.
    # It is often constructed using the field F_13.
    # Let's assume the standard "algebraic" coloring works with a specific mapping.
    # Since I cannot derive the exact partition of 14 slopes into 5 C4-free sets from memory with 100% certainty,
    # I will use a robust heuristic search for n=26 as well, but seeded with a good structure,
    # OR I will use the fact that n=26 is small enough that a fast SA finds it instantly.
    
    # However, the prompt asks for a "direct implementation" if possible.
    # Let's try the mapping:
    # Slopes: 0, inf, 1, 2, ..., 12.
    # Let's group them:
    # C0: {0}
    # C1: {inf}
    # C2: {1, 3, 4, 9} (QRs excluding some?)
    # This is risky.
    
    # Alternative: Use the known fact that a 5-coloring of K_26 exists.
    # I will implement a very fast SA for n=26 to guarantee finding it, 
    # and then use the remaining time for n=27.
    # This is safer than a potentially buggy hardcoded algebraic construction.
    # But wait, the "Inspiration" says "Direct implementation ... guarantees ... instantly".
    # If I can't be 100% sure of the algebraic mapping, I should use a high-performance search for 26.
    # n=26 is very small. SA finds it in milliseconds.
    
    # Let's stick to the high-performance SA for 26 and 27.
    # The current program's weakness is that it might not find 26 quickly enough or gets stuck.
    # My improvement: 
    # 1. Optimize the SA loop (precomputed masks, efficient bit ops).
    # 2. Use a better initialization (greedy).
    # 3. Focus 80% of time on n=26 to ensure we get the base score, 
    #    and 20% on n=27 for a potential jump.
    
    best_n = 1
    best_colors = [0]
    
    # Precompute edge lists for n=26 and n=27
    edge_lists = {}
    for n in [26, 27]:
        el = []
        for i in range(n):
            for j in range(i + 1, n):
                el.append((i, j, edge_idx(i, j, n)))
        edge_lists[n] = el
        
    # Function to run SA for a given n
    def run_sa(n, time_budget):
        num_edges = n * (n - 1) // 2
        edge_list = edge_lists[n]
        
        # Greedy Initialization
        colors = [0] * num_edges
        adj = [[0] * n for _ in range(5)]
        
        # Random order
        edge_indices = list(range(num_edges))
        random.shuffle(edge_indices)
        
        for idx in edge_indices:
            u, v, e_idx = edge_list[idx]
            best_c = 0
            min_common = float('inf')
            
            for c in range(5):
                common = adj[c][u] & adj[c][v]
                cnt = common.bit_count()
                if cnt < min_common:
                    min_common = cnt
                    best_c = c
            
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
        
        energy = compute_energy(adj, n)
        
        if energy == 0:
            return colors
        
        best_energy = energy
        best_colors_local = colors[:]
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.01
        start_time = time.time()
        stuck_count = 0
        max_stuck = 5000
        
        while time.time() - start_time < time_budget:
            if temp < min_temp:
                break
            if stuck_count > max_stuck:
                # Restore best
                if best_energy < energy:
                    for i in range(num_edges):
                        colors[i] = best_colors_local[i]
                    for c in range(5):
                        for i in range(n):
                            adj[c][i] = 0
                    for i in range(num_edges):
                        u, v, _ = edge_list[i]
                        c = colors[i]
                        adj[c][u] |= (1 << v)
                        adj[c][v] |= (1 << u)
                    energy = best_energy
                    temp = 1.0
                    stuck_count = 0
                else:
                    break
            
            ei = random.randint(0, num_edges - 1)
            u, v, e_idx = edge_list[ei]
            c_old = colors[e_idx]
            c_new = random.randint(0, 4)
            if c_new == c_old:
                continue
            
            delta = 0
            
            # Remove from c_old
            mask_u = adj[c_old][u]
            mask_v = adj[c_old][v]
            # Iterate over neighbors of u in c_old
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m -= lsb
                if w == v:
                    continue
                if adj[c_old][w] & (1 << v):
                    k = (adj[c_old][u] & adj[c_old][w]).bit_count()
                    delta -= (k - 1)
            
            # Iterate over neighbors of v in c_old
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m -= lsb
                if w == u:
                    continue
                if adj[c_old][w] & (1 << u):
                    k = (adj[c_old][v] & adj[c_old][w]).bit_count()
                    delta -= (k - 1)
            
            # Add to c_new
            mask_u = adj[c_new][u]
            mask_v = adj[c_new][v]
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m -= lsb
                if w == v:
                    continue
                if adj[c_new][w] & (1 << v):
                    k = (adj[c_new][u] & adj[c_new][w]).bit_count()
                    delta += k
            
            m = mask_v
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m -= lsb
                if w == u:
                    continue
                if adj[c_new][w] & (1 << u):
                    k = (adj[c_new][v] & adj[c_new][w]).bit_count()
                    delta += k
            
            new_energy = energy + delta
            
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                energy = new_energy
                adj[c_old][u] &= ~(1 << v)
                adj[c_old][v] &= ~(1 << u)
                adj[c_new][u] |= (1 << v)
                adj[c_new][v] |= (1 << u)
                colors[e_idx] = c_new
                
                if energy < best_energy:
                    best_energy = energy
                    best_colors_local = colors[:]
                    stuck_count = 0
                    if energy == 0:
                        return colors
            else:
                stuck_count += 1
            
            temp *= cooling
        
        # Return best found even if not 0, but we only care about 0
        if best_energy == 0:
            return best_colors_local
        return None

    # Try n=27 first (shorter time) then n=26 (longer time) or vice versa?
    # n=26 is guaranteed to be solvable. n=27 is unknown.
    # If we spend too much time on 27, we might fail to find 26 if the search is bad.
    # But SA is very fast.
    # Let's allocate 30s to n=27 and 80s to n=26.
    
    # Attempt n=27
    if time_left() > 35:
        t_27 = min(30, time_left() - 5)
        res_27 = run_sa(27, t_27)
        if res_27:
            best_n = 27
            best_colors = res_27
            return {'n': best_n, 'colors': best_colors}
    
    # Attempt n=26
    if time_left() > 5:
        t_26 = time_left() - 2
        res_26 = run_sa(26, t_26)
        if res_26:
            best_n = 26
            best_colors = res_26
            return {'n': best_n, 'colors': best_colors}
            
    # Fallback: try smaller n if 26 fails (unlikely)
    for n in range(25, 10, -1):
        if time_left() < 5:
            break
        t_n = min(10, time_left() - 2)
        res = run_sa(n, t_n)
        if res:
            best_n = n
            best_colors = res
            return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
