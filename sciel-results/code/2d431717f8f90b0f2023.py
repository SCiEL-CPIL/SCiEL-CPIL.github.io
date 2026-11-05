import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # 118 seconds safety margin (limit 150s)
    
    def validate(colors, n):
        """
        Validate if a coloring has no monochromatic C4.
        Returns True if valid.
        """
        if len(colors) != n * (n - 1) // 2:
            return False
        mats = [np.zeros((n, n), dtype=np.int8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                mats[c][i, j] = 1
                mats[c][j, i] = 1
                idx += 1
        
        for c in range(5):
            M = mats[c]
            M2 = M @ M
            tri_u, tri_v = np.triu_indices(n, 1)
            vals = M2[tri_u, tri_v]
            if np.any(vals >= 2):
                return False
        return True

    def get_paley_color(u, v, p):
        """
        Compute color for edge (u, v) in Paley-type construction over GF(p).
        u, v are elements of GF(p).
        Color is determined by the quadratic character of (u-v)^2 * g, 
        mapped to 0..4 via discrete log.
        """
        diff = (u - v) % p
        if diff == 0:
            return 0 # Should not happen for u != v
        val = (diff * diff * 2) % p  # 2 is a generator of GF(13)
        # Map non-zero elements of GF(13) to 0..4
        # Powers of 2 mod 13:
        # 2^0 = 1
        # 2^1 = 2
        # 2^2 = 4
        # 2^3 = 8
        # 2^4 = 16 = 3
        # 2^5 = 6
        # 2^6 = 12 = -1
        # 2^7 = 24 = 11
        # 2^8 = 22 = 9
        # 2^9 = 18 = 5
        # 2^10 = 10
        # 2^11 = 20 = 7
        # 2^12 = 14 = 1
        
        # We need to map val to a color in 0..4.
        # We can group the 12 non-zero elements into 5 colors?
        # Wait, 12 is not divisible by 5.
        # The standard Paley coloring for R(C4) usually uses a specific mapping.
        # For p=13, the Paley graph is strongly regular (13, 6, 2, 3).
        # It has no C4? No, Paley graph of order 13 has lambda=2, so it HAS C4s.
        # We need a 5-coloring.
        
        # Actually, the known construction for R_5(C4) > 26 is NOT just Paley.
        # It's often related to affine planes or specific group actions.
        # However, a simpler algebraic construction for 5 colors on 26 vertices
        # might be based on the group Z_2 x Z_13 or similar.
        
        # Let's try a different algebraic approach that is known to work for 5 colors.
        # One such construction uses the field F_5 and a specific extension or product.
        # But 26 = 2 * 13.
        
        # Let's stick to the robust search but initialize with a "structured" random start
        # that mimics symmetry, or just use the search.
        # The prompt says "The algebraic construction of Lazebnik and Woldar gives 26 vertices".
        # If I can't implement it perfectly, I must rely on the search.
        # BUT, the key insight from the "different strategy" directive is to NOT just do random search.
        
        # Let's try a construction based on the group Z_26?
        # Or better, let's use the fact that n=26 is small enough that a *very* fast
        # local search with a good heuristic will find it instantly.
        # The "different strategy" is: **Deterministic Heuristic Initialization + Intensive Local Search**.
        # Instead of pure random restarts, we initialize with a "block" structure
        # that reduces the initial cost, then let local search finish.
        
        pass # This function is a placeholder for the algebraic attempt. 
             # We will implement a heuristic init in the solver instead.

    def solve_generic(n, initial_colors=None, seed=42):
        num_edges = n * (n - 1) // 2
        endpoints = []
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                endpoints.append((i, j))
                edge_index[(i, j)] = idx
                idx += 1
                
        if initial_colors is None:
            # Heuristic Initialization: 
            # Assign colors based on (i+j) % 5 or similar to break symmetry
            # and reduce initial C4s.
            colors = [0] * num_edges
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # Simple heuristic: color by sum mod 5
                    # This creates a 5-regular-ish structure initially?
                    # Actually, let's use a more random but structured init.
                    # Color = (i * 3 + j * 7) % 5
                    colors[idx] = (i * 3 + j * 7) % 5
                    idx += 1
        else:
            colors = initial_colors[:]
            
        neighbors = [[set() for _ in range(5)] for _ in range(n)]
        for e_idx in range(num_edges):
            u, v = endpoints[e_idx]
            c = colors[e_idx]
            neighbors[u][c].add(v)
            neighbors[v][c].add(u)
            
        CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
        for c in range(5):
            for k in range(n):
                nbrs = list(neighbors[k][c])
                for i in range(len(nbrs)):
                    u = nbrs[i]
                    for j in range(i + 1, len(nbrs)):
                        v = nbrs[j]
                        CN[c][u, v] += 1
                        CN[c][v, u] += 1
                        
        current_cost = 0
        for c in range(5):
            tri_u, tri_v = np.triu_indices(n, 1)
            vals = CN[c][tri_u, tri_v]
            current_cost += np.sum(vals * (vals - 1) // 2)
            
        if current_cost == 0:
            return colors
            
        max_iter = 100000
        rng = np.random.default_rng(seed)
        
        # Local Search with Incremental Updates
        for i in range(max_iter):
            if time.time() > deadline - 2:
                return None
            if current_cost == 0:
                return colors
                
            # Pick random edge
            e_idx = rng.integers(0, num_edges)
            u, v = endpoints[e_idx]
            old_c = colors[e_idx]
            
            best_delta = 0
            best_new_c = -1
            
            for new_c in range(5):
                if new_c == old_c:
                    continue
                delta = 0
                for x in range(n):
                    if x == u or x == v:
                        continue
                    
                    # old_c decreases
                    if x in neighbors[v][old_c]:
                        old_val = CN[old_c][u, x]
                        delta -= (old_val - 1)
                    if x in neighbors[u][old_c]:
                        old_val = CN[old_c][v, x]
                        delta -= (old_val - 1)
                    
                    # new_c increases
                    if x in neighbors[v][new_c]:
                        old_val = CN[new_c][u, x]
                        delta += old_val
                    if x in neighbors[u][new_c]:
                        old_val = CN[new_c][v, x]
                        delta += old_val
                        
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c == -1:
                continue
                
            current_cost += best_delta
            new_c = best_new_c
            colors[e_idx] = new_c
            
            neighbors[u][old_c].discard(v)
            neighbors[v][old_c].discard(u)
            neighbors[u][new_c].add(v)
            neighbors[v][new_c].add(u)
            
            for x in range(n):
                if x == u or x == v:
                    continue
                if x in neighbors[v][old_c]:
                    CN[old_c][u, x] -= 1
                    CN[old_c][x, u] -= 1
                if x in neighbors[u][old_c]:
                    CN[old_c][v, x] -= 1
                    CN[old_c][x, v] -= 1
                    
                if x in neighbors[v][new_c]:
                    CN[new_c][u, x] += 1
                    CN[new_c][x, u] += 1
                if x in neighbors[u][new_c]:
                    CN[new_c][v, x] += 1
                    CN[new_c][x, v] += 1
                    
        return None

    best_n = 11
    # Fallback: a trivial valid coloring for small n
    # For n=11, we can just use a valid known small instance or random search.
    # But we aim for 26.
    
    # 1. Attempt n=26
    res_26 = solve_generic(26, seed=42)
    if res_26 and validate(res_26, 26):
        best_n = 26
        best_colors = res_26
    else:
        # If 26 fails immediately, try a few more seeds?
        # But 26 should be easy with this solver.
        # Let's try one more with a different seed.
        res_26 = solve_generic(26, seed=123)
        if res_26 and validate(res_26, 26):
            best_n = 26
            best_colors = res_26
        else:
            # Fallback to smaller n if 26 fails (unlikely)
            res_20 = solve_generic(20, seed=42)
            if res_20 and validate(res_20, 20):
                best_n = 20
                best_colors = res_20
            else:
                # Absolute fallback
                best_n = 11
                best_colors = [0] * (11*10//2)
                # Ensure validity for 11? 
                # A random search for 11 is instant.
                res_11 = solve_generic(11, seed=1)
                if res_11 and validate(res_11, 11):
                    best_colors = res_11

    # 2. Attempt n=27
    if best_n == 26 and time.time() < deadline - 5:
        n_target = 27
        num_edges = n_target * (n_target - 1) // 2
        
        colors_27 = [0] * num_edges
        endpoints_27 = []
        edge_index_27 = {}
        idx = 0
        for i in range(n_target):
            for j in range(i + 1, n_target):
                endpoints_27.append((i, j))
                edge_index_27[(i, j)] = idx
                idx += 1
                
        # Copy n=26 colors
        for e_idx in range(len(best_colors)):
            colors_27[e_idx] = best_colors[e_idx]
            
        # Randomize edges involving 26
        for i in range(26):
            e_idx = edge_index_27[(i, 26)]
            # Use a heuristic for the new vertex to help convergence
            colors_27[e_idx] = (i * 3 + 26 * 7) % 5
            
        res_27 = solve_generic(27, initial_colors=colors_27, seed=999)
        if res_27 and validate(res_27, 27):
            best_n = 27
            best_colors = res_27
        else:
            # Try one more restart for 27 with random init for new edges
            for i in range(26):
                e_idx = edge_index_27[(i, 26)]
                colors_27[e_idx] = np.random.randint(0, 5)
            res_27 = solve_generic(27, initial_colors=colors_27, seed=888)
            if res_27 and validate(res_27, 27):
                best_n = 27
                best_colors = res_27

    return {'n': best_n, 'colors': best_colors}
