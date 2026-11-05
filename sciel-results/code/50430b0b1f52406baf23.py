import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0

    def build_seed_32():
        """Build the Clebsch graph 4-coloring for n=32."""
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)

        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def count_monochromatic_triangles(colors, N):
        """Count total monochromatic triangles in a 4-coloring of K_N."""
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                idx += 1
        total = 0
        for c in range(4):
            M2 = M[c] @ M[c]
            total += np.sum(M2)
        return total

    def block_construction(N, time_limit):
        """
        Block construction: partition N vertices into 4 groups.
        Inter-group edges use a fixed 3-coloring of K_4 (no mono triangles).
        Intra-group edges use 3 colors (0,1,2) optimized via local search.
        Group g uses color g for intra-group edges' "base", but we use colors 0,1,2 
        within each group and color 3 is only for inter-group.
        
        Actually, let's use: 4 groups A,B,C,D.
        Inter-group: edge between group i and group j gets color f(i,j) where f is a 
        triangle-free 3-coloring of K_4.
        Intra-group: edges within group i get colors from {0,1,2} (not color i to avoid 
        confusion, or we can use all 4 but that's complex).
        
        Simpler: Use colors 0,1,2 for intra-group, color 3 for inter-group.
        But then inter-group all color 3 -> triangles in K_4 with 3 colors.
        
        Better: Inter-group uses 3 colors in a triangle-free pattern on 4 groups.
        Intra-group uses the 4th color plus... no, let's think differently.
        
        Standard approach: 4 groups. Edge between group i and j (i!=j) gets color c_{ij}.
        We need {c_{ij}} to be a proper 3-coloring of K_4 (no mono triangle in the group graph).
        K_4 with 3 colors: one color appears on 3 edges (a triangle), which is bad.
        Actually K_4 chromatic number is 4, so we can't 3-color K_4 without mono triangles.
        
        So we need a different structure. Let's use 2 groups instead.
        """
        # Use 2 groups of size n/2.
        # Inter-group: 4 colors distributed.
        # Intra-group: 3 colors (0,1,2) with no mono triangles.
        # This is like the seed_32 construction!
        
        # For N=50: two groups of 25.
        # Intra-group: 3-coloring of K_25 with no mono triangles.
        # Inter-group: 4-coloring of K_{25,25} with no mono triangles.
        
        # This is complex. Let's try a different structural approach.
        
        # Approach: Start from seed_32 and try to extend to N=50 by adding 18 vertices.
        # Use local search to fix conflicts.
        
        pass  # Placeholder - will implement in the main loop

    def local_search_extend(N, seed_colors, seed_size, time_limit):
        """
        Extend a valid coloring of K_{seed_size} to K_N by adding N-seed_size vertices.
        Use local search to minimize monochromatic triangles.
        """
        if seed_size >= N:
            return seed_colors[:N*(N-1)//2]
        
        # Initialize: seed colors + random for new edges
        colors = [0] * (N*(N-1)//2)
        
        # Copy seed colors
        idx = 0
        for i in range(seed_size):
            for j in range(i+1, seed_size):
                colors[idx] = seed_colors[idx]
                idx += 1
        
        # Random for new edges
        for i in range(N):
            for j in range(i+1, N):
                idx = i * N - i*(i+1)//2 + (j - i - 1)
                if i >= seed_size or j >= seed_size:
                    colors[idx] = random.randint(0, 3)
        
        # Build matrices
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1
        
        M2 = [M[c] @ M[c] for c in range(4)]
        
        def get_bad_edges():
            bad = []
            for i in range(N):
                for j in range(i + 1, N):
                    c = C[i][j]
                    if M2[c][i][j] > 0:
                        bad.append((i, j, c))
            return bad
        
        bad_edges = get_bad_edges()
        
        if not bad_edges:
            return colors
        
        t_end = time.time() + time_limit
        current_T = 2.0
        cooling = 0.999
        min_T = 0.001
        
        while time.time() < t_end:
            if not bad_edges:
                return colors
            
            u, v, c_old = random.choice(bad_edges)
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                k_old = M2[c_old][u][v]
                k_new = M2[c_new][u][v]
                delta = k_new - k_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                C[u][v] = best_c
                C[v][u] = best_c
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c
                
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T
        
        return None

    def direct_search(N, time_limit):
        """Direct local search from random initialization."""
        colors = [random.randint(0, 3) for _ in range(N*(N-1)//2)]
        
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1
        
        M2 = [M[c] @ M[c] for c in range(4)]
        
        def get_bad_edges():
            bad = []
            for i in range(N):
                for j in range(i + 1, N):
                    c = C[i][j]
                    if M2[c][i][j] > 0:
                        bad.append((i, j, c))
            return bad
        
        bad_edges = get_bad_edges()
        
        if not bad_edges:
            return colors
        
        t_end = time.time() + time_limit
        current_T = 3.0
        cooling = 0.998
        min_T = 0.01
        
        while time.time() < t_end:
            if not bad_edges:
                return colors
            
            u, v, c_old = random.choice(bad_edges)
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                k_old = M2[c_old][u][v]
                k_new = M2[c_new][u][v]
                delta = k_new - k_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                C[u][v] = best_c
                C[v][u] = best_c
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c
                
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T
        
        return None

    # Strategy: Try multiple approaches in sequence
    seed_32 = build_seed_32()
    
    # 1. Try extending seed_32 to N=50
    if time.time() < DEADLINE - 30:
        res = local_search_extend(50, seed_32, 32, 25.0)
        if res and count_monochromatic_triangles(res, 50) == 0:
            return {'n': 50, 'colors': res}
    
    # 2. Try direct search on N=50
    if time.time() < DEADLINE - 20:
        for attempt in range(3):
            if time.time() > DEADLINE - 20:
                break
            res = direct_search(50, 6.0)
            if res and count_monochromatic_triangles(res, 50) == 0:
                return {'n': 50, 'colors': res}
    
    # 3. Try N=48
    if time.time() < DEADLINE - 15:
        res = local_search_extend(48, seed_32, 32, 10.0)
        if res and count_monochromatic_triangles(res, 48) == 0:
            return {'n': 48, 'colors': res}
    
    # 4. Try N=46
    if time.time() < DEADLINE - 10:
        res = local_search_extend(46, seed_32, 32, 8.0)
        if res and count_monochromatic_triangles(res, 46) == 0:
            return {'n': 46, 'colors': res}
    
    # 5. Try N=44
    if time.time() < DEADLINE - 5:
        res = local_search_extend(44, seed_32, 32, 5.0)
        if res and count_monochromatic_triangles(res, 44) == 0:
            return {'n': 44, 'colors': res}
    
    # Fallback
    return {'n': 32, 'colors': seed_32}
