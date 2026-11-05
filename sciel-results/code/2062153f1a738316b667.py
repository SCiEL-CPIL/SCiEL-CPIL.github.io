import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # Safety margin (150s kill, keep 32s buffer)

    # --- Helper: GF(16) multiplication for the Clebsch graph ---
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

    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def solve_limited(N, time_limit, seed_colors=None):
        t_start = time.time()
        t_end = t_start + time_limit
        
        # Initialize state
        if seed_colors and len(seed_colors) == N * (N - 1) // 2:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        # Build M matrices (4 NxN matrices) and C (color lookup matrix)
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

        # Compute M2 = M @ M for each color
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
            return colors.tolist()

        current_T = 5.0
        cooling = 0.998
        min_T = 0.01
        
        while time.time() < t_end:
            if not bad_edges:
                return colors.tolist()
            
            # Pick a random bad edge
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
            
            # Acceptance
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                # Update C
                C[u][v] = best_c
                C[v][u] = best_c
                # Update colors array (flattened)
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c
                
                # Update M matrices
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                # Efficient M2 update: only recompute rows/cols affected
                # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                # When we change edge (u,v) from c_old to best_c:
                # For c_old: M2[c_old] loses contributions involving edge (u,v)
                # For best_c: M2[best_c] gains contributions involving edge (u,v)
                
                # For c_old: M2[c_old][i,j] changes for all (i,j) where
                # i or j is u or v, because M[c_old][u,v] changed
                # Actually, M2[c][i,j] = sum_k M[c][i,k]*M[c][k,j]
                # The change in M[c_old][u,v] affects M2[c_old][i,j] for all i,j
                # where the sum includes k=u with M[c_old][i,u]*M[c_old][u,j] 
                # or k=v with M[c_old][i,v]*M[c_old][v,j]
                
                # Simpler: just recompute M2 for the two affected colors
                # But that's O(N^3) which is slow for N=50
                # N=50: 50^3 = 125000 ops per color, 2 colors = 250000 ops
                # At ~100M ops/sec in numpy, that's ~2.5ms per update
                # With ~1000 updates in 20s, that's 2.5s total. Acceptable.
                
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                # Update bad_edges
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    def extend_seed_to_N(seed_colors, N):
        """Extend a valid coloring of K_m to K_N by adding vertices with random colors,
        then repair."""
        m = len(seed_colors)
        if m >= N:
            return seed_colors[:N*(N-1)//2]
        
        # Start with seed and random colors for new edges
        new_colors = seed_colors[:]
        # Edges among new vertices and between old and new
        # Old vertices: 0..m-1, new vertices: m..N-1
        # Edges to add: (i,j) where i>=m or j>=m, i<j
        
        # Generate random colors for new edges
        for i in range(m, N):
            for j in range(i+1, N):
                new_colors.append(random.randint(0, 3))
            for j in range(i):
                # Edge (j, i) where j < i, j < m
                # But we need to add in lexicographic order
                # Actually, let's rebuild properly
                pass
        
        # Rebuild: create full color array for K_N
        full_colors = [0] * (N*(N-1)//2)
        idx_seed = 0
        idx_new = 0
        
        # First, copy seed colors for edges among old vertices
        for i in range(m):
            for j in range(i+1, m):
                full_colors[idx_seed] = seed_colors[idx_seed]
                idx_seed += 1
        
        # Now fill in edges involving new vertices
        # We need to be careful about lexicographic order
        # Let's just use random for all new edges
        for i in range(N):
            for j in range(i+1, N):
                # Compute index
                idx = i * N - i*(i+1)//2 + (j - i - 1)
                if i < m and j < m:
                    # Already filled
                    continue
                else:
                    full_colors[idx] = random.randint(0, 3)
        
        return full_colors

    # Plan:
    # 1. Try N=50 with multiple restarts
    # 2. Try extending seed 32 to larger N
    # 3. Fallback to seed 32
    
    seed_32 = build_seed_32()
    
    # Strategy 1: Direct SA on N=50
    # Use time-efficient restarts
    time_per_attempt = 15.0
    num_attempts = 6  # 6 * 15 = 90s
    
    for i in range(num_attempts):
        if time.time() > DEADLINE - 25:
            break
        res = solve_limited(50, time_per_attempt)
        if res:
            return {'n': 50, 'colors': res}
    
    # Strategy 2: Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 15:
            break
        res = solve_limited(48, 12.0)
        if res:
            return {'n': 48, 'colors': res}
    
    # Strategy 3: Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 8:
            break
        res = solve_limited(46, 8.0)
        if res:
            return {'n': 46, 'colors': res}
    
    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
