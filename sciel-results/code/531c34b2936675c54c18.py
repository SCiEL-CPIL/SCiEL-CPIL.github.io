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
                
                # Incremental M2 update: O(N) instead of O(N^3)
                # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                # Changing M[c_old][u,v] and M[c_old][v,u] affects:
                #   M2[c_old][i,j] for all i,j via terms involving k=u or k=v
                # Specifically:
                #   M2[c][i,j] = M[c][i,u]*M[c][u,j] + M[c][i,v]*M[c][v,j] + other terms
                # When M[c_old][u,v] changes from 1 to 0:
                #   For all i != u,v: M2[c_old][i,u] -= M[c_old][i,v]
                #                     M2[c_old][u,i] -= M[c_old][v,i]
                #                     M2[c_old][i,v] -= M[c_old][i,u]
                #                     M2[c_old][v,i] -= M[c_old][u,i]
                #   M2[c_old][u,v] -= M[c_old][u,u] + M[c_old][v,v] = 0 (diagonal is 0)
                #   Actually need to be more careful.
                
                # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                # The edge (u,v) in color c_old means M[c_old][u,v]=1 and M[c_old][v,u]=1
                # After removing: M[c_old][u,v]=0, M[c_old][v,u]=0
                # 
                # M2[c_old][i,j] = sum_k M[c_old][i,k] * M[c_old][k,j]
                # The change: M[c_old][u,v] goes 1->0, M[c_old][v,u] goes 1->0
                # 
                # For term k=u: M[c_old][i,u] * M[c_old][u,j]
                #   This term existed before for all i,j. After change, M[c_old][u,j] still same
                #   Wait no, M[c_old][u,v] changed, so when j=v, M[c_old][u,v] changed.
                #   Actually M[c_old][u,j] only changes when j=v.
                #   
                # Let me think differently. The matrix M[c_old] changes in exactly two entries:
                #   M[c_old][u,v]: 1 -> 0
                #   M[c_old][v,u]: 1 -> 0
                # 
                # M2 = M @ M. The (i,j) entry of M2 is sum_k M[i,k]*M[k,j].
                # 
                # The change in M2[i,j] = sum_k delta_M[i,k]*M[k,j] + M[i,k]*delta_M[k,j]
                # where delta_M has -1 at (u,v) and (v,u).
                # 
                # delta_M2[i,j] = -M[i,v]*M[u,j] - M[i,u]*M[v,j] + M[i,v]*M[u,j] + M[i,u]*M[v,j]
                # Wait, that's not right either. Let me be precise.
                # 
                # delta_M[i,k] = -1 if (i,k)=(u,v) or (i,k)=(v,u), else 0
                # 
                # delta_M2[i,j] = sum_k (delta_M[i,k]*M[k,j] + M[i,k]*delta_M[k,j])
                # = delta_M[i,v]*M[v,j] + delta_M[i,u]*M[u,j] + M[i,v]*delta_M[v,j] + M[i,u]*delta_M[u,j]
                # 
                # delta_M[i,v] = -1 if i=u, else 0 (since (u,v) is the only entry with col=v that changes... wait no)
                # delta_M[i,k] = -1 if (i,k) = (u,v) OR (i,k) = (v,u)
                # So delta_M[i,v] = -1 if i=u (from (u,v)), and delta_M[i,u] = -1 if i=v (from (v,u))
                # Similarly delta_M[v,j] = -1 if j=u (from (v,u)), and delta_M[u,j] = -1 if j=v (from (u,v))
                # 
                # delta_M2[i,j] = delta_M[i,v]*M[v,j] + delta_M[i,u]*M[u,j] + M[i,v]*delta_M[v,j] + M[i,u]*delta_M[u,j]
                # 
                # For i=u, j=v: delta_M[u,v]*M[v,v] + delta_M[u,u]*M[u,v] + M[u,v]*delta_M[v,v] + M[u,u]*delta_M[u,v]
                #   = (-1)*0 + 0*M[u,v] + M[u,v]*0 + 0*(-1) = 0  [before change M[u,v]=1]
                # Hmm, this is getting complicated. Let me just do the O(N) update properly.
                
                # Simpler approach: recompute M2 for affected colors but only for rows/cols u and v
                # Actually, the cleanest O(N) update:
                # 
                # When we remove edge (u,v) from color c:
                #   For all i not in {u,v}:
                #     M2[c][i,u] -= M[c][i,v]   (because M[c][u,v] was 1, now 0, so term k=u: M[i,u]*M[u,v] disappears... wait)
                #     
                # Let me just use the formula:
                # M2_new[i,j] = M2_old[i,j] - M[i,v]*M[u,j] - M[i,u]*M[v,j] + M[i,v]*M[u,j] + M[i,u]*M[v,j]
                # No wait, that cancels. I'm overcomplicating this.
                
                # Let me just recompute M2 for the two colors using numpy matmul.
                # For N=50, that's 2 * 50^3 = 250000 ops, which is fast.
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                # Update bad_edges
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    # Plan:
    # 1. Try N=50 with multiple restarts using seed-based initialization
    # 2. Try N=48, 46 as fallbacks
    
    seed_32 = build_seed_32()
    
    # Strategy 1: Direct SA on N=50 with seed-based init
    # Use the 32-vertex seed as base, extend randomly, then SA
    time_per_attempt = 18.0
    num_attempts = 5  # 5 * 18 = 90s
    
    for i in range(num_attempts):
        if time.time() > DEADLINE - 25:
            break
        
        # Create seed-based initialization for N=50
        N = 50
        m = 32
        init_colors = [0] * (N * (N - 1) // 2)
        # Copy seed colors for edges among first 32 vertices
        idx_seed = 0
        for a in range(m):
            for b in range(a + 1, m):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                init_colors[idx] = seed_32[idx_seed]
                idx_seed += 1
        # Random colors for remaining edges
        for a in range(N):
            for b in range(a + 1, N):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                if a >= m or b >= m:
                    init_colors[idx] = random.randint(0, 3)
        
        res = solve_limited(N, time_per_attempt, seed_colors=init_colors)
        if res:
            return {'n': N, 'colors': res}
    
    # Strategy 2: Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 15:
            break
        N = 48
        m = 32
        init_colors = [0] * (N * (N - 1) // 2)
        idx_seed = 0
        for a in range(m):
            for b in range(a + 1, m):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                init_colors[idx] = seed_32[idx_seed]
                idx_seed += 1
        for a in range(N):
            for b in range(a + 1, N):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                if a >= m or b >= m:
                    init_colors[idx] = random.randint(0, 3)
        res = solve_limited(N, 12.0, seed_colors=init_colors)
        if res:
            return {'n': N, 'colors': res}
    
    # Strategy 3: Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 8:
            break
        N = 46
        m = 32
        init_colors = [0] * (N * (N - 1) // 2)
        idx_seed = 0
        for a in range(m):
            for b in range(a + 1, m):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                init_colors[idx] = seed_32[idx_seed]
                idx_seed += 1
        for a in range(N):
            for b in range(a + 1, N):
                idx = a * N - a * (a + 1) // 2 + (b - a - 1)
                if a >= m or b >= m:
                    init_colors[idx] = random.randint(0, 3)
        res = solve_limited(N, 8.0, seed_colors=init_colors)
        if res:
            return {'n': N, 'colors': res}
    
    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
