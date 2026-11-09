import time
import random
import numpy as np

def construct():
    """
    Finds a 4-coloring of K_n with no monochromatic triangles, maximizing n.
    """
    deadline = time.time() + 118.0  # Safety margin for 120s limit
    
    # Fallback: The Greenwood-Gleason coloring for n=32
    def get_seed_solution():
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
        return {'n': 32, 'colors': colors}

    seed_sol = get_seed_solution()
    best_n = 32
    best_colors = seed_sol['colors']
    
    # Store the seed coloring in a 2D array for initialization
    N_seed = 32
    seed_c = [[0] * N_seed for _ in range(N_seed)]
    idx = 0
    for i in range(N_seed):
        for j in range(i + 1, N_seed):
            seed_c[i][j] = best_colors[idx]
            seed_c[j][i] = best_colors[idx]
            idx += 1

    # Local Search Solver for a given N
    def try_solve(N, time_limit, init_c=None):
        if N < 4:
            return None
        
        # Initialize coloring
        c = [[0] * N for _ in range(N)]
        
        if init_c and N >= N_seed:
            # Copy seed
            for i in range(N_seed):
                for j in range(i + 1, N_seed):
                    c[i][j] = init_c[i][j]
            # Random for rest
            for i in range(N):
                for j in range(i + 1, N):
                    if i < N_seed and j < N_seed:
                        continue # Already set
                    c[i][j] = random.randint(0, 3)
        else:
            # All random
            for i in range(N):
                for j in range(i + 1, N):
                    c[i][j] = random.randint(0, 3)
        
        # Precompute A using numpy for speed
        A = np.zeros((4, N, N), dtype=np.int32)
        
        # Build adjacency matrices for each color
        for col in range(4):
            M = np.zeros((N, N), dtype=np.int32)
            for u in range(N):
                for w in range(N):
                    if u == w:
                        continue
                    if u < w:
                        if c[u][w] == col:
                            M[u, w] = 1
                    else:
                        if c[w][u] == col:
                            M[u, w] = 1
            
            # A[col][u][v] = number of common neighbors w where both (u,w) and (v,w) have color col
            # This is the (u,v) entry of M @ M.T, minus diagonal if u==v
            AM = M @ M.T
            np.fill_diagonal(AM, 0)
            A[col] = AM
        
        total_bad = int(np.sum(A[:, :, :].reshape(4, -1)))
        
        if total_bad == 0:
            flat = []
            for i in range(N):
                for j in range(i + 1, N):
                    flat.append(c[i][j])
            return flat

        start_time = time.time()
        no_improve_count = 0
        best_total_bad = total_bad
        
        while time.time() < start_time + time_limit:
            # Find best move
            best_delta = 1000000
            best_u, best_v, best_col = -1, -1, -1
            
            # Vectorized search for best move
            for u in range(N):
                for v in range(u + 1, N):
                    old_col = c[u][v]
                    for new_col in range(4):
                        if new_col == old_col:
                            continue
                        delta = int(A[new_col, u, v]) - int(A[old_col, u, v])
                        if delta < best_delta:
                            best_delta = delta
                            best_u, best_v, best_col = u, v, new_col
            
            if best_u == -1:
                break

            if best_delta > 0:
                # Local minimum - try random perturbation
                if random.random() < 0.7:  # Increased probability
                    u = random.randint(0, N - 1)
                    v = random.randint(0, N - 1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    old_col = c[u][v]
                    new_col = random.choice([0, 1, 2, 3])
                    if new_col == old_col:
                        continue
                    best_delta = int(A[new_col, u, v]) - int(A[old_col, u, v])
                    best_col = new_col
                    best_u, best_v = u, v
                else:
                    # Try a few random moves to escape
                    for _ in range(3):
                        u = random.randint(0, N - 1)
                        v = random.randint(0, N - 1)
                        if u == v:
                            continue
                        if u > v:
                            u, v = v, u
                        old_col = c[u][v]
                        new_col = random.choice([0, 1, 2, 3])
                        if new_col == old_col:
                            continue
                        delta = int(A[new_col, u, v]) - int(A[old_col, u, v])
                        if delta < best_delta:
                            best_delta = delta
                            best_u, best_v, best_col = u, v, new_col
                    if best_u == -1:
                        continue

            # Apply Move
            old_col = c[best_u][best_v]
            new_col = best_col
            c[best_u][best_v] = new_col
            c[best_v][best_u] = new_col
            
            u, v = best_u, best_v
            
            # Update A efficiently
            for w in range(N):
                if w == u or w == v:
                    continue
                
                col_vw = c[v][w] if v < w else c[w][v]
                col_uw = c[u][w] if u < w else c[w][u]
                
                if col_vw == old_col:
                    A[old_col, u, w] -= 1
                    A[old_col, w, u] -= 1
                if col_vw == new_col:
                    A[new_col, u, w] += 1
                    A[new_col, w, u] += 1
                    
                if col_uw == old_col:
                    A[old_col, v, w] -= 1
                    A[old_col, w, v] -= 1
                if col_uw == new_col:
                    A[new_col, v, w] += 1
                    A[new_col, w, v] += 1
            
            total_bad += best_delta
            
            if total_bad < best_total_bad:
                best_total_bad = total_bad
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # If stuck for too long, restart with different initialization
            if no_improve_count > 5000:
                break
            
            if total_bad == 0:
                flat = []
                for i in range(N):
                    for j in range(i + 1, N):
                        flat.append(c[i][j])
                return flat

        return None

    # Try to find solutions for N > 32
    # Focus on 50 first with significant time
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    # Time allocation: prioritize n=50 heavily
    for N in targets:
        if time.time() > deadline - 3.0:
            break
        
        if N == 50:
            t_limit = 50.0
        elif N == 49:
            t_limit = 15.0
        elif N == 48:
            t_limit = 10.0
        elif N >= 46:
            t_limit = 5.0
        else:
            t_limit = 3.0
            
        if time.time() + t_limit > deadline:
            t_limit = deadline - time.time() - 1.0
            if t_limit < 1.0:
                break
            
        # Try with seed initialization for the first 32 vertices
        sol = try_solve(N, t_limit, seed_c)
        if sol is not None:
            return {'n': N, 'colors': sol}
        
        # If it failed, try again with pure random initialization for larger N
        if N >= 48 and time.time() < deadline - 5.0:
            t_limit2 = min(10.0, deadline - time.time() - 2.0)
            if t_limit2 > 1.0:
                sol = try_solve(N, t_limit2, None)
                if sol is not None:
                    return {'n': N, 'colors': sol}

    # If no better solution found, return seed
    return seed_sol
