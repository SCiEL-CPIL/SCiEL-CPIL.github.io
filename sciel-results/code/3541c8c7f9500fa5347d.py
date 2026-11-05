import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    def solve(N, time_limit):
        t_start = time.time()
        t_end = t_start + time_limit

        # Initialize colors randomly
        colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        # Build M matrices
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        # Also keep a 2D array C for O(1) color lookup
        C = np.zeros((N, N), dtype=np.int8)
        
        # Precompute edge index mapping
        edge_to_idx = {}
        idx_to_edge = {}
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                edge_to_idx[(i, j)] = idx
                idx_to_edge[idx] = (i, j)
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1

        # Helper to get index of edge (i, j) in flattened array
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return edge_to_idx[(i, j)]

        # Helper to check if an edge is bad (has a triangle in its color class)
        def is_bad(i, j):
            c = C[i][j]
            # Count common neighbors in color c
            k = np.dot(M[c][i], M[c][:, j])
            return k > 0

        # Initialize bad set with edge indices
        bad_set = set()
        for i in range(N):
            for j in range(i + 1, N):
                if is_bad(i, j):
                    bad_set.add(edge_idx(i, j))

        if not bad_set:
            return colors.tolist()

        current_T = 5.0
        cooling = 0.999
        min_T = 0.01

        while time.time() < t_end:
            if not bad_set:
                return colors.tolist()

            # Pick random bad edge
            bad_idx = random.choice(list(bad_set))
            u, v = idx_to_edge[bad_idx]
            c_old = C[u][v]
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                # Compute k_old and k_new
                k_old = np.dot(M[c_old][u], M[c_old][:, v])
                k_new = np.dot(M[c_new][u], M[c_new][:, v])
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
                idx_u_v = edge_idx(u, v)
                colors[idx_u_v] = best_c
                
                # Update M
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                # Update bad_set
                # Remove all edges incident to u or v from bad_set
                # Then re-evaluate them
                to_recheck = set()
                for w in range(N):
                    if w == u or w == v:
                        continue
                    # Edge (u, w)
                    i, j = (u, w) if u < w else (w, u)
                    idx_e = edge_idx(i, j)
                    if idx_e in bad_set:
                        bad_set.remove(idx_e)
                    to_recheck.add(idx_e)
                    
                    # Edge (v, w)
                    i, j = (v, w) if v < w else (w, v)
                    idx_e = edge_idx(i, j)
                    if idx_e in bad_set:
                        bad_set.remove(idx_e)
                    to_recheck.add(idx_e)
                
                # Remove (u, v) itself
                if idx_u_v in bad_set:
                    bad_set.remove(idx_u_v)
                to_recheck.add(idx_u_v)
                
                # Re-evaluate to_recheck
                for idx_e in to_recheck:
                    i, j = idx_to_edge[idx_e]
                    if is_bad(i, j):
                        bad_set.add(idx_e)
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    def build_seed_32():
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
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    seed_32 = build_seed_32()

    # Try N=50 with multiple restarts
    for i in range(5):
        if time.time() > DEADLINE - 10:
            break
        res = solve(50, 20.0)
        if res:
            return {'n': 50, 'colors': res}
            
    # Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 10:
            break
        res = solve(48, 15.0)
        if res:
            return {'n': 48, 'colors': res}
            
    # Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 10:
            break
        res = solve(46, 10.0)
        if res:
            return {'n': 46, 'colors': res}
            
    # Fallback
    return {'n': 32, 'colors': seed_32}
