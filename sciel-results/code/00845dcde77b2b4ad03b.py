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
        
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
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
            return i * N - i * (i + 1) // 2 + (j - i - 1)

        # Helper to check if an edge is bad
        def is_bad(i, j):
            c = C[i][j]
            if M[c][i][j] == 0:
                return False
            # Count common neighbors in color c
            # dot(M[c][i], M[c][:, j])
            # Since M[c][i][j] is 1, the dot product includes k=j?
            # M[c][i][j] * M[c][j][j] (which is 0) + ...
            # Actually, M[c][i] is row i. M[c][:, j] is col j.
            # The sum is sum_k M[c][i][k] M[c][k][j].
            # If k=j, M[c][i][j] * M[c][j][j] = 1 * 0 = 0.
            # So it correctly counts triangles (i, j, k) for k != i, j.
            k = np.dot(M[c][i], M[c][:, j])
            return k > 0

        # Initialize bad set
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
            # Reverse map index to (i, j)
            # This is slow. We can store (i, j) in the set instead of idx.
            # But we need idx to update `colors` array.
            # Let's store (i, j) in bad_set for faster selection, and compute idx when needed.
            # Actually, storing (i, j) tuples in a set is fine.
            
            # Let's refactor: bad_set contains (i, j) tuples.
            # Re-initialize bad_set with tuples.
            # (This block is for the first iteration, but we are in the loop)
            # We need to change the initialization above to use tuples.
            # I will rewrite the loop logic to use tuples.
            
            # For now, let's assume we have (u, v) from bad_set.
            # Since I initialized with idx, I need to map back.
            # Mapping back is O(1) with formula?
            # idx = i * N - i(i+1)/2 + (j-i-1)
            # It's a quadratic. We can precompute a map idx -> (i, j).
            
            # Let's precompute idx_map
            idx_map = {}
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    idx_map[idx] = (i, j)
                    idx += 1
            
            u, v = idx_map[bad_idx]
            c_old = C[u][v]
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                # Compute k_old and k_new
                # k_old = dot(M[c_old][u], M[c_old][:, v])
                # k_new = dot(M[c_new][u], M[c_new][:, v])
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
                    i, j = idx_map[idx_e]
                    if is_bad(i, j):
                        bad_set.add(idx_e)
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    # Strategy:
    # 1. Try N=50 with multiple restarts.
    # 2. If fail, return N=32 seed.
    
    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                # Clebsch graph construction
                # Split into two sets of 16?
                # The seed code used:
                # log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                # This implies vertices 0-15 and 16-31.
                # If i, j in same block, color based on XOR.
                # If different, color 3.
                
                # We need the GF(16) log table.
                # Let's reuse the logic from the original seed.
                pass
        # Re-implementing GF(16) log is tedious.
        # Let's just use a known valid 32-vertex coloring or random?
        # Random 32-vertex is likely to have triangles.
        # The Clebsch construction is specific.
        
        # Let's try to find a 32-vertex solution quickly or just return 50 if found.
        # If 50 fails, we need a valid 32.
        
        # I will hardcode a simple check or use the original code's logic.
        # Actually, I can just copy the `build_seed_32` logic from the prompt.
        
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

        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    seed_32 = build_seed_32()

    # Try N=50
    # 5 restarts, 20s each
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
