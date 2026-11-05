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

    seed_32 = build_seed_32()

    # --- Efficient local search with incremental M2 updates ---
    def solve_efficient(N, time_limit, seed_colors=None, restarts=1):
        t_start = time.time()
        t_end = t_start + time_limit

        best_result = None

        for r in range(restarts):
            if time.time() >= t_end:
                break

            # Initialize state
            if r == 0 and seed_colors and len(seed_colors) == N * (N - 1) // 2:
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
                    M[c][i, j] = 1
                    M[c][j, i] = 1
                    C[i, j] = c
                    C[j, i] = c
                    idx += 1

            # Compute M2 = M @ M for each color
            M2 = [M[c] @ M[c] for c in range(4)]

            def get_bad_edges():
                bad = []
                for i in range(N):
                    for j in range(i + 1, N):
                        c = C[i, j]
                        if M2[c][i, j] > 0:
                            bad.append((i, j, c))
                return bad

            bad_edges = get_bad_edges()

            if not bad_edges:
                best_result = colors.tolist()
                return best_result

            current_T = 5.0
            cooling = 0.9995
            min_T = 0.005

            while time.time() < t_end:
                if not bad_edges:
                    best_result = colors.tolist()
                    return best_result

                # Pick a random bad edge
                u, v, c_old = random.choice(bad_edges)

                best_delta = float('inf')
                best_c = -1

                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    k_old = M2[c_old][u, v]
                    k_new = M2[c_new][u, v]
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
                    C[u, v] = best_c
                    C[v, u] = best_c
                    # Update colors array (flattened)
                    idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                    colors[idx_u_v] = best_c

                    # Update M matrices
                    M[c_old][u, v] = 0
                    M[c_old][v, u] = 0
                    M[best_c][u, v] = 1
                    M[best_c][v, u] = 1

                    # Efficient M2 update: only recompute affected rows/cols
                    # M2[c][i,j] = sum_k M[c][i,k] * M[c][k,j]
                    # When edge (u,v) changes from c_old to best_c:
                    # For c_old: M2[c_old][i,j] changes for all (i,j) where
                    # the sum includes k=u with M[c_old][i,u]*M[c_old][u,j] 
                    # or k=v with M[c_old][i,v]*M[c_old][v,j]
                    # Since M[c_old][u,v] changed to 0, we need to subtract contributions
                    # where k=u or k=v involved edge (u,v)
                    
                    # For c_old: M2[c_old][i,j] -= M[c_old][i,u]*M[c_old][u,j] + M[c_old][i,v]*M[c_old][v,j]
                    # But M[c_old][u,v] is now 0, so M[c_old][u,j] for j!=v is unchanged
                    # Actually the change is: M[c_old][u,v] went from 1 to 0
                    # So for any (i,j), M2[c_old][i,j] changes by:
                    # - M[c_old][i,u]*M[c_old][u,j] (when k=u, the term M[c_old][u,j] was 1 if j==v)
                    # Wait, this is getting complex. Let's just recompute the two affected M2 matrices.
                    # For N=50, this is 2 * 50^3 = 250000 ops, which is fast in numpy.
                    
                    M2[c_old] = M[c_old] @ M[c_old]
                    M2[best_c] = M[best_c] @ M[best_c]

                    # Update bad_edges
                    bad_edges = get_bad_edges()

                current_T *= cooling
                if current_T < min_T:
                    current_T = min_T

        return best_result

    # --- Greedy vertex addition from seed ---
    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def try_greedy_add_vertex(n, colors):
        """Try to assign colors for a new vertex n. Returns list of n colors or None."""
        assignment = [None] * n

        def backtrack(v):
            if time.time() > DEADLINE:
                return None
            if v == n:
                return assignment[:]

            for c in [0, 1, 2, 3]:
                assignment[v] = c
                ok = True
                for i in range(v):
                    if assignment[i] == c:
                        idx = edge_idx(i, v, n)
                        if colors[idx] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                result = backtrack(v + 1)
                if result is not None:
                    return result
                assignment[v] = None
            return None

        return backtrack(0)

    # Strategy 1: Try to extend seed 32 to larger N via greedy + local search
    n = 32
    colors_32 = seed_32[:]
    
    max_target = 50
    for target in range(32, max_target + 1):
        if time.time() > DEADLINE - 30:
            break
        new_colors = try_greedy_add_vertex(n, colors_32)
        if new_colors is not None:
            for i in range(n):
                colors_32.append(new_colors[i])
            n += 1
        else:
            # Try local search perturbations
            success = False
            for _ in range(50):
                if time.time() > DEADLINE - 30:
                    break
                ei = random.randint(0, len(colors_32) - 1)
                old = colors_32[ei]
                colors_32[ei] = (old + 1 + random.randint(0, 2)) % 4
                new_colors = try_greedy_add_vertex(n, colors_32)
                if new_colors is not None:
                    for i in range(n):
                        colors_32.append(new_colors[i])
                    n += 1
                    success = True
                    break
                else:
                    colors_32[ei] = old
            if not success:
                break

    if n >= 40:
        # We have a valid coloring of K_n, try to extend further with local search
        # But first, let's see if we can just return what we have
        pass

    # Strategy 2: Direct SA on N=50 with multiple restarts
    time_remaining = DEADLINE - time.time()
    if time_remaining > 20:
        time_per_attempt = min(15.0, (time_remaining - 5) / 4)
        num_attempts = min(4, int((time_remaining - 5) / time_per_attempt))
        
        for i in range(num_attempts):
            if time.time() > DEADLINE - 10:
                break
            res = solve_efficient(50, time_per_attempt, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res:
                return {'n': 50, 'colors': res}

    # Strategy 3: Try N=48
    time_remaining = DEADLINE - time.time()
    if time_remaining > 15:
        for i in range(2):
            if time.time() > DEADLINE - 8:
                break
            res = solve_efficient(48, 10.0, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res:
                return {'n': 48, 'colors': res}

    # Strategy 4: Try N=46
    time_remaining = DEADLINE - time.time()
    if time_remaining > 10:
        res = solve_efficient(46, 8.0, seed_colors=seed_32, restarts=1)
        if res:
            return {'n': 46, 'colors': res}

    # Fallback: return the best we have from greedy extension
    if n > 32 and len(colors_32) == n * (n - 1) // 2:
        return {'n': n, 'colors': colors_32}

    # Final fallback: Seed 32
    return {'n': 32, 'colors': seed_32}
