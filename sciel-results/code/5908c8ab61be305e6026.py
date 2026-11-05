import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 118.0  # Safety margin (150s kill, keep 32s buffer)

    def edge_idx(i, j, n):
        """Flat index for edge (i,j) in lexicographic order of K_n."""
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def build_M_and_C(colors, n):
        """Build M (4 NxN adjacency matrices) and C (color matrix) from flat colors array."""
        M = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
        C = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                M[c][i, j] = 1
                M[c][j, i] = 1
                C[i, j] = c
                C[j, i] = c
                idx += 1
        return M, C

    def compute_M2(M):
        return [M[c] @ M[c] for c in range(4)]

    def count_bad(C, M2):
        """Count monochromatic triangles."""
        total = 0
        for i in range(C.shape[0]):
            for j in range(i + 1, C.shape[0]):
                c = C[i, j]
                if M2[c][i, j] > 0:
                    total += M2[c][i, j]
        return total

    def get_bad_edges(C, M2):
        bad = []
        for i in range(C.shape[0]):
            for j in range(i + 1, C.shape[0]):
                c = C[i, j]
                if M2[c][i, j] > 0:
                    bad.append((i, j, c))
        return bad

    # --- Build a good seed: Clebsch-graph-based 32-vertex coloring ---
    def build_seed_32():
        # GF(16) with polynomial x^4 + x + 1
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

    # --- Local search with incremental updates ---
    def local_search(N, time_limit, seed_colors=None, restarts=1):
        t_start = time.time()
        t_end = t_start + time_limit
        best_result = None
        best_bad = float('inf')

        for r in range(restarts):
            if time.time() >= t_end:
                break

            if r == 0 and seed_colors and len(seed_colors) == N * (N - 1) // 2:
                colors = np.array(seed_colors, dtype=np.int8).copy()
            else:
                colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

            M, C = build_M_and_C(colors, N)
            M2 = compute_M2(M)
            bad_count = count_bad(C, M2)

            if bad_count == 0:
                return colors.tolist()

            if bad_count < best_bad:
                best_bad = bad_count
                best_result = colors.tolist()

            current_T = 3.0
            cooling = 0.9998
            min_T = 0.01
            no_improve_steps = 0
            max_no_improve = 5000

            while time.time() < t_end:
                if bad_count == 0:
                    return colors.tolist()

                if no_improve_steps > max_no_improve:
                    # Restart with perturbation
                    for _ in range(20):
                        ei = random.randint(0, len(colors) - 1)
                        colors[ei] = random.randint(0, 3)
                    M, C = build_M_and_C(colors, N)
                    M2 = compute_M2(M)
                    bad_count = count_bad(C, M2)
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_result = colors.tolist()
                    no_improve_steps = 0
                    current_T = 3.0
                    continue

                # Pick a random bad edge
                bad_edges = get_bad_edges(C, M2)
                if not bad_edges:
                    return colors.tolist()

                u, v, c_old = random.choice(bad_edges)

                # Try all 3 alternative colors, pick best
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
                    no_improve_steps = 0
                else:
                    p = np.exp(-best_delta / max(current_T, 1e-10))
                    if random.random() < p:
                        accept = True
                    no_improve_steps += 1

                if accept:
                    # Update C
                    C[u, v] = best_c
                    C[v, u] = best_c
                    idx_u_v = edge_idx(u, v, N)
                    colors[idx_u_v] = best_c

                    # Update M matrices
                    M[c_old][u, v] = 0
                    M[c_old][v, u] = 0
                    M[best_c][u, v] = 1
                    M[best_c][v, u] = 1

                    # Recompute M2 for affected colors (fast for N<=50)
                    M2[c_old] = M[c_old] @ M[c_old]
                    M2[best_c] = M[best_c] @ M[best_c]

                    bad_count = count_bad(C, M2)
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_result = colors.tolist()

                current_T *= cooling
                if current_T < min_T:
                    current_T = min_T

        return best_result

    # --- Try to extend a valid coloring by one vertex ---
    def extend_by_one(n, colors):
        """Try to add vertex n to a valid coloring of K_n. Returns new flat colors or None."""
        new_n = n + 1
        new_colors = colors[:]

        # For each existing vertex i, we need to choose color for edge (i, new_n)
        # such that no monochromatic triangle is formed.
        # A triangle (i, j, new_n) is monochromatic in color c if:
        #   colors[edge(i,j)] == c AND color(i,new_n) == c AND color(j,new_n) == c

        # This is a constraint satisfaction: for each color c, the set of vertices
        # connected to new_n by color c must be an independent set in the c-colored subgraph.

        assignment = [None] * n

        def backtrack(v):
            if time.time() > DEADLINE:
                return False
            if v == n:
                return True

            for c in [0, 1, 2, 3]:
                assignment[v] = c
                ok = True
                for i in range(v):
                    if assignment[i] == c:
                        # Check if edge (i, v) in original coloring is color c
                        idx = edge_idx(i, v, n)
                        if colors[idx] == c:
                            ok = False
                            break
                if not ok:
                    assignment[v] = None
                    continue
                if backtrack(v + 1):
                    return True
                assignment[v] = None
            return False

        if backtrack(0):
            for i in range(n):
                new_colors.append(assignment[i])
            return new_colors
        return None

    # --- Strategy 1: Extend seed 32 greedily, then local search to fix ---
    n = 32
    colors_cur = seed_32[:]

    for target in range(33, 51):
        if time.time() > DEADLINE - 25:
            break
        extended = extend_by_one(n, colors_cur)
        if extended is not None:
            colors_cur = extended
            n += 1
        else:
            # Try a few random perturbations of existing edges to enable extension
            success = False
            for _ in range(100):
                if time.time() > DEADLINE - 25:
                    break
                ei = random.randint(0, len(colors_cur) - 1)
                old = colors_cur[ei]
                colors_cur[ei] = (old + 1 + random.randint(0, 2)) % 4
                extended = extend_by_one(n, colors_cur)
                if extended is not None:
                    colors_cur = extended
                    n += 1
                    success = True
                    break
                else:
                    colors_cur[ei] = old
            if not success:
                break

    # If we got a valid coloring of size n, try to extend further with local search
    if n >= 40 and time.time() < DEADLINE - 20:
        # Run local search on the current n-vertex coloring to see if it's valid
        M, C = build_M_and_C(np.array(colors_cur, dtype=np.int8), n)
        M2 = compute_M2(M)
        bad = count_bad(C, M2)
        if bad == 0:
            # Valid! Try to extend more
            for target in range(n + 1, 51):
                if time.time() > DEADLINE - 15:
                    break
                extended = extend_by_one(n, colors_cur)
                if extended is not None:
                    colors_cur = extended
                    n += 1
                else:
                    break

    # --- Strategy 2: Direct local search on N=50 ---
    time_remaining = DEADLINE - time.time()
    if time_remaining > 30:
        # Multiple attempts with different seeds
        attempts = min(6, int((time_remaining - 5) / 10))
        for i in range(attempts):
            if time.time() > DEADLINE - 10:
                break
            tlim = min(12.0, (DEADLINE - 10 - time.time()))
            if tlim < 3:
                break
            res = local_search(50, tlim, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res is not None:
                # Verify
                M, C = build_M_and_C(np.array(res, dtype=np.int8), 50)
                M2 = compute_M2(M)
                if count_bad(C, M2) == 0:
                    return {'n': 50, 'colors': res}

    # --- Strategy 3: Try N=48 ---
    time_remaining = DEADLINE - time.time()
    if time_remaining > 20:
        for i in range(3):
            if time.time() > DEADLINE - 8:
                break
            tlim = min(10.0, (DEADLINE - 8 - time.time()))
            if tlim < 3:
                break
            res = local_search(48, tlim, seed_colors=seed_32 if i == 0 else None, restarts=1)
            if res is not None:
                M, C = build_M_and_C(np.array(res, dtype=np.int8), 48)
                M2 = compute_M2(M)
                if count_bad(C, M2) == 0:
                    return {'n': 48, 'colors': res}

    # --- Strategy 4: Try N=46 ---
    time_remaining = DEADLINE - time.time()
    if time_remaining > 12:
        res = local_search(46, min(8.0, time_remaining - 5), seed_colors=seed_32, restarts=1)
        if res is not None:
            M, C = build_M_and_C(np.array(res, dtype=np.int8), 46)
            M2 = compute_M2(M)
            if count_bad(C, M2) == 0:
                return {'n': 46, 'colors': res}

    # --- Fallback: return best valid coloring found ---
    if n > 32 and len(colors_cur) == n * (n - 1) // 2:
        # Verify
        M, C = build_M_and_C(np.array(colors_cur, dtype=np.int8), n)
        M2 = compute_M2(M)
        if count_bad(C, M2) == 0:
            return {'n': n, 'colors': colors_cur}

    # Final fallback: seed 32
    return {'n': 32, 'colors': seed_32}
