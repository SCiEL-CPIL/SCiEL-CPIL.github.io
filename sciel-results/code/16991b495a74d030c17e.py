import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    num_colors = 5

    # Try n from high to low
    for n in range(28, 10, -1):
        if time.time() > deadline - 15:
            break

        # Algebraic-inspired initialization
        def init_structured():
            col = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    d = j - i
                    h = (i * 7 + j * 13 + d * 11) % 5
                    col[i][j] = h
                    col[j][i] = h
            return col

        def build_adj_masks(col):
            adj = [[0]*n for _ in range(num_colors)]
            for i in range(n):
                for j in range(i+1, n):
                    c = col[i][j]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            return adj

        def total_penalty(adj):
            total = 0
            for c in range(num_colors):
                for u in range(n):
                    for v in range(u+1, n):
                        common = adj[c][u] & adj[c][v]
                        k = common.bit_count()
                        if k >= 2:
                            total += k * (k - 1) // 2
            return total

        def compute_delta(i, j, old_c, new_c, col, adj):
            """Compute change in total C4 count when edge (i,j) changes from old_c to new_c."""
            delta = 0

            for c in [old_c, new_c]:
                sign = -1 if c == old_c else 1

                for k in range(n):
                    if k == i or k == j:
                        continue

                    # Pair (i, k)
                    j_in_k = (adj[c][k] >> j) & 1
                    if j_in_k:
                        k_before = (adj[c][i] & adj[c][k]).bit_count()
                        if sign == -1:
                            delta -= (k_before - 1)
                        else:
                            delta += k_before

                    # Pair (j, k)
                    i_in_k = (adj[c][k] >> i) & 1
                    if i_in_k:
                        k_before = (adj[c][j] & adj[c][k]).bit_count()
                        if sign == -1:
                            delta -= (k_before - 1)
                        else:
                            delta += k_before

            return delta

        def apply_change(i, j, old_c, new_c, col, adj):
            col[i][j] = new_c
            col[j][i] = new_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)

        def try_solve():
            best_col = None
            best_penalty = float('inf')

            for init_method in range(5):
                if time.time() > deadline:
                    break

                if init_method == 0:
                    col = init_structured()
                elif init_method == 1:
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = random.randint(0, num_colors-1)
                            col[i][j] = c
                            col[j][i] = c
                elif init_method == 2:
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = (i + 2*j) % 5
                            col[i][j] = c
                            col[j][i] = c
                elif init_method == 3:
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = (i*j) % 5
                            col[i][j] = c
                            col[j][i] = c
                else:
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = (i + j) % 5
                            col[i][j] = c
                            col[j][i] = c

                adj = build_adj_masks(col)
                penalty = total_penalty(adj)

                if penalty == 0:
                    return col

                best_local_penalty = penalty
                best_local_col = [row[:] for row in col]
                best_local_adj = [[row[:] for row in adj[c]] for c in range(num_colors)]

                no_improve = 0
                temp = 3.0

                while time.time() < deadline:
                    if penalty == 0:
                        return col

                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i

                    old_c = col[i][j]
                    new_c = random.randint(0, num_colors-1)
                    if new_c == old_c:
                        continue

                    delta = compute_delta(i, j, old_c, new_c, col, adj)

                    if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / max(temp, 0.1))):
                        apply_change(i, j, old_c, new_c, col, adj)
                        penalty += delta
                        no_improve = 0
                        if penalty < best_local_penalty:
                            best_local_penalty = penalty
                            best_local_col = [row[:] for row in col]
                            best_local_adj = [[row[:] for row in adj[c]] for c in range(num_colors)]
                    else:
                        no_improve += 1

                    if no_improve > 500:
                        break
                    temp *= 0.995

                if best_local_penalty < best_penalty:
                    best_penalty = best_local_penalty
                    best_col = best_local_col

            if best_col is not None and best_penalty == 0:
                return best_col
            return None

        sol = try_solve()
        if sol is not None:
            adj = build_adj_masks(sol)
            if total_penalty(adj) == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}

    # Fallback: n=11 with simple construction
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
