import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 148.0  # Leave 2s margin

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def popcount(x):
        return bin(x).count('1')

    def build_N(n, colors):
        N = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
        return N

    def count_violations_from_N(n, N, colors):
        if n < 4:
            return 0
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                common = N[c][i] & N[c][j]
                if popcount(common) >= 2:
                    violations += 1
        return violations

    def local_search(n, max_iters, deadline, initial_colors=None):
        n_edges = n * (n - 1) // 2
        if n_edges == 0:
            return ([], 0)

        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

        if initial_colors is None:
            # Use modular initialization which is structurally better
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append((i - j) % 5)
        else:
            colors = list(initial_colors)

        N = build_N(n, colors)
        cur_viol = count_violations_from_N(n, N, colors)

        if cur_viol == 0:
            return (colors, 0)

        best_colors = list(colors)
        best_viol = cur_viol

        T = 1.0
        T_min = 0.001
        cooling = 0.9999

        for it in range(max_iters):
            if time.time() > deadline:
                break

            k = random.randint(0, n_edges - 1)
            i, j = edge_list[k]
            c = colors[k]

            new_c = random.randint(0, 4)
            if new_c == c:
                continue

            # Affected edges are (i, v) and (j, v) for all v != i, j, plus (i, j)
            # We compute old violations for these edges
            old_v = 0
            for v in range(n):
                if v == i or v == j:
                    continue
                # Edge (i, v)
                idx_iv = get_idx(min(i, v), max(i, v), n)
                col_iv = colors[idx_iv]
                if popcount(N[col_iv][i] & N[col_iv][v]) >= 2:
                    old_v += 1
                # Edge (j, v)
                idx_jv = get_idx(min(j, v), max(j, v), n)
                col_jv = colors[idx_jv]
                if popcount(N[col_jv][j] & N[col_jv][v]) >= 2:
                    old_v += 1
            
            # Edge (i, j)
            if popcount(N[c][i] & N[c][j]) >= 2:
                old_v += 1

            # Apply change
            N[c][i] &= ~(1 << j)
            N[c][j] &= ~(1 << i)
            N[new_c][i] |= (1 << j)
            N[new_c][j] |= (1 << i)
            colors[k] = new_c

            # Compute new violations for affected edges
            new_v = 0
            for v in range(n):
                if v == i or v == j:
                    continue
                idx_iv = get_idx(min(i, v), max(i, v), n)
                col_iv = colors[idx_iv]
                if popcount(N[col_iv][i] & N[col_iv][v]) >= 2:
                    new_v += 1
                idx_jv = get_idx(min(j, v), max(j, v), n)
                col_jv = colors[idx_jv]
                if popcount(N[col_jv][j] & N[col_jv][v]) >= 2:
                    new_v += 1
            
            if popcount(N[new_c][i] & N[new_c][j]) >= 2:
                new_v += 1

            delta = new_v - old_v

            if delta <= 0 or random.random() < np.exp(-delta / T):
                cur_viol += delta
                if cur_viol < best_viol:
                    best_viol = cur_viol
                    best_colors = list(colors)
                if cur_viol == 0:
                    return (colors, 0)
            else:
                # Revert
                N[new_c][i] &= ~(1 << j)
                N[new_c][j] &= ~(1 << i)
                N[c][i] |= (1 << j)
                N[c][j] |= (1 << i)
                colors[k] = c

            T *= cooling
            if T < T_min:
                T = T_min

        return (best_colors, best_viol)

    def geometric_init(n):
        # Use finite field GF(5) structure for vertices 0..24
        # Vertex v corresponds to point (v % 5, v // 5)
        # Edge (u, v) gets color based on slope of line through (u) and (v)
        # Slope in GF(5): (y2-y1)/(x2-x1) mod 5, or infinity if x1==x2
        # Map slope 0,1,2,3,4 to colors 0,1,2,3,4
        # For x1==x2 (vertical), use color 4 (infinity)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if n <= 25:
                    x1, y1 = i % 5, i // 5
                    x2, y2 = j % 5, j // 5
                    if x1 == x2:
                        c = 4  # vertical / infinity
                    else:
                        dx = (x2 - x1) % 5
                        dy = (y2 - y1) % 5
                        # Inverse of dx in GF(5)
                        inv = {1: 1, 2: 3, 3: 2, 4: 4}[dx]
                        slope = (dy * inv) % 5
                        c = slope
                else:
                    # For n > 25, use modular fallback for extra vertices
                    c = (i - j) % 5
                colors.append(c)
        return colors

    best_n = 11
    # Fallback valid coloring for n=11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]

    # Try n from 26 down to 12
    for n in range(26, 11, -1):
        if time.time() > deadline:
            break

        time_left = deadline - time.time()
        if time_left < 1.0:
            break

        # Give more time to larger n
        max_restarts = 30 if n > 20 else 80
        found = False

        for r in range(max_restarts):
            if time.time() > deadline:
                break

            max_iters = 150000 if n > 20 else 80000

            # Use geometric initialization for n<=25, modular for n=26
            if n <= 25:
                init_colors = geometric_init(n)
            else:
                init_colors = None  # Use default modular

            colors, viol = local_search(n, max_iters, deadline, initial_colors=init_colors)

            if viol == 0:
                best_n = n
                best_colors = colors
                found = True
                break

        if found:
            break

    return {'n': best_n, 'colors': best_colors}
