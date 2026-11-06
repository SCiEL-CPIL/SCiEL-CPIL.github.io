import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1
                       for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback[:]

    target_ns = [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]

    for n in target_ns:
        if time.time() > deadline - 20.0:
            break

        m = n * (n - 1) // 2
        edge_endpoints = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for attempt in range(25):
            if time.time() > deadline - 20.0:
                break

            colors = [random.randint(0, 4) for _ in range(m)]
            adj = [[0] * n for _ in range(5)]
            for idx, (i, j) in enumerate(edge_endpoints):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)

            def cv():
                t = 0
                for c in range(5):
                    for a in range(n):
                        ma = adj[c][a]
                        for b in range(a + 1, n):
                            if (ma & adj[c][b]).bit_count() >= 2:
                                t += 1
                return t

            cur_v = cv()
            if cur_v == 0:
                return {'n': n, 'colors': colors}

            best_v = cur_v
            best_col = colors[:]
            T = 3.0

            for step in range(15000):
                if time.time() > deadline - 20.0:
                    break
                if cur_v == 0:
                    return {'n': n, 'colors': colors}
                if cur_v < best_v:
                    best_v = cur_v
                    best_col = colors[:]

                idx = random.randint(0, m - 1)
                i, j = edge_endpoints[idx]
                old_c = colors[idx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue

                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                colors[idx] = new_c

                nv = cv()
                if nv > cur_v + 8:
                    adj[new_c][i] &= ~(1 << j)
                    adj[new_c][j] &= ~(1 << i)
                    adj[old_c][i] |= (1 << j)
                    adj[old_c][j] |= (1 << i)
                    colors[idx] = old_c
                else:
                    cur_v = nv
                T *= 0.9995

            if best_v == 0:
                return {'n': n, 'colors': best_col}

    return {'n': best_n, 'colors': best_colors}
