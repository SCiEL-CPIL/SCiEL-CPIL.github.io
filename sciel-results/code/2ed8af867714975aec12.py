import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # GF(5) arithmetic
    # GF(5) = {0, 1, 2, 3, 4} with mod 5 arithmetic
    # Multiplication table for GF(5) is just mod 5
    gf5_mult = np.array([[i * j % 5 for j in range(5)] for i in range(5)], dtype=np.int32)
    gf5_inv = {1: 1, 2: 3, 3: 2, 4: 4}

    def gf5_add(a, b):
        return (a + b) % 5

    def gf5_sub(a, b):
        return (a - b) % 5

    def gf5_mul(a, b):
        return gf5_mult[a, b]

    def gf5_div(a, b):
        if b == 0:
            return None
        return gf5_mul(a, gf5_inv[b])

    # Construction for n=25: vertices are elements of GF(5)^2
    # Edge colors determined by slope of line connecting them
    # Colors: 0=infinity (vertical), 1-4=slopes 1-4
    n = 25
    colors = []
    for i in range(n):
        ai, bi = i // 5, i % 5
        for j in range(i + 1, n):
            aj, bj = j // 5, j % 5
            d1 = gf5_sub(aj, ai)  # x2 - x1
            d2 = gf5_sub(bj, bi)  # y2 - y1

            if d1 == 0:
                c = 0  # vertical line (infinite slope)
            else:
                slope = gf5_div(d2, d1)
                c = slope + 1  # slopes 1-4 mapped to colors 1-4
            colors.append(c)

    # Verify C4-freeness
    def has_c4(color_val, n, colors):
        """Check if any color class contains a C4."""
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color_val:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1

        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return True
        return False

    def verify_valid(n, colors):
        for c in range(5):
            if has_c4(c, n, colors):
                return False
        return True

    # Fallback construction for n=16 using GF(4)^2
    def construct_gf4():
        gf4_mult = {
            (0,0):0, (0,1):0, (0,2):0, (0,3):0,
            (1,0):0, (1,1):1, (1,2):2, (1,3):3,
            (2,0):0, (2,1):2, (2,2):3, (2,3):2,
            (3,0):0, (3,1):3, (3,2):2, (3,3):1,
        }
        gf4_inv = {1: 1, 2: 3, 3: 2}

        def gf4_add(a, b):
            return a ^ b

        def gf4_mul(a, b):
            return gf4_mult[(a, b)]

        def gf4_div(a, b):
            return gf4_mul(a, gf4_inv[b])

        n = 16
        colors = []
        for i in range(n):
            ai, bi = i // 4, i % 4
            for j in range(i + 1, n):
                aj, bj = j // 4, j % 4
                d1 = gf4_add(ai, aj)
                d2 = gf4_add(bi, bj)

                if d1 == 0 and d2 == 0:
                    c = 0
                elif d1 == 0:
                    c = 0
                elif d2 == 0:
                    c = 1
                else:
                    slope = gf4_div(d2, d1)
                    if slope == 1:
                        c = 2
                    elif slope == 2:
                        c = 3
                    else:
                        c = 4
                colors.append(c)
        return n, colors

    # Fallback construction for n=11
    def construct_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return n, colors

    # First, verify the GF(5) construction
    if verify_valid(n, colors):
        return {'n': n, 'colors': colors}

    # Try greedy construction for larger n
    def greedy_forbidden_pair(target_n, max_restarts=100):
        import random

        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break

            random.seed(42 + restart * 7)
            n = target_n
            color_of = {}
            adj = [[set() for _ in range(n)] for _ in range(5)]
            cn = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]

            valid = True
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            edge_order = list(range(len(edges)))
            random.shuffle(edge_order)

            for idx in edge_order:
                if time.time() > deadline - 10:
                    valid = False
                    break

                u, v = edges[idx]
                best_color = -1
                best_risk = float('inf')

                for c in range(5):
                    if cn[c][u][v] != 0:
                        continue

                    ok = True
                    for w in adj[c][u]:
                        if cn[c][w][v] != 0:
                            ok = False
                            break
                    if not ok:
                        continue

                    for w in adj[c][v]:
                        if cn[c][u][w] != 0:
                            ok = False
                            break
                    if not ok:
                        continue

                    risk = len(adj[c][u]) + len(adj[c][v])
                    if risk < best_risk:
                        best_risk = risk
                        best_color = c

                if best_color == -1:
                    valid = False
                    break

                c = best_color
                color_of[(u, v)] = c

                for w in adj[c][u]:
                    cn[c][w][v] += 1
                    cn[c][v][w] += 1
                for w in adj[c][v]:
                    cn[c][u][w] += 1
                    cn[c][w][u] += 1

                adj[c][u].add(v)
                adj[c][v].add(u)

            if not valid:
                continue

            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors_list.append(color_of[(i, j)])

            if verify_valid(n, colors_list):
                return n, colors_list

        return None

    # Try greedy for n=29 down to 26
    for target_n in [29, 28, 27, 26]:
        if time.time() > deadline - 20:
            break
        result = greedy_forbidden_pair(target_n, max_restarts=200)
        if result is not None:
            n_found, colors_found = result
            return {'n': n_found, 'colors': colors_found}

    # Fallback to GF(4) construction
    n_fb, colors_fb = construct_gf4()
    if verify_valid(n_fb, colors_fb):
        return {'n': n_fb, 'colors': colors_fb}

    # Final fallback
    n_fb2, colors_fb2 = construct_fallback()
    return {'n': n_fb2, 'colors': colors_fb2}
