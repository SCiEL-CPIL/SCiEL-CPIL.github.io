import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    q = p * p  # 49
    total_points = q * q  # 2401

    elements = [(a, b) for a in range(p) for b in range(p)]

    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1 * a2 + b1 * b2 * k) % p, (a1 * b2 + a2 * b1) % p)

    points = [(x, y) for x in elements for y in elements]

    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = [((1, 0), m) for m in elements] + [((0, 1), (0, 0))]
    num_dirs = len(directions)  # 50

    # Precompute line mates: for each point and direction, list of other points on the line
    point_line_mates = [[[] for _ in range(num_dirs)] for _ in range(total_points)]
    for d in range(num_dirs):
        vx, vy = directions[d]
        line_groups = {}
        for i, pt in enumerate(points):
            x, y = pt
            if vx == (1, 0):
                mx = mul(vy, x)
                c = ((y[0] - mx[0]) % p, (y[1] - mx[1]) % p)
            else:
                c = x
            if c not in line_groups:
                line_groups[c] = []
            line_groups[c].append(i)
        for group in line_groups.values():
            for i in group:
                for j in group:
                    if i != j:
                        point_line_mates[i][d].append(j)

    deadline = time.time() + 570

    def greedy_build(order):
        bc = [[0] * num_dirs for _ in range(total_points)]
        S = set()
        for pt in order:
            if min(bc[pt]) == 0:
                S.add(pt)
                for d in range(num_dirs):
                    for mate in point_line_mates[pt][d]:
                        bc[mate][d] += 1
        return S, bc

    best_S = set()
    best_bc = None
    rng = random.Random(999)

    for trial in range(1000):
        if time.time() > deadline:
            break
        order = list(range(total_points))
        rng.shuffle(order)
        S, bc = greedy_build(order)
        if len(S) > len(best_S):
            best_S = S
            best_bc = bc

    if best_bc is not None and time.time() < deadline - 20:
        in_S = [False] * total_points
        for pt in best_S:
            in_S[pt] = True
        bc = [row[:] for row in best_bc]

        for _ in range(500000):
            if time.time() > deadline:
                break
            pt = rng.randint(0, total_points - 1)
            if not in_S[pt] and min(bc[pt]) == 0:
                in_S[pt] = True
                best_S.add(pt)
                for d in range(num_dirs):
                    for mate in point_line_mates[pt][d]:
                        bc[mate][d] += 1
            elif in_S[pt] and rng.random() < 0.3:
                in_S[pt] = False
                best_S.discard(pt)
                for d in range(num_dirs):
                    for mate in point_line_mates[pt][d]:
                        bc[mate][d] -= 1

        # Final pass to add any feasible points
        for pt in range(total_points):
            if time.time() > deadline:
                break
            if not in_S[pt] and min(bc[pt]) == 0:
                in_S[pt] = True
                best_S.add(pt)
                for d in range(num_dirs):
                    for mate in point_line_mates[pt][d]:
                        bc[mate][d] += 1

    N_points = []
    for i, pt in enumerate(points):
        if i not in best_S:
            x, y = pt
            N_points.append([[x[0], x[1]], [y[0], y[1]]])

    return np.array(N_points, dtype=np.int64)
