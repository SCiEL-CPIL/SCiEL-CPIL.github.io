import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    sqrt3 = np.sqrt(3)

    def project_to_triangle(pts):
        pts = np.array(pts, dtype=float)
        for _ in range(5):
            # y >= 0
            mask = pts[:, 1] < 0
            if np.any(mask):
                pts[mask, 1] = 0

            # y <= sqrt(3) * x
            vals = -sqrt3*pts[:, 0] + pts[:, 1]
            mask = vals > 0
            if np.any(mask):
                d = vals[mask] / 2.0
                pts[mask, 0] += d * sqrt3
                pts[mask, 1] -= d

            # y <= -sqrt(3) * (x - 1)
            vals = sqrt3*pts[:, 0] + pts[:, 1] - sqrt3
            mask = vals > 0
            if np.any(mask):
                d = vals[mask] / 2.0
                pts[mask, 0] -= d * sqrt3
                pts[mask, 1] -= d

        return pts

    def min_area(pts):
        min_a = float('inf')
        p = pts
        for i in range(N-2):
            p1 = p[i]
            for j in range(i+1, N-1):
                p2 = p[j]
                v1 = p2 - p1
                for k in range(j+1, N):
                    p3 = p[k]
                    v2 = p3 - p1
                    area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                    if area < min_a:
                        min_a = area
        return min_a

    def objective(pts):
        pts = project_to_triangle(pts)
        return -min_area(pts)

    inits = []

    # Init 1: Hex Grid
    scale = 0.22
    pts = []
    for k in range(6):
        y = k * scale * sqrt3
        if y > sqrt3/2: break
        for j in range(6):
            x = j * scale + (k % 2) * scale/2
            if x > 1.0: break
            if y <= sqrt3*x and y <= -sqrt3*(x-1):
                pts.append([x, y])
    if len(pts) >= N:
        center = np.array([0.5, sqrt3/6])
        dists = np.linalg.norm(np.array(pts) - center, axis=1)
        idx = np.argsort(dists)
        inits.append(np.array(pts)[idx[:N]])

    # Init 2: 4-3-4 Grid
    pts = []
    ys = [0.08, 0.35, 0.62]
    counts = [4, 3, 4]
    for y, c in zip(ys, counts):
        x_min = y / sqrt3
        x_max = 1 - y / sqrt3
        if x_max < x_min: continue
        xs = np.linspace(x_min + 0.02, x_max - 0.02, c)
        for x in xs:
            pts.append([x, y])
    if len(pts) == N:
        inits.append(np.array(pts))

    # Init 3: 3-4-4 Grid
    pts = []
    ys = [0.08, 0.35, 0.62]
    counts = [3, 4, 4]
    for y, c in zip(ys, counts):
        x_min = y / sqrt3
        x_max = 1 - y / sqrt3
        if x_max < x_min: continue
        xs = np.linspace(x_min + 0.02, x_max - 0.02, c)
        for x in xs:
            pts.append([x, y])
    if len(pts) == N:
        inits.append(np.array(pts))

    # Init 4: 5-3-3 Grid
    pts = []
    ys = [0.05, 0.35, 0.65]
    counts = [5, 3, 3]
    for y, c in zip(ys, counts):
        x_min = y / sqrt3
        x_max = 1 - y / sqrt3
        if x_max < x_min: continue
        xs = np.linspace(x_min + 0.02, x_max - 0.02, c)
        for x in xs:
            pts.append([x, y])
    if len(pts) == N:
        inits.append(np.array(pts))
        
    # Init 5: Random
    rng = np.random.default_rng(123)
    valid_pts = []
    while len(valid_pts) < N:
        u, v = rng.random(2)
        if u + v < 1:
            p = u * v1 + v * v2
            valid_pts.append(p)
    inits.append(np.array(valid_pts))

    best_score = -float('inf')
    best_pts = None

    for i, init_pts in enumerate(inits):
        if time.time() > deadline:
            break

        try:
            res = minimize(
                objective,
                init_pts,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'maxiter': 1000,
                    'maxfev': 5000
                }
            )
            final_pts = project_to_triangle(res.x)
            score = -res.fun
            if score > best_score:
                best_score = score
                best_pts = final_pts
        except Exception:
            pass

    if best_pts is None:
        best_pts = inits[0]

    return best_pts
