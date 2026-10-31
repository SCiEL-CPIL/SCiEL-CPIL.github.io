import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0

    # 1. Generate compact cluster
    points_axial = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            d = (abs(q) + abs(r) + abs(q+r)) / 2.0
            points_axial.append((d, q, r))
    points_axial.sort(key=lambda x: x[0])
    selected_axial = [(q, r) for _, q, r in points_axial[:N]]

    pts = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        pts.append([x, y])
    pts = np.array(pts)

    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid

    best_L = float('inf')
    best_theta = 0.0

    # Grid search theta
    thetas = np.linspace(0, np.pi / 3, 361) # 0.5 deg steps

    for theta in thetas:
        if time.time() > deadline - safety_margin:
            break

        angles = theta + np.arange(6) * np.pi / 3
        V = np.array([[np.cos(a), np.sin(a)] for a in angles])

        proj = pts_centered.T @ V.T
        A = np.max(np.abs(proj))
        L = 2 * A / np.sqrt(3)

        if L < best_L:
            best_L = L
            best_theta = theta

    inner = [[p[0], p[1], 0.0] for p in pts]
    return inner, (centroid[0], centroid[1]), best_L, best_theta
