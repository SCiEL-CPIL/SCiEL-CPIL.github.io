import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_extents(points, cx, cy, angle_deg):
    """For given center and orientation, find min side length to contain all points."""
    angle = np.radians(angle_deg)
    max_extent = 0.0
    for i in range(6):
        a = angle + i * np.pi / 3 + np.pi / 6
        n = np.array([np.cos(a), np.sin(a)])
        ds = (points[:, 0] - cx) * n[0] + (points[:, 1] - cy) * n[1]
        max_extent = max(max_extent, np.max(ds))
    return max_extent / (np.sqrt(3) / 2)

def objective(params):
    n = len(params) // 3
    inner = params[:3*n].reshape(n, 3)
    ocx, ocy, oangle = params[3*n], params[3*n+1], params[3*n+2]
    centers = inner[:, :2]
    s = hexagon_extents(centers, ocx, ocy, oangle)
    return s

def construct():
    start_time = time.time()
    deadline = 140.0

    # Config 1: flower + 4 extras
    positions = [[0.0, 0.0, 0.0]]
    for i in range(6):
        angle = i * 60
        positions.append([np.sqrt(3) * np.cos(np.radians(angle)), np.sqrt(3) * np.sin(np.radians(angle)), 0.0])
    for i in range(4):
        angle = 30 + i * 90
        positions.append([2 * np.sqrt(3) * np.cos(np.radians(angle)), 2 * np.sqrt(3) * np.sin(np.radians(angle)), 0.0])
    positions = positions[:N]
    init_params = []
    for p in positions:
        init_params.extend([p[0], p[1], p[2]])
    init_params.extend([0.0, 0.0, 0.0])

    best_params = np.array(init_params)
    best_score = objective(best_params)

    configs = [np.array(init_params)]

    # Config 2: linear row
    row_positions = [[i * np.sqrt(3), 0.0, 0.0] for i in range(N)]
    row_params = []
    for p in row_positions:
        row_params.extend([p[0], p[1], p[2]])
    row_params.extend([row_positions[N//2][0], 0.0, 0.0])
    configs.append(np.array(row_params))

    # Config 3: 3-5-3
    r353 = []
    for i in range(3):
        r353.append([(i - 1) * np.sqrt(3), 0.0, 0.0])
    for i in range(5):
        r353.append([(i - 2) * np.sqrt(3), np.sqrt(3) * 1.5, 0.0])
    for i in range(3):
        r353.append([(i - 1) * np.sqrt(3), np.sqrt(3) * 3.0, 0.0])
    r353 = r353[:N]
    r353_params = []
    for p in r353:
        r353_params.extend([p[0], p[1], p[2]])
    cx = np.mean([p[0] for p in r353])
    cy = np.mean([p[1] for p in r353])
    r353_params.extend([cx, cy, 0.0])
    configs.append(np.array(r353_params))

    # Config 4: 5-3-3
    r533 = []
    for i in range(5):
        r533.append([(i - 2) * np.sqrt(3), 0.0, 0.0])
    for i in range(3):
        r533.append([(i - 1) * np.sqrt(3), np.sqrt(3) * 1.5, 0.0])
    for i in range(3):
        r533.append([(i - 1) * np.sqrt(3), np.sqrt(3) * 3.0, 0.0])
    r533 = r533[:N]
    r533_params = []
    for p in r533:
        r533_params.extend([p[0], p[1], p[2]])
    cx = np.mean([p[0] for p in r533])
    cy = np.mean([p[1] for p in r533])
    r533_params.extend([cx, cy, 0.0])
    configs.append(np.array(r533_params))

    # Config 5: 4-3-4
    r434 = []
    for i in range(4):
        r434.append([(i - 1.5) * np.sqrt(3), 0.0, 0.0])
    for i in range(3):
        r434.append([(i - 1) * np.sqrt(3), np.sqrt(3) * 1.5, 0.0])
    for i in range(4):
        r434.append([(i - 1.5) * np.sqrt(3), np.sqrt(3) * 3.0, 0.0])
    r434 = r434[:N]
    r434_params = []
    for p in r434:
        r434_params.extend([p[0], p[1], p[2]])
    cx = np.mean([p[0] for p in r434])
    cy = np.mean([p[1] for p in r434])
    r434_params.extend([cx, cy, 0.0])
    configs.append(np.array(r434_params))

    for cfg in configs:
        if time.time() - start_time > deadline:
            break
        try:
            result = minimize(objective, cfg, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    if time.time() - start_time < deadline - 10:
        try:
            result = minimize(objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    n = N
    inner = []
    for i in range(n):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])

    ocx = best_params[3*n]
    ocy = best_params[3*n+1]
    oangle = best_params[3*n+2]

    centers = np.array([[p[0], p[1]] for p in inner])
    side = hexagon_extents(centers, ocx, ocy, oangle)

    return inner, (ocx, ocy), side, oangle
