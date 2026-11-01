import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
TIME_LIMIT = 100.0

def construct():
    sqrt3 = np.sqrt(3)
    min_dist = sqrt3
    
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    u_normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def get_positions(x):
        S = x[0]
        pos = x[1:].reshape(N, 2)
        return S, pos

    def objective(x):
        S, _ = get_positions(x)
        return S

    def constraint_func(x):
        S, pos = get_positions(x)
        vals = []
        for i in range(N):
            for j in range(i+1, N):
                diff = pos[i] - pos[j]
                dist_sq = np.dot(diff, diff)
                vals.append(dist_sq - min_dist**2)
        for i in range(N):
            for k in range(6):
                proj = np.dot(pos[i], u_normals[k])
                vals.append((sqrt3/2.0) * (S - 1.0) - proj)
        return np.array(vals)

    def is_valid(S, pos):
        for i in range(N):
            for j in range(i+1, N):
                diff = pos[i] - pos[j]
                if np.dot(diff, diff) < min_dist**2 - 1e-8:
                    return False
        for i in range(N):
            for k in range(6):
                proj = np.dot(pos[i], u_normals[k])
                if proj > (sqrt3/2.0) * (S - 1.0) + 1e-8:
                    return False
        return True

    best_S = float('inf')
    best_x = None

    # Initial guess 1: 7+4
    def init_1():
        pos = np.array([
            [0, 0],
            [sqrt3, 0], [-sqrt3, 0],
            [sqrt3/2, 1.5], [-sqrt3/2, 1.5],
            [sqrt3/2, -1.5], [-sqrt3/2, -1.5],
            [2*sqrt3, 0], [-2*sqrt3, 0],
            [0, 3], [0, -3]
        ])
        pos += np.random.normal(0, 0.05, pos.shape)
        return np.concatenate(([3.5], pos.flatten()))

    # Initial guess 2: Grid
    def init_2():
        pos = []
        for k in range(-2, 3): pos.append([k*sqrt3, 0])
        for k in range(-1, 2):
            pos.append([k*sqrt3 + sqrt3/2, 1.5])
            pos.append([k*sqrt3 + sqrt3/2, -1.5])
        pos = np.array(pos)
        pos += np.random.normal(0, 0.05, pos.shape)
        return np.concatenate(([3.5], pos.flatten()))

    # Initial guess 3: Random
    def init_3():
        pos = np.random.normal(0, 1.5, (N, 2))
        return np.concatenate(([4.0], pos.flatten()))

    guesses = [init_1(), init_2(), init_3()]
    
    for x0 in guesses:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraint_func,
                           bounds=[(3.0, 10.0)] + [(-10.0, 10.0)]*2*N,
                           options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success:
                S, pos = get_positions(res.x)
                if is_valid(S, pos):
                    if S < best_S:
                        best_S = S
                        best_x = res.x
        except:
            pass

    if best_x is not None:
        S, pos = get_positions(best_x)
        inner = [[pos[i][0], pos[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), S, 0.0
    else:
        # Fallback
        inner = [[k*sqrt3, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 20.0, 0.0
