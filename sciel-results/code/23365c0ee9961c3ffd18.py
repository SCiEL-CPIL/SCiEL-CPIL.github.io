import numpy as np
from scipy.optimize import minimize

N = 26

def construct():
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    def objective(params):
        pts = params.reshape(N, 3)
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        
        sum_r = -np.sum(r)
        
        coll_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    coll_pen += (min_dist - dist)**2
        
        bound_pen = 0.0
        for i in range(N):
            if x[i] < r[i]:
                bound_pen += (r[i] - x[i])**2
            if x[i] + r[i] > 1:
                bound_pen += (x[i] + r[i] - 1)**2
            if y[i] < r[i]:
                bound_pen += (r[i] - y[i])**2
            if y[i] + r[i] > 1:
                bound_pen += (y[i] + r[i] - 1)**2
                
        return sum_r + 100.0 * coll_pen + 100.0 * bound_pen

    def check_validity(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        r = pts[:, 2]
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] + r[i] > 1 + 1e-6 or y[i] < r[i] - 1e-6 or y[i] + r[i] > 1 + 1e-6:
                return False
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-6:
                    return False
        return True

    best_result = None
    best_score = -np.inf

    # Strategy 1: Grid Init
    side = 5
    x_coords = np.linspace(0.5/side, 1-0.5/side, side)
    y_coords = np.linspace(0.5/side, 1-0.5/side, side)
    points = []
    for i in range(side):
        for j in range(side):
            points.append([x_coords[i], y_coords[j], 0.1])
    x0_grid = np.array(points[:N]).flatten()

    # Strategy 2: Boundary First Init
    points2 = []
    # Corners
    points2.append([0.2, 0.2, 0.18])
    points2.append([0.8, 0.2, 0.18])
    points2.append([0.2, 0.8, 0.18])
    points2.append([0.8, 0.8, 0.18])
    # Edges (Midpoints)
    points2.append([0.5, 0.15, 0.14])
    points2.append([0.5, 0.85, 0.14])
    points2.append([0.15, 0.5, 0.14])
    points2.append([0.85, 0.5, 0.14])
    # Center (18 circles)
    # 5x4 grid in center
    cx = np.linspace(0.3, 0.7, 5)
    cy = np.linspace(0.3, 0.7, 4)
    for i in range(5):
        for j in range(4):
            points2.append([cx[i], cy[j], 0.1])
    # We have 4+4+20 = 28 points. We need 26. Remove 2.
    # Remove two from the center grid, e.g., the ones furthest from center? Or just take first 18.
    # Let's take first 18 from the generated center list.
    center_pts = points2[8:]
    center_pts = center_pts[:18]
    final_points2 = points2[:8] + center_pts
    x0_boundary = np.array(final_points2).flatten()

    inits = [x0_grid, x0_boundary]
    
    for x0 in inits:
        # Try a few restarts with jitter
        for restart in range(3):
            if restart > 0:
                x0_curr = x0.copy()
                for i in range(N):
                    x0_curr[3*i] += np.random.uniform(-0.02, 0.02)
                    x0_curr[3*i+1] += np.random.uniform(-0.02, 0.02)
                    x0_curr[3*i] = np.clip(x0_curr[3*i], 0.01, 0.99)
                    x0_curr[3*i+1] = np.clip(x0_curr[3*i+1], 0.01, 0.99)
            else:
                x0_curr = x0
                
            try:
                res = minimize(objective, x0_curr, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-7})
                pts = res.x.reshape(N, 3)
                if check_validity(pts):
                    score = np.sum(pts[:, 2])
                    if score > best_score:
                        best_score = score
                        best_result = pts
            except Exception:
                pass

    if best_result is not None:
        return best_result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-START
# (The code above is the content)
# EVOLVE-BLOCK-END
