import numpy as np
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7

    def objective_and_penalty(params):
        p = params.reshape(n, 3)
        cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
        
        total = np.sum(s)
        penalty = 0.0
        
        # Boundary: ensure [cx - s/2, cx + s/2] in [-tol, 1+tol]
        for i in range(n):
            if cx[i] - s[i]/2 < -tol:
                penalty += 1e5 * (-tol - (cx[i] - s[i]/2))**2
            if cx[i] + s[i]/2 > 1 + tol:
                penalty += 1e5 * ((cx[i] + s[i]/2) - (1 + tol))**2
            if cy[i] - s[i]/2 < -tol:
                penalty += 1e5 * (-tol - (cy[i] - s[i]/2))**2
            if cy[i] + s[i]/2 > 1 + tol:
                penalty += 1e5 * ((cy[i] + s[i]/2) - (1 + tol))**2
        
        # Pairwise non-overlap
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    ox = t - dx
                    oy = t - dy
                    penalty += 1e5 * (ox*ox + oy*oy)
        
        return total - penalty

    def neg_obj(params):
        return -objective_and_penalty(params)

    # Multiple starts
    starts = []
    
    # Start 1: 2x(1/2) bottom + 8x(1/4) top in 4x2 grid
    s1 = np.zeros(3*n)
    s1[0:3] = [0.25, 0.25, 0.5]
    s1[3:6] = [0.75, 0.25, 0.5]
    for i, (x, y) in enumerate([(0.125,0.625),(0.375,0.625),(0.625,0.625),(0.875,0.625),
                                  (0.125,0.875),(0.375,0.875),(0.625,0.875),(0.875,0.875)]):
        s1[6+3*i:9+3*i] = [x, y, 0.25]
    starts.append(s1)
    
    # Start 2: uniform grid 4x3
    s2 = np.zeros(3*n)
    for i in range(n):
        col = i % 4
        row = i // 4
        s2[3*i] = (col + 0.5) * 0.25
        s2[3*i+1] = (row + 0.5) * 0.25
        s2[3*i+2] = 0.25
    starts.append(s2)
    
    # Start 3: 3 in bottom (1/3 each), 7 in top
    s3 = np.zeros(3*n)
    for i in range(3):
        s3[3*i] = (i + 0.5) / 3
        s3[3*i+1] = 1/6
        s3[3*i+2] = 1/3
    for i in range(7):
        col = i % 4
        row = i // 4
        s3[9+3*i] = (col + 0.5) * 0.25
        s3[9+3*i+1] = 0.625 + row * 0.25
        s3[9+3*i+2] = 0.2
    starts.append(s3)
    
    bounds = [(0.01, 0.99) for _ in range(2*n)] + [(0.05, 0.9)] * n
    
    best = None
    best_val = -np.inf
    for st in starts:
        res = minimize(neg_obj, st, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        val = -res.fun
        if val > best_val:
            best_val = val
            best = res.x
    
    cx = best.reshape(n, 3)[:, 0]
    cy = best.reshape(n, 3)[:, 1]
    s  = best.reshape(n, 3)[:, 2]
    
    centers = np.column_stack([cx, cy]).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides
