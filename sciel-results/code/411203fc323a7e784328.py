import numpy as np
from scipy.optimize import minimize

N = 13

# Precompute indices for all triangles
def get_triangle_indices(n):
    i_idx = []
    j_idx = []
    k_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    return np.array(i_idx), np.array(j_idx), np.array(k_idx)

TRI_I, TRI_J, TRI_K = get_triangle_indices(N)

def construct():
    # Initial guess: Regular 13-gon angles
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    def score_from_angles(angles):
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Hull Area (Polygon Area via Shoelace)
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        hull_area = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        if hull_area < 1e-10:
            return 0.0

        # Vectorized Min Triangle Area
        xi = x[TRI_I]
        yi = y[TRI_I]
        xj = x[TRI_J]
        yj = y[TRI_J]
        xk = x[TRI_K]
        yk = y[TRI_K]
        
        areas = 0.5 * np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        min_area = np.min(areas)
        
        return min_area / hull_area

    def negative_score(angles):
        return -score_from_angles(angles)

    best_result = None
    best_val = -float('inf')
    
    rng = np.random.default_rng(42)
    
    # Run optimization from multiple starting points
    for attempt in range(10):
        if attempt == 0:
            start_angles = angles.copy()
        else:
            # Small random perturbations
            start_angles = angles + rng.normal(0, 0.05, N)
            # Keep them sorted and in range to help optimizer
            start_angles = np.sort(start_angles % (2 * np.pi))
            
        try:
            res = minimize(negative_score, start_angles, method='Powell', 
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    final_angles = np.sort(best_result % (2 * np.pi))
    points = np.column_stack([np.cos(final_angles), np.sin(final_angles)])
    
    return points
