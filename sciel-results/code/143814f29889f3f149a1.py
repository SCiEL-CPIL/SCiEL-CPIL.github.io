"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_areas(P):
    """Compute all C(n,3) triangle areas for points P of shape (n,2)."""
    n = P.shape[0]
    # Use vectorized computation
    # For each triple (i, j, k), area = 0.5 * |cross(P[j]-P[i], P[k]-P[i])|
    # We'll compute all pairs of differences first
    # P[i], P[j], P[k] -> area = 0.5 * |(P[j][0]-P[i][0])*(P[k][1]-P[i][1]) - (P[j][1]-P[i][1])*(P[k][0]-P[i][0])|
    
    # Vectorized: create index arrays
    idx = np.arange(n)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing='ij')
    # Only keep i < j < k
    mask = (ii < jj) & (jj < kk)
    ii, jj, kk = ii[mask], jj[mask], kk[mask]
    
    Pi = P[ii]
    Pj = P[jj]
    Pk = P[kk]
    
    # Cross product of (Pj-Pi) and (Pk-Pi)
    cross = (Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1]) - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return areas

def _min_area(P):
    """Return the minimum triangle area for points P."""
    areas = _compute_areas(P)
    return np.min(areas)

def _in_triangle(P):
    """Check if all points are inside the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Point (x,y) is inside if:
    # y >= 0
    # y <= sqrt(3)*x
    # y <= sqrt(3)*(1-x)
    # i.e., y >= 0, x >= 0, 1-x >= 0, y <= sqrt(3)*min(x, 1-x)
    
    x = P[:, 0]
    y = P[:, 1]
    s3 = np.sqrt(3)
    
    # Check y >= 0
    if np.any(y < -1e-9):
        return False
    # Check x >= 0 and x <= 1
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return False
    # Check y <= sqrt(3) * x and y <= sqrt(3) * (1-x)
    if np.any(y > s3 * x + 1e-9):
        return False
    if np.any(y > s3 * (1 - x) + 1e-9):
        return False
    
    return True

def _clip_to_triangle(P):
    """Clip points to be inside the triangle."""
    x = P[:, 0].copy()
    y = P[:, 1].copy()
    s3 = np.sqrt(3)
    
    # Ensure y >= 0
    y = np.maximum(y, 0)
    # Ensure x >= 0
    x = np.maximum(x, 0)
    # Ensure x <= 1
    x = np.minimum(x, 1)
    
    # Clip y to be below the two slanted sides
    # y <= s3 * x
    y = np.minimum(y, s3 * x)
    # y <= s3 * (1 - x)
    y = np.minimum(y, s3 * (1 - x))
    
    return np.column_stack([x, y])

def _objective(params):
    """Objective function to minimize (negative of min area)."""
    n = N
    P = params.reshape(n, 2)
    
    # Check if all points are inside the triangle
    x = P[:, 0]
    y = P[:, 1]
    s3 = np.sqrt(3)
    
    # Violation penalties
    penalty = 0.0
    # y < 0
    viol_y_neg = np.maximum(0, -y)
    penalty += np.sum(viol_y_neg ** 2)
    # x < 0
    viol_x_neg = np.maximum(0, -x)
    penalty += np.sum(viol_x_neg ** 2)
    # x > 1
    viol_x_pos = np.maximum(0, x - 1)
    penalty += np.sum(viol_x_pos ** 2)
    # y > s3 * x
    viol_side1 = np.maximum(0, y - s3 * x)
    penalty += np.sum(viol_side1 ** 2)
    # y > s3 * (1 - x)
    viol_side2 = np.maximum(0, y - s3 * (1 - x))
    penalty += np.sum(viol_side2 ** 2)
    
    if penalty > 1e-6:
        return 1.0 + penalty  # Large penalty for infeasible
    
    # Compute min area
    areas = _compute_areas(P)
    min_a = np.min(areas)
    
    return -min_a

def construct():
    deadline = time.time() + 140  # Safety margin
    
    s3 = np.sqrt(3)
    
    # Initial points: vertices, edge midpoints, centroid
    # Vertices: (0,0), (1,0), (0.5, s3/2)
    # Edge midpoints: (0.5,0), (0.75, s3/4), (0.25, s3/4)
    # Centroid: (0.5, s3/6)
    # That's 7 points. Need 4 more.
    # Let's use a more systematic approach with multiple random starts
    
    best_P = None
    best_score = -np.inf
    
    # Try several random initial configurations
    rng = np.random.RandomState(42)
    
    n_starts = 30
    for start_i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Generate random initial points
        # Mix of structured and random
        if start_i == 0:
            # Structured: vertices, midpoints, centroid, and some interior
            P = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, s3 / 2],
                [0.5, 0.0],
                [0.75, s3 / 4],
                [0.25, s3 / 4],
                [0.5, s3 / 6],
                [0.33, s3 / 6],
                [0.67, s3 / 6],
                [0.5, s3 / 3],
                [0.5, s3 / 12],
            ])
        else:
            # Random points inside triangle
            P = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                P[i, 0] = u + 0.5 * v
                P[i, 1] = s3 / 2 * v
            # Add some jitter
            P += rng.randn(N, 2) * 0.05
            P = _clip_to_triangle(P)
        
        # Ensure all points are distinct (add small perturbation if needed)
        # Use Nelder-Mead to maximize min area
        
        params0 = P.flatten()
        
        result = minimize(
            _objective,
            params0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        if result.success or result.fun < 1.0:
            P_opt = result.x.reshape(N, 2)
            P_opt = _clip_to_triangle(P_opt)
            
            # Verify feasibility
            if _in_triangle(P_opt):
                min_a = _min_area(P_opt)
                if min_a > best_score:
                    best_score = min_a
                    best_P = P_opt.copy()
        
        # Also try L-BFGS-B with bounds
        if time.time() > deadline - 15:
            break
        
        # Define bounds for L-BFGS-B
        # x in [0, 1], y in [0, s3/2]
        bounds = [(0, 1), (0, s3 / 2)] * N
        
        # For L-BFGS-B, we need to handle the triangular constraint via penalty
        # since L-BFGS-B only supports box constraints
        
        result2 = minimize(
            _objective,
            params0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result2.fun < 1.0:
            P_opt2 = result2.x.reshape(N, 2)
            P_opt2 = _clip_to_triangle(P_opt2)
            
            if _in_triangle(P_opt2):
                min_a2 = _min_area(P_opt2)
                if min_a2 > best_score:
                    best_score = min_a2
                    best_P = P_opt2.copy()
    
    if best_P is None:
        # Fallback: return a simple configuration
        best_P = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, s3 / 2],
            [0.5, 0.0],
            [0.75, s3 / 4],
            [0.25, s3 / 4],
            [0.5, s3 / 6],
            [0.33, s3 / 6],
            [0.67, s3 / 6],
            [0.5, s3 / 3],
            [0.5, s3 / 12],
        ])
    
    return best_P

# EVOLVE-BLOCK-END
