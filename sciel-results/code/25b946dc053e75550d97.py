"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    N = 11
    h = np.sqrt(3) / 2
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    def get_points(x):
        # x is flattened barycentric coords [a1, b1, ..., aN, bN]
        # a_i, b_i >= 0, a_i + b_i <= 1
        # c_i = 1 - a_i - b_i
        # P_i = a_i * V0 + b_i * V1 + c_i * V2
        a = x[0::2]
        b = x[1::2]
        c = 1 - a - b
        # P = a*V0 + b*V1 + c*V2
        # V0 = [0,0], V1 = [1,0], V2 = [0.5, h]
        # Px = b*1 + c*0.5 = b + 0.5*(1-a-b) = 0.5 + 0.5*b - 0.5*a
        # Py = c*h = (1-a-b)*h
        px = 0.5 + 0.5*b - 0.5*a
        py = h * (1 - a - b)
        return np.column_stack((px, py))

    def min_area(points):
        # points shape (N, 2)
        # Calculate areas for all triplets
        indices = list(itertools.combinations(range(N), 3))
        idx_i = [idx[0] for idx in indices]
        idx_j = [idx[1] for idx in indices]
        idx_k = [idx[2] for idx in indices]
        
        p1 = points[idx_i]
        p2 = points[idx_j]
        p3 = points[idx_k]
        
        # Cross product magnitude
        # (p2 - p1) x (p3 - p1)
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def objective(x):
        points = get_points(x)
        # Check constraints: a >= 0, b >= 0, a+b <= 1
        # These are handled by bounds and constraints in minimize
        # But we should ensure numerical stability
        a = x[0::2]
        b = x[1::2]
        if np.any(a < -1e-9) or np.any(b < -1e-9) or np.any(a + b > 1 + 1e-9):
            return 1e10
        return -min_area(points)

    bounds = [(0, 1)] * (2 * N)
    
    cons = []
    for i in range(N):
        def make_constraint(idx):
            def constraint(x):
                a = x[2*idx]
                b = x[2*idx+1]
                return 1 - a - b
            return constraint
        cons.append({'type': 'ineq', 'fun': make_constraint(i)})

    best_points = None
    best_score = -np.inf
    
    # Multi-start strategy
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random initial points in barycentric coords
        x0 = rng.random(2 * N)
        # Ensure a+b <= 1 by scaling or rejection
        # Simple rejection
        while True:
            a0 = x0[0::2]
            b0 = x0[1::2]
            if np.all(a0 + b0 <= 1):
                break
            x0 = rng.random(2 * N)
            
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 200, 'ftol': 1e-10})
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = get_points(result.x)
        except Exception:
            continue

    if best_points is None:
        # Fallback to a simple grid or random
        # Just return random points
        x0 = rng.random(2 * N)
        while True:
            a0 = x0[0::2]
            b0 = x0[1::2]
            if np.all(a0 + b0 <= 1):
                break
            x0 = rng.random(2 * N)
        best_points = get_points(x0)

    return best_points
# EVOLVE-BLOCK-END
