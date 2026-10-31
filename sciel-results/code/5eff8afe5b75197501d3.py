"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SIDE = 1.0
HEIGHT = np.sqrt(3) / 2.0
# Area of the equilateral triangle with side 1
TRI_AREA = 0.5 * SIDE * HEIGHT

# Precompute all triples of indices for C(11,3) = 165 triangles
TRIPLES = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRIPLES.append((i, j, k))
TRIPLES = np.array(TRIPLES)

DEADLINE = 150.0  # seconds, with safety margin we'll use ~145s


def points_to_bary(pts):
    """Convert Cartesian points to barycentric coordinates relative to the reference triangle."""
    # Reference triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, HEIGHT])
    
    # Area of reference triangle
    ref_area = TRI_AREA
    
    # Barycentric coordinates: for point P,
    # alpha = area(P, B, C) / ref_area
    # beta = area(A, P, C) / ref_area
    # gamma = area(A, B, P) / ref_area
    
    n = len(pts)
    bary = np.zeros((n, 3))
    for idx, p in enumerate(pts):
        # alpha: area of triangle (p, B, C)
        alpha = 0.5 * abs((B[0] - p[0]) * (C[1] - p[1]) - (C[0] - p[0]) * (B[1] - p[1])) / ref_area
        # beta: area of triangle (A, p, C)
        beta = 0.5 * abs((p[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (p[1] - A[1])) / ref_area
        # gamma: area of triangle (A, B, p)
        gamma = 0.5 * abs((B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0])) / ref_area
        bary[idx] = [alpha, beta, gamma]
    return bary


def bary_to_points(bary):
    """Convert barycentric coordinates back to Cartesian."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, HEIGHT])
    
    n = len(bary)
    pts = np.zeros((n, 2))
    for idx in range(n):
        alpha, beta, gamma = bary[idx]
        pts[idx] = alpha * A + beta * B + gamma * C
    return pts


def min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three of the given points."""
    n = len(pts)
    min_area = float('inf')
    for i, j, k in TRIPLES:
        # Area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
        d1 = pts[j] - pts[i]
        d2 = pts[k] - pts[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        area = 0.5 * abs(cross)
        if area < min_area:
            min_area = area
    return min_area


def objective(x):
    """Objective to minimize: negative of the minimum triangle area (we want to maximize it).
    x is a flat array of 2*N barycentric-like coordinates, but we use a simpler parameterization.
    
    We parameterize points directly in Cartesian, but with a soft constraint that they stay in the triangle.
    Actually, let's use barycentric coordinates with a softmax to ensure they sum to 1 and are non-negative.
    """
    # x has 3*N elements: barycentric coords for each point
    # We use a soft constraint: penalize if any barycentric coord is negative or if they don't sum to 1
    bary = x.reshape(N, 3)
    
    # Soft constraint penalty
    # bary should have each row sum to 1 and each element >= 0
    sum_penalty = np.sum((np.sum(bary, axis=1) - 1.0) ** 2)
    neg_penalty = np.sum(np.maximum(0, -bary) ** 2)
    
    # Convert to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, HEIGHT])
    
    pts = np.zeros((N, 2))
    for idx in range(N):
        alpha, beta, gamma = bary[idx]
        pts[idx] = alpha * A + beta * B + gamma * C
    
    # Compute min triangle area
    min_area = min_triangle_area(pts)
    
    # Return negative min area plus penalties
    return -min_area + 100.0 * (sum_penalty + neg_penalty)


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Start with a good initial guess: vertices + centroid + midpoints + some interior points
    # Let's try a structured initial configuration
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, HEIGHT])
    
    # Initial points: 3 vertices, 3 midpoints of edges, 1 centroid, 4 more
    mid_AB = (A + B) / 2.0
    mid_BC = (B + C) / 2.0
    mid_CA = (C + A) / 2.0
    centroid = (A + B + C) / 3.0
    
    initial_pts = [A, B, C, mid_AB, mid_BC, mid_CA, centroid]
    
    # Add 4 more points: quarter points
    q1 = 0.25 * A + 0.75 * B
    q2 = 0.25 * B + 0.75 * C
    q3 = 0.25 * C + 0.75 * A
    q4 = 0.5 * centroid + 0.5 * A  # between centroid and A
    
    initial_pts.extend([q1, q2, q3, q4])
    initial_pts = np.array(initial_pts[:N])
    
    # Convert to barycentric for optimization
    bary_init = points_to_bary(initial_pts)
    x_init = bary_init.flatten()
    
    best_x = x_init.copy()
    best_score = -objective(x_init)  # This is min_area - penalties, we want to maximize min_area
    # Actually, let's just track the min area directly
    best_min_area = min_triangle_area(bary_to_points(bary_init))
    
    # Try differential evolution for global optimization
    # Bounds: barycentric coords in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * N)
    
    # We need to ensure rows sum to 1, so let's use a different parameterization
    # Use 2*N parameters: for each point, two barycentric coords, third is 1 - a - b
    # This ensures sum to 1 automatically. Constraint: a >= 0, b >= 0, a + b <= 1
    
    def objective_2param(y):
        """y has 2*N elements. For point i: a_i = y[2*i], b_i = y[2*i+1], c_i = 1 - a_i - b_i"""
        a = y[0::2]
        b = y[1::2]
        c = 1.0 - a - b
        
        # Penalty if any barycentric coord is negative
        neg_penalty = np.sum(np.maximum(0, -a) ** 2) + np.sum(np.maximum(0, -b) ** 2) + np.sum(np.maximum(0, -c) ** 2)
        
        # Convert to Cartesian
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, HEIGHT])
        
        pts = np.zeros((N, 2))
        for idx in range(N):
            alpha, beta, gamma = a[idx], b[idx], c[idx]
            pts[idx] = alpha * A + beta * B + gamma * C
        
        min_area = min_triangle_area(pts)
        return -min_area + 100.0 * neg_penalty
    
    # Initial values in 2-param form
    y_init = np.zeros(2 * N)
    for i in range(N):
        y_init[2 * i] = bary_init[i, 0]   # alpha
        y_init[2 * i + 1] = bary_init[i, 1]  # beta
    
    # Bounds for 2-param: a in [0, 1], b in [0, 1], but we need a + b <= 1
    # We'll use bounds [0, 1] and rely on the penalty
    bounds_2param = [(0.0, 1.0)] * (2 * N)
    
    # Try a few optimization strategies
    strategies = []
    
    # Strategy 1: Differential Evolution
    strategies.append(('DE', lambda: differential_evolution(
        objective_2param, bounds_2param, seed=42, maxiter=200, tol=1e-8,
        polish=False, update_interval=1
    )))
    
    # Strategy 2: L-BFGS-B from multiple random starts
    n_starts = 20
    rng = np.random.RandomState(42)
    for s in range(n_starts):
        y_rand = rng.uniform(0.0, 1.0, size=2 * N)
        # Ensure a + b <= 1 by scaling
        for i in range(N):
            if y_rand[2 * i] + y_rand[2 * i + 1] > 1.0:
                scale = 1.0 / (y_rand[2 * i] + y_rand[2 * i + 1] + 1e-10)
                y_rand[2 * i] *= scale
                y_rand[2 * i + 1] *= scale
        strategies.append((f'L-BFGS-B start {s}', lambda yr=y_rand: minimize(
            objective_2param, yr, method='L-BFGS-B', bounds=bounds_2param,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )))
    
    # Strategy 3: Nelder-Mead from the initial structured guess
    strategies.append(('Nelder-Mead', lambda: minimize(
        objective_2param, y_init, method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
    )))
    
    # Run strategies until deadline
    for name, func in strategies:
        if time.time() > deadline:
            break
        try:
            result = func()
            if result.success or hasattr(result, 'fun'):
                y_opt = result.x
                # Evaluate the actual min area (without penalty)
                a = y_opt[0::2]
                b = y_opt[1::2]
                c = 1.0 - a - b
                A = np.array([0.0, 0.0])
                B = np.array([1.0, 0.0])
                C = np.array([0.5, HEIGHT])
                pts = np.zeros((N, 2))
                for idx in range(N):
                    pts[idx] = a[idx] * A + b[idx] * B + c[idx] * C
                min_area = min_triangle_area(pts)
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_x = y_opt.copy()
        except Exception:
            continue
    
    # If we have time, do a final polish with tighter tolerances
    if time.time() < deadline - 10.0 and best_x is not None:
        try:
            result = minimize(
                objective_2param, best_x, method='L-BFGS-B', bounds=bounds_2param,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if hasattr(result, 'x'):
                y_opt = result.x
                a = y_opt[0::2]
                b = y_opt[1::2]
                c = 1.0 - a - b
                A = np.array([0.0, 0.0])
                B = np.array([1.0, 0.0])
                C = np.array([0.5, HEIGHT])
                pts = np.zeros((N, 2))
                for idx in range(N):
                    pts[idx] = a[idx] * A + b[idx] * B + c[idx] * C
                min_area = min_triangle_area(pts)
                if min_area > best_min_area:
                    best_min_area = min_area
                    best_x = y_opt.copy()
        except Exception:
            pass
    
    # Convert best_x back to Cartesian points
    if best_x is not None:
        a = best_x[0::2]
        b = best_x[1::2]
        c = 1.0 - a - b
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, HEIGHT])
        pts = np.zeros((N, 2))
        for idx in range(N):
            pts[idx] = a[idx] * A + b[idx] * B + c[idx] * C
    else:
        pts = initial_pts
    
    # Clip to ensure points are inside the triangle (small numerical errors)
    # Re-convert to barycentric and clip
    bary_final = points_to_bary(pts)
    # Clip negative values and renormalize
    bary_final = np.maximum(0, bary_final)
    row_sums = np.sum(bary_final, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    bary_final = bary_final / row_sums
    
    pts_final = bary_to_points(bary_final)
    
    return pts_final
