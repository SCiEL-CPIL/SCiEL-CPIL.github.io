"""Heilbronn triangle problem: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import Delaunay

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # area of unit equilateral triangle with side 1

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])

# Precompute all pairs (i, j) for i < j, 0 <= i, j < N
N_PAIRS = N * (N - 1) // 2
pair_indices = []
for i in range(N):
    for j in range(i + 1, N):
        pair_indices.append((i, j))
pair_indices = np.array(pair_indices)

# Precompute all triples (i, j, k) for i < j < k
triple_indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            triple_indices.append((i, j, k))
triple_indices = np.array(triple_indices)
N_TRIPLES = len(triple_indices)


def bary_to_cart(bary):
    """Convert barycentric coordinates (n, 3) to Cartesian (n, 2)."""
    # bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2
    # V0 = (0,0), V1 = (1,0), V2 = (0.5, sqrt(3)/2)
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2.0) * bary[:, 2]
    return np.column_stack([x, y])


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    min_area = np.inf
    for t in range(N_TRIPLES):
        i, j, k = triple_indices[t]
        # Area = 0.5 * |cross(pj - pi, pk - pi)|
        dx1 = points[j, 0] - points[i, 0]
        dy1 = points[j, 1] - points[i, 1]
        dx2 = points[k, 0] - points[i, 0]
        dy2 = points[k, 1] - points[i, 1]
        area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
        if area < min_area:
            min_area = area
    return min_area


def soft_min_triangle_area(points, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    n = len(points)
    # Compute all triangle areas
    areas = np.empty(N_TRIPLES)
    for t in range(N_TRIPLES):
        i, j, k = triple_indices[t]
        dx1 = points[j, 0] - points[i, 0]
        dy1 = points[j, 1] - points[i, 1]
        dx2 = points[k, 0] - points[i, 0]
        dy2 = points[k, 1] - points[i, 1]
        areas[t] = abs(dx1 * dy2 - dy1 * dx2)  # 2*area
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, subtract max
    max_val = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_val)))) + (-beta * max_val)
    soft_min = -log_sum / beta
    return soft_min


def objective_bary(bary, beta):
    """Objective in barycentric coordinates: negative soft-min triangle area."""
    points = bary_to_cart(bary)
    return -soft_min_triangle_area(points, beta)


def random_bary_init(rng):
    """Generate random barycentric coordinates for N points."""
    bary = np.zeros((N, 3))
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        bary[_] = [u, v, 1.0 - u - v]
    return bary


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial barycentric coordinates
        if restart == 0:
            # Start with a structured initialization
            bary = np.zeros((N, 3))
            # Place points roughly in a grid pattern
            grid_positions = [
                (0.5, 0.0), (0.0, 0.5), (0.0, 0.0),  # corners and edge midpoints
                (0.25, 0.25), (0.75, 0.25), (0.25, 0.75),
                (0.5, 0.25), (0.25, 0.5), (0.75, 0.5),
                (0.5, 0.5), (0.5, 0.75)
            ]
            for idx, (u, v) in enumerate(grid_positions):
                if idx < N:
                    bary[idx] = [u, v, 1.0 - u - v]
        else:
            bary = random_bary_init(rng)
        
        # Constraint: barycentric coords must be non-negative and sum to 1
        # We'll use a penalty approach or project after each step
        
        # Optimize in Cartesian space with constraints
        points_init = bary_to_cart(bary)
        
        # Use L-BFGS-B with multiple beta values (annealing)
        betas = [50, 100, 200, 500, 1000]
        current_points = points_init.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Define objective and gradient
            def obj(x):
                pts = x.reshape(N, 2)
                return -soft_min_triangle_area(pts, beta)
            
            # Simple gradient via finite differences for L-BFGS-B
            res = minimize(
                obj,
                current_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                current_points = res.x.reshape(N, 2)
        
        # Project back into the triangle (clip to barycentric constraints)
        # Convert to barycentric and clip
        # For each point, compute barycentric coords
        for i in range(N):
            px, py = current_points[i]
            # Barycentric coordinates
            # Solve: p = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
            # x = b1 + 0.5*b2
            # y = (sqrt(3)/2)*b2
            b2 = py / (SQRT3 / 2.0)
            b1 = px - 0.5 * b2
            b0 = 1.0 - b1 - b2
            # Clip to [0, 1] and renormalize
            bary_i = np.array([b0, b1, b2])
            bary_i = np.clip(bary_i, 0, None)
            s = bary_i.sum()
            if s > 0:
                bary_i = bary_i / s
            current_points[i] = bary_to_cart(bary_i.reshape(1, 3))[0]
        
        # Evaluate
        score = min_triangle_area(current_points)
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
        
        # Polish with high precision
        if time.time() > deadline:
            break
        
        # Final polish with very high beta
        for beta in [2000, 5000]:
            if time.time() > deadline:
                break
            
            def obj_polish(x):
                pts = x.reshape(N, 2)
                return -soft_min_triangle_area(pts, beta)
            
            res = minimize(
                obj_polish,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.success:
                polished = res.x.reshape(N, 2)
                # Project into triangle
                for i in range(N):
                    px, py = polished[i]
                    b2 = py / (SQRT3 / 2.0)
                    b1 = px - 0.5 * b2
                    b0 = 1.0 - b1 - b2
                    bary_i = np.array([b0, b1, b2])
                    bary_i = np.clip(bary_i, 0, None)
                    s = bary_i.sum()
                    if s > 0:
                        bary_i = bary_i / s
                    polished[i] = bary_to_cart(bary_i.reshape(1, 3))[0]
                
                polished_score = min_triangle_area(polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = polished.copy()
    
    # Final fallback: ensure we have valid points
    if best_points is None:
        # Use a simple deterministic configuration
        best_points = np.array([
            [0.5, 0.0], [0.0, 0.5], [0.0, 0.0],
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75],
            [0.5, 0.25], [0.25, 0.5], [0.75, 0.5],
            [0.5, 0.5], [0.5, 0.75]
        ])
        # Project into triangle
        for i in range(N):
            px, py = best_points[i]
            b2 = py / (SQRT3 / 2.0)
            b1 = px - 0.5 * b2
            b0 = 1.0 - b1 - b2
            bary_i = np.array([b0, b1, b2])
            bary_i = np.clip(bary_i, 0, None)
            s = bary_i.sum()
            if s > 0:
                bary_i = bary_i / s
            best_points[i] = bary_to_cart(bary_i.reshape(1, 3))[0]
    
    return best_points
