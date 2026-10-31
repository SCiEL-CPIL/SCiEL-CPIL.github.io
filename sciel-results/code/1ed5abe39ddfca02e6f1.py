"""Rewritten program for Heilbronn triangles task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0  # area of the unit equilateral triangle

# Pre-compute all triangle index combinations
from itertools import combinations
TRI_IDX = np.array(list(combinations(range(N), 3)), dtype=np.int64)
N_TRI = TRI_IDX.shape[0]

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, SQRT3 / 2.0])
VERTS = np.array([V0, V1, V2])


def points_to_bary(p):
    """Convert Cartesian points to barycentric coordinates relative to the triangle."""
    # p shape: (N, 2)
    # Barycentric: point = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1, bi>=0
    # Using V0 as origin: point = b1*(V1-V0) + b2*(V2-V0)
    e1 = V1 - V0  # [1, 0]
    e2 = V2 - V0  # [0.5, sqrt(3)/2]
    # Solve [e1 e2] @ [b1, b2]^T = point
    M = np.column_stack([e1, e2])
    M_inv = np.linalg.inv(M)
    b = (p - V0) @ M_inv.T  # (N, 2) gives [b1, b2]
    b0 = 1.0 - b[:, 0] - b[:, 1]
    return np.column_stack([b0, b[:, 0], b[:, 1]])


def bary_to_points(b):
    """Convert barycentric coords to Cartesian points."""
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    x = b1 * 1.0 + b2 * 0.5
    y = b2 * (SQRT3 / 2.0)
    return np.column_stack([x, y])


def compute_areas(b):
    """Compute areas of all triangles formed by 11 points in barycentric coords.
    
    Area of triangle with barycentric coords can be computed via determinant.
    For points with barycentric (b0,b1,b2), the area relative to the reference triangle
    is |det of the 3x3 matrix with rows being the barycentric coords|.
    """
    # b shape: (N, 3)
    i, j, k = TRI_IDX[:, 0], TRI_IDX[:, 1], TRI_IDX[:, 2]
    # Build 3x3 matrices for each triangle
    # det = b0_i*(b1_j*b2_k - b2_j*b1_k) - b1_i*(b0_j*b2_k - b2_j*b0_k) + b2_i*(b0_j*b1_k - b1_j*b0_k)
    bi, bj, bk = b[i], b[j], b[k]
    det = (bi[:, 0] * (bj[:, 1] * bk[:, 2] - bj[:, 2] * bk[:, 1])
           - bi[:, 1] * (bj[:, 0] * bk[:, 2] - bj[:, 2] * bk[:, 0])
           + bi[:, 2] * (bj[:, 0] * bk[:, 1] - bj[:, 1] * bk[:, 0]))
    return np.abs(det) * TRI_AREA


def soft_min(areas, beta):
    """Soft minimum: -log(sum(exp(-beta*areas)))/beta approximates min(areas) for large beta."""
    # Shift for numerical stability
    a_min = np.min(areas)
    shifted = areas - a_min
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return a_min - log_sum / beta


def objective(b, beta):
    """Objective to minimize: -soft_min of triangle areas."""
    areas = compute_areas(b)
    return -soft_min(areas, beta)


def is_valid_bary(b):
    """Check if all barycentric coordinates are within [0, 1] and sum to 1."""
    if b.shape != (N, 3):
        return False
    if np.any(b < -1e-6) or np.any(b > 1.0 + 1e-6):
        return False
    if np.any(np.abs(b.sum(axis=1) - 1.0) > 1e-4):
        return False
    return True


def project_to_feasible(b):
    """Project barycentric coords to be feasible (clip and renormalize)."""
    b = np.clip(b, 0, 1)
    sums = b.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-10, 1.0, sums)
    b = b / sums
    return b


def random_bary_init(rng):
    """Generate random feasible barycentric coordinates."""
    b = rng.random((N, 3))
    b = b / b.sum(axis=1, keepdims=True)
    return b


def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill
    
    best_score = -np.inf
    best_b = None
    
    # Try multiple restarts with different beta schedules
    rng = np.random.RandomState(42)
    
    # Beta schedule: start low for smooth landscape, increase for sharp min
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > DEADLINE:
            break
        
        b = random_bary_init(rng)
        
        # Multi-stage optimization with escalating beta
        for beta in beta_schedule:
            if time.time() > DEADLINE:
                break
            
            try:
                res = minimize(
                    lambda x: objective(x, beta),
                    b,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                b = project_to_feasible(res.x)
            except Exception:
                pass
        
        # Evaluate final result
        if is_valid_bary(b):
            areas = compute_areas(b)
            min_area = np.min(areas)
            if min_area > best_score:
                best_score = min_area
                best_b = b.copy()
        
        # If we have a good result, do a final refinement with high beta
        if best_b is not None and time.time() < DEADLINE - 10:
            try:
                res = minimize(
                    lambda x: objective(x, 2000.0),
                    best_b,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                b_refined = project_to_feasible(res.x)
                if is_valid_bary(b_refined):
                    areas = compute_areas(b_refined)
                    min_area = np.min(areas)
                    if min_area > best_score:
                        best_score = min_area
                        best_b = b_refined.copy()
            except Exception:
                pass
    
    # If no valid result found, use a fallback
    if best_b is None:
        # Fallback: place points at vertices and midpoints
        b = np.zeros((N, 3))
        # 3 vertices
        b[0] = [1, 0, 0]
        b[1] = [0, 1, 0]
        b[2] = [0, 0, 1]
        # 3 edge midpoints
        b[3] = [0.5, 0.5, 0]
        b[4] = [0.5, 0, 0.5]
        b[5] = [0, 0.5, 0.5]
        # 3 more interior points
        b[6] = [1/3, 1/3, 1/3]
        b[7] = [0.4, 0.3, 0.3]
        b[8] = [0.3, 0.4, 0.3]
        b[9] = [0.3, 0.3, 0.4]
        b[10] = [0.2, 0.4, 0.4]
        best_b = b
    
    # Convert to Cartesian and return
    points = bary_to_points(best_b)
    
    # Ensure points are within the triangle (clip to feasible region)
    # Verify each point is inside the triangle
    for i in range(N):
        x, y = points[i]
        # Check if point is inside the equilateral triangle
        # Using barycentric check
        bx = (V1[0] - V0[0]) * (y - V0[1]) - (V1[1] - V0[1]) * (x - V0[0])
        by = (V2[0] - V0[0]) * (y - V0[1]) - (V2[1] - V0[1]) * (x - V0[0])
        bz = (V0[0] - V1[0]) * (y - V1[1]) - (V0[1] - V1[1]) * (x - V1[0])
        # Normalize by triangle area * 2
        norm = 2 * TRI_AREA
        b0 = bx / norm
        b1 = by / norm
        b2 = bz / norm
        if b0 < 0 or b1 < 0 or b2 < 0:
            # Project back
            b[i] = np.clip(b[i], 0, 1)
            b[i] = b[i] / b[i].sum()
            points[i] = bary_to_points(b[i])
    
    return points

