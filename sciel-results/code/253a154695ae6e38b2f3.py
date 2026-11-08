"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can compute all areas for a fixed i by vectorizing j and k
    for i in range(n):
        pi = points[i]
        # Vectors from pi to all other points
        vecs = points - pi  # shape (n, 2)
        
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        # We need min of 0.5 * |C[j,k]| for j < k, j != i, k != i
        # Since C is antisymmetric, C[j,k] = -C[k,j]
        # We can take the upper triangle
        # Set diagonal and i-th row/col to inf to ignore them
        C_abs = np.abs(C)
        # Set i-th row and column to inf
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        # Set diagonal to inf (j==k area is 0, but we want distinct points)
        np.fill_diagonal(C_abs, float('inf'))
        
        # Find min in upper triangle
        # To be safe and fast, just find global min of C_abs (excluding i and diag)
        # Since it's symmetric, min in upper == min in lower
        current_min = np.min(C_abs)
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-9: # Early exit if degenerate
                return 0.0

    return (0.5 * min_area) / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _bary_to_cart(bary):
    """Convert barycentric coordinates (shape N, 3) to Cartesian coordinates (shape N, 2)."""
    # bary[:, 0] * V0 + bary[:, 1] * V1 + bary[:, 2] * V2
    # V0 is (0,0), so:
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * (np.sqrt(3) / 2.0)
    return np.column_stack((x, y))


def _cart_to_bary(cart):
    """Convert Cartesian coordinates (shape N, 2) to barycentric coordinates (shape N, 3)."""
    # x = u + 0.5v
    # y = (sqrt(3)/2) v
    v = 2 * cart[:, 1] / np.sqrt(3)
    u = cart[:, 0] - 0.5 * v
    w = 1 - u - v
    return np.column_stack((w, u, v))


def _objective_bary(bary_params):
    """
    Objective function for minimization: -min_area + penalty.
    bary_params shape (N, 3).
    We want to maximize min_area, so minimize -min_area.
    Penalty for points outside triangle or coincident points.
    """
    # Ensure barycentric coordinates are non-negative
    # Differential Evolution operates on bounds, so we define bounds as [0, 1] for each coordinate.
    # However, the constraint w+u+v=1 is not naturally enforced by DE bounds on [0,1]^3.
    # We must project or penalize. 
    # Better approach for DE: Use 2N parameters (u, v) with bounds [0,1] and u+v<=1 is hard.
    # Let's stick to 3N parameters but normalize them to sum to 1? 
    # If we normalize, we lose the "distance" information for DE steps? No, DE works on the vector.
    # But if we normalize inside the objective, the search space is effectively the simplex.
    # However, DE steps are linear. Normalizing after a linear step keeps it in the simplex.
    # Let's try: interpret bary_params as weights, normalize to sum 1.
    
    # Check for negative values (shouldn't happen if bounds are [0,1])
    # But DE might go out of bounds? No, DE respects bounds.
    # So bary_params are in [0, 1] for each component.
    # Normalize:
    sums = np.sum(bary_params, axis=1, keepdims=True)
    # Avoid division by zero
    sums[sums == 0] = 1.0
    bary_norm = bary_params / sums
    
    points = _bary_to_cart(bary_norm)
    
    # Penalty for points outside? 
    # Since we normalized positive weights, points are always inside the convex hull of V0, V1, V2.
    # So no boundary penalty needed for being outside.
    
    # Penalty for coincident points?
    # If two points are identical, min area is 0.
    # We can add a small penalty if min area is 0 to help gradient? 
    # DE is derivative free, so it handles 0s fine, but might get stuck.
    # Let's just return -min_area.
    
    min_area = _min_triangle_area(points)
    
    # To help DE explore, we can add a small penalty for clustering
    # But let's keep it simple first.
    return -min_area


def _generate_structured_initial_bary():
    """Generate a structured initial configuration in barycentric coords."""
    # Vertices, midpoints, centroid
    # V0: (1,0,0), V1: (0,1,0), V2: (0,0,1)
    # Mid V0V1: (0.5, 0.5, 0)
    # Mid V1V2: (0, 0.5, 0.5)
    # Mid V2V0: (0.5, 0, 0.5)
    # Centroid: (1/3, 1/3, 1/3)
    # Quarter points...
    
    bary = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [1/3, 1/3, 1/3],
        [0.25, 0.5, 0.25],
        [0.5, 0.25, 0.25],
        [0.25, 0.25, 0.5],
        [0.4, 0.4, 0.2]
    ])
    return bary


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Define bounds for Differential Evolution
    # Each point has 3 barycentric coords, each in [0, 1]
    # Total 33 parameters
    bounds = [(0.0, 1.0)] * (N * 3)
    
    # Strategy 1: Differential Evolution (Global Search)
    # DE is robust to non-smoothness and local minima.
    # We use the normalized barycentric representation.
    
    try:
        # Use a moderate population size and max iterations
        # popsize=15 means 15*33=495 individuals initially
        # maxiter=500 might be too slow? 
        # Each eval is ~1ms for N=11. 495 * 500 = 247,500 evals. ~250s. Too slow.
        # We need to reduce.
        # popsize=10, maxiter=200 -> 10*33 * 200 = 66,000 evals. ~66s. OK.
        # Let's try popsize=12, maxiter=300. 12*33*300 = 118,800. ~120s. Risky.
        # Let's do popsize=10, maxiter=200.
        
        # Update callback to check time
        def _de_callback(xk, convergence):
            if time.time() > deadline - 30:
                raise TimeoutError("DE timeout")
            return False

        result = differential_evolution(
            _objective_bary, 
            bounds, 
            seed=42,
            popsize=10,
            maxiter=200,
            tol=1e-6,
            polish=False, # We will polish manually
            callback=_de_callback,
            init='latinhypercube'
        )
        
        if result.success or result.x is not None:
            best_bary = result.x.reshape(N, 3)
            # Normalize
            sums = np.sum(best_bary, axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            best_bary_norm = best_bary / sums
            best_points_candidate = _bary_to_cart(best_bary_norm)
            score = _min_triangle_area(best_points_candidate)
            
            if score > best_score:
                best_score = score
                best_points = best_points_candidate.copy()
                
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Strategy 2: Local Polish with Nelder-Mead
    # NM is good for final refinement of the min function.
    # We need to convert best_points to a parameterization suitable for NM.
    # NM works on Cartesian coords directly, but we need to handle bounds.
    # We can use a penalty method or just project after each step.
    # Simpler: Use the barycentric coords as parameters for NM, but NM doesn't handle bounds well.
    # Let's use Cartesian coords and a penalty function for NM.
    
    if best_points is not None:
        # Define objective for NM in Cartesian space
        def _nm_objective(cart_params):
            points = cart_params.reshape(N, 2)
            # Penalty for being outside
            penalty = 0.0
            for p in points:
                if not _is_inside_triangle(p):
                    # Distance to boundary
                    x, y = p
                    # Simple penalty: squared distance to nearest edge
                    # This is approximate
                    penalty += 10.0 
            min_area = _min_triangle_area(points)
            return -min_area + penalty
        
        # Run NM on the best point found by DE
        # NM doesn't support bounds directly, so we rely on the penalty
        # Or we can stick to the barycentric space and use a constrained optimizer?
        # Let's try NM on Cartesian with penalty.
        
        try:
            nm_result = minimize(
                _nm_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if nm_result.fun < -best_score: # i.e. -min_area + penalty < -best_min_area => min_area > best_min_area (if penalty is 0)
                nm_points = nm_result.x.reshape(N, 2)
                # Check if all points are inside
                valid = True
                for p in nm_points:
                    if not _is_inside_triangle(p):
                        valid = False
                        break
                
                if valid:
                    nm_score = _min_triangle_area(nm_points)
                    if nm_score > best_score:
                        best_score = nm_score
                        best_points = nm_points.copy()
        except Exception:
            pass
    
    # Fallback if nothing found
    if best_points is None:
        # Generate a random valid set
        rng = random.Random(42)
        points = []
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                p = V0 * (1 - u - v) + V1 * u + V2 * v
                points.append(p)
        best_points = np.array(points)
        
    return best_points
# EVOLVE-BLOCK-END
