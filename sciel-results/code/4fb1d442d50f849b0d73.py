"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog
import itertools

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    """
    # We'll use a grid-based approach with optimization.
    # Start with a reasonable initial configuration.
    # For 21 circles, try a 5x5 grid but only use 21 cells, or other arrangements.
    
    # Strategy: Use a Delaunay-based approach.
    # 1. Generate initial centers on a grid.
    # 2. Compute Delaunay triangulation to find neighbors.
    # 3. Solve LP to maximize sum of radii subject to non-overlap and boundary constraints.
    
    # Initial configuration: 5 columns, 5 rows, but we only need 21 circles.
    # Let's try a more compact arrangement.
    
    # Try different grid layouts and pick the best one.
    best_value = -np.inf
    best_result = None
    
    # Candidate layouts: (rows, cols)
    layouts = [(5, 5), (4, 6), (3, 7), (7, 3), (6, 4), (5, 4), (4, 5)]
    
    for rows, cols in layouts:
        if rows * cols < N:
            continue
        
        # Create grid of centers
        # We'll use the first N points from a rows x cols grid
        centers = []
        for i in range(rows):
            for j in range(cols):
                centers.append([j, i])
        
        centers = np.array(centers[:N], dtype=float)
        
        # Normalize to fit in unit square initially
        # The rectangle will have width W and height H with 2(W+H) <= 4, so W+H <= 2
        # Let's parameterize by aspect ratio.
        
        # For now, let's try to optimize in a unit square and then scale.
        # Actually, let's work directly with the constraint that the bounding box has perimeter <= 4.
        
        # Let's fix the aspect ratio and optimize positions and radii.
        # Try several aspect ratios.
        
        for aspect in [1.0, 0.8, 1.25, 0.75, 1.5, 0.67]:
            # If aspect = W/H, then W = aspect * H, and 2(W + H) = 2(aspect + 1) * H <= 4
            # So H <= 2 / (aspect + 1), W <= 2 * aspect / (aspect + 1)
            
            # Let's set H = 2 / (aspect + 1) and W = 2 * aspect / (aspect + 1)
            H = 2.0 / (aspect + 1.0)
            W = 2.0 * aspect / (aspect + 1.0)
            
            # Normalize centers to [0, W] x [0, H]
            if len(centers) > 0:
                cmin = centers.min(axis=0)
                cmax = centers.max(axis=0)
                cspan = cmax - cmin
                cspan = np.where(cspan < 1e-10, 1.0, cspan)
                centers_norm = (centers - cmin) / cspan
                centers_scaled = centers_norm * np.array([W, H])
            else:
                centers_scaled = centers
            
            # Now solve LP to find optimal radii and possibly adjust positions slightly.
            # For simplicity, first just compute radii given fixed positions.
            
            # Compute pairwise distances
            n = len(centers_scaled)
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers_scaled[i] - centers_scaled[j])
                    dist[i, j] = dist[j, i] = d
            
            # Constraints:
            # r_i + r_j <= dist[i, j] for all i != j
            # r_i <= centers_scaled[i][0] (left boundary)
            # r_i <= W - centers_scaled[i][0] (right boundary)
            # r_i <= centers_scaled[i][1] (bottom boundary)
            # r_i <= H - centers_scaled[i][1] (top boundary)
            # r_i >= 0
            
            # Maximize sum of r_i
            
            # LP formulation:
            # c = -ones(n)  (since linprog minimizes)
            # A_ub, b_ub for pair constraints
            # A_ub, b_ub for boundary constraints
            
            c = -np.ones(n)
            
            A_ub = []
            b_ub = []
            
            # Pair constraints: r_i + r_j <= dist[i, j]
            for i in range(n):
                for j in range(i + 1, n):
                    row = np.zeros(n)
                    row[i] = 1
                    row[j] = 1
                    A_ub.append(row)
                    b_ub.append(dist[i, j])
            
            # Boundary constraints
            for i in range(n):
                # r_i <= centers_scaled[i][0]
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(centers_scaled[i][0])
                
                # r_i <= W - centers_scaled[i][0]
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(W - centers_scaled[i][0])
                
                # r_i <= centers_scaled[i][1]
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(centers_scaled[i][1])
                
                # r_i <= H - centers_scaled[i][1]
                row = np.zeros(n)
                row[i] = 1
                A_ub.append(row)
                b_ub.append(H - centers_scaled[i][1])
            
            if not A_ub:
                continue
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            bounds = [(0, None)] * n
            
            try:
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    radii = res.x
                    value = np.sum(radii)
                    if value > best_value:
                        best_value = value
                        result = np.zeros((n, 3))
                        result[:, :2] = centers_scaled
                        result[:, 2] = radii
                        best_result = result
            except Exception:
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
