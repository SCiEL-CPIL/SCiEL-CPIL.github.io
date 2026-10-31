"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def _circle_distance(c1, c2):
    """Minimum distance between two circles (center distance minus sum of radii)."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist - c1[2] - c2[2]


def _wall_distance(c, W, H):
    """Minimum distance from circle to the four walls of rectangle [0,W]x[0,H]."""
    r = c[2]
    return min(c[0] - r, W - c[0] - r, c[1] - r, H - c[1] - r)


def _perimeter(W, H):
    return 2.0 * (W + H)


def construct():
    """
    Pack 21 circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    
    Uses force-directed relaxation with a contact graph approach:
    - Initialize with a grid-like configuration
    - Iteratively push circles apart based on overlaps
    - Adjust rectangle dimensions to maintain perimeter constraint
    - Maximize sum of radii
    """
    # Start with a reasonable initial configuration
    # Try different aspect ratios for the rectangle
    best_result = None
    best_score = -np.inf
    
    # Perimeter = 4, so W + H = 2
    # Try various aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    
    for ar in aspect_ratios:
        # W/H = ar, W + H = 2
        # W = ar*H, so H*(ar+1) = 2, H = 2/(ar+1), W = 2*ar/(ar+1)
        H = 2.0 / (ar + 1.0)
        W = 2.0 * ar / (ar + 1.0)
        
        # Initialize circles in a grid pattern
        # 21 circles: try 5x5 grid with some removed, or 4x6, etc.
        # Let's use a 5x5 grid (25 positions) and pick 21
        cols = 5
        rows = 5
        total = cols * rows
        
        # Initial positions: grid centers
        xs = np.linspace(W * 0.1, W * 0.9, cols)
        ys = np.linspace(H * 0.1, H * 0.9, rows)
        
        # Pick 21 out of 25 positions - remove corners or edges to optimize
        # For now, use all 25 and then we'll adjust
        # Actually, let's use exactly 21 by using a 4x6 grid minus 3, or 5x5 minus 4
        
        # Use 5x5 grid, remove 4 corner-adjacent positions
        positions = []
        for i in range(rows):
            for j in range(cols):
                positions.append([xs[j], ys[i]])
        
        # Take first 21 positions (this is a heuristic; we'll optimize)
        # Better: use a more symmetric selection
        # Remove 4 positions from the edges/corners
        # For 5x5, remove positions at (0,0), (0,4), (4,0), (4,4) -> corners
        # That leaves 21
        indices_to_remove = [(0,0), (0,4), (4,0), (4,4)]
        kept_indices = []
        for i in range(rows):
            for j in range(cols):
                if (i,j) not in indices_to_remove:
                    kept_indices.append((i,j))
        
        positions = []
        for (i,j) in kept_indices:
            positions.append([xs[j], ys[i]])
        
        positions = np.array(positions)
        
        # Initial radius: estimate based on grid spacing
        dx = W / cols
        dy = H / rows
        r_init = min(dx, dy) * 0.45
        
        # State: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        n = N
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = r_init
        
        # Objective: maximize sum of radii, subject to:
        # 1. Circles don't overlap: dist(ci, cj) >= ri + rj
        # 2. Circles inside rectangle: ri <= xi <= W-ri, ri <= yi <= H-ri
        # 3. Perimeter constraint: 2(W+H) <= 4 (fixed here since W,H are fixed)
        
        # Use SLSQP with constraints
        def objective(x):
            # Maximize sum of radii = minimize negative sum
            radii = x[2::3]
            return -np.sum(radii)
        
        # Constraints: each pair of circles must not overlap
        # And each circle must be within bounds
        constraints = []
        
        # Circle-circle non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                def constraint_func(x, i=i, j=j):
                    xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                    xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    return dist - ri - rj
                constraints.append({'type': 'ineq', 'fun': constraint_func})
        
        # Wall constraints
        for i in range(n):
            # Left wall
            def left_constraint(x, i=i):
                return x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': left_constraint})
            
            # Right wall
            def right_constraint(x, i=i):
                return W - x[3*i] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': right_constraint})
            
            # Bottom wall
            def bottom_constraint(x, i=i):
                return x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bottom_constraint})
            
            # Top wall
            def top_constraint(x, i=i):
                return H - x[3*i+1] - x[3*i+2]
            constraints.append({'type': 'ineq', 'fun': top_constraint})
        
        # Bounds: radii must be positive, positions within some range
        bounds = []
        for i in range(n):
            bounds.append((0.0, W))      # x
            bounds.append((0.0, H))      # y
            bounds.append((1e-6, min(W,H)/2))  # radius
        
        # Run optimization
        try:
            result = minimize(
                objective, x0, method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success:
                x_opt = result.x
                radii = x_opt[2::3]
                score = np.sum(radii)
                
                if score > best_score:
                    best_score = score
                    best_result = x_opt.copy()
                    
        except Exception:
            continue
    
    # If optimization failed for all aspect ratios, fall back to simple grid
    if best_result is None:
        # Simple fallback: equal circles in a 5x5 grid minus 4
        side = 5
        W = 1.0
        H = 1.0  # Square, perimeter = 4
        r = 0.5 / side - 1e-3
        cells = []
        indices_to_remove = {(0,0), (0,4), (4,0), (4,4)}
        count = 0
        for i in range(side):
            for j in range(side):
                if (i,j) not in indices_to_remove:
                    cells.append([((j + 0.5) / side) * W, ((i + 0.5) / side) * H, r])
                    count += 1
                    if count == N:
                        break
            if count == N:
                break
        return np.array(cells[:N])
    
    # Extract final result
    result_array = np.zeros((N, 3))
    for i in range(N):
        result_array[i, 0] = best_result[3*i]
        result_array[i, 1] = best_result[3*i+1]
        result_array[i, 2] = best_result[3*i+2]
    
    return result_array
# EVOLVE-BLOCK-END
