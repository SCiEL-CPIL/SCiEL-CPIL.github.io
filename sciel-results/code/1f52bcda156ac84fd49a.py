"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import linprog, nnls

N = 21

# EVOLVE-BLOCK-START

def _pairwise_distances(centers):
    """Compute pairwise distances for n circles. Returns (n, n) matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def _validate(centers, radii):
    """Check if circles are valid (non-overlapping, within bounds)."""
    n = len(centers)
    if np.any(centers[:, 0] < -1e-6) or np.any(centers[:, 0] > 1 + 1e-6):
        return False
    if np.any(centers[:, 1] < -1e-6) or np.any(centers[:, 1] > 1 + 1e-6):
        return False
    if np.any(radii < -1e-6):
        return False
    d = _pairwise_distances(centers)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < radii[i] + radii[j] - 1e-9:
                return False
    return True

def _solve_qp(c, A_ub, b_ub, A_eq, b_eq, x0, bounds):
    """
    Solve min c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds.
    Uses scipy linprog with interior-point method.
    Returns (x, success, message).
    """
    from scipy.optimize import linprog
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-9})
        if res.success:
            return res.x, True, res.message
        else:
            return x0, False, res.message
    except Exception:
        return x0, False, "exception"

def _scr_pack(w, h, n_circles=21, max_sweeps=50, tol=1e-8):
    """
    Successive Convex Relaxation for circle packing.
    Variables: x = [x_1,...,x_n, y_1,...,y_n, r_1,...,r_n]
    Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    Linearized non-overlap: for pair (i,j), at linearization point (xi,yi,ri),(xj,yj,rj):
    dist_ij * (direction) >= ri + rj
    where direction is the unit vector from i to j at the linearization point.
    
    Boundary constraints: xi - ri >= 0, xi + ri <= w, yi - ri >= 0, yi + ri <= h
    """
    n = n_circles
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                cells.append(((i + 0.5) / side * w, (j + 0.5) / side * h, 0.4 / side * min(w, h)))
    
    # Ensure we have exactly n circles
    while len(cells) < n:
        cells.append((w / 2, h / 2, 0.01))
    
    centers = np.array([(c[0], c[1]) for c in cells[:n]])
    radii = np.array([c[2] for c in cells[:n]])
    
    # Clip initial radii to be feasible
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
        radii[i] = max(radii[i], 1e-6)
    
    best_x = None
    best_obj = -np.inf
    
    for sweep in range(max_sweeps):
        # Build linearized constraints
        # Variables: [x_1..x_n, y_1..y_n, r_1..r_n], total 3n vars
        dim = 3 * n
        
        # Objective: minimize -sum(r_i)
        c = np.zeros(dim)
        c[2*n:] = -1.0  # minimize -sum(r)
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_ub_rows = []
        b_ub_vals = []
        
        # Boundary constraints:
        # x_i - r_i >= 0  =>  -x_i + r_i <= 0
        # x_i + r_i <= w  =>  x_i + r_i <= w
        # y_i - r_i >= 0  =>  -y_i + r_i <= 0
        # y_i + r_i <= h  =>  y_i + r_i <= h
        for i in range(n):
            # -x_i + r_i <= 0
            row = np.zeros(dim)
            row[i] = -1.0
            row[2*n + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
            # x_i + r_i <= w
            row = np.zeros(dim)
            row[i] = 1.0
            row[2*n + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(w)
            
            # -y_i + r_i <= 0
            row = np.zeros(dim)
            row[n + i] = -1.0
            row[2*n + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
            # y_i + r_i <= h
            row = np.zeros(dim)
            row[n + i] = 1.0
            row[2*n + i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(h)
        
        # Linearized non-overlap constraints
        # For pair (i, j), at current positions:
        # dx = x_j - x_i, dy = y_j - y_i, dist = sqrt(dx^2 + dy^2)
        # Linearized: dist * (unit_x * (x_j - x_i) + unit_y * (y_j - y_i)) >= r_i + r_j
        # => -unit_x * x_j + unit_x * x_i - unit_y * y_j + unit_y * y_i - r_i - r_j <= -dist
        # Wait, let me redo this.
        # The constraint is: ||p_j - p_i|| >= r_i + r_j
        # Linearize around current point: 
        # grad_p_i (||p_j - p_i||) = -(p_j - p_i)/||p_j - p_i||
        # grad_p_j (||p_j - p_i||) = (p_j - p_i)/||p_j - p_i||
        # grad_{r_i} (||p_j - p_i|| - r_i - r_j) = -1
        # grad_{r_j} (||p_j - p_i|| - r_i - r_j) = -1
        # So linearized: dist + grad_p_i . (p_i - p_i0) + grad_p_j . (p_j - p_j0) - (r_i - r_i0) - (r_j - r_j0) >= 0
        # => -unit_x * (x_i - x_i0) + unit_x * (x_j - x_j0) - unit_y * (y_i - y_i0) + unit_y * (y_j - y_j0) - (r_i - r_i0) - (r_j - r_j0) >= -dist
        # => -unit_x * x_i + unit_x * x_j - unit_y * y_i + unit_y * y_j - r_i - r_j >= -dist - unit_x*(x_i0 - x_j0) - unit_y*(y_i0 - y_j0) - (r_i0 + r_j0 - dist)
        # Hmm, let me be more careful.
        # f(p_i, p_j, r_i, r_j) = ||p_j - p_i|| - r_i - r_j >= 0
        # Linearize at (p_i0, p_j0, r_i0, r_j0):
        # f0 + ∇f . Δ = 0 where f0 = dist0 - r_i0 - r_j0
        # ∇_{p_i} f = -(p_j0 - p_i0)/dist0 = -u
        # ∇_{p_j} f = (p_j0 - p_i0)/dist0 = u
        # ∇_{r_i} f = -1, ∇_{r_j} f = -1
        # So: f0 - u . (p_i - p_i0) + u . (p_j - p_j0) - (r_i - r_i0) - (r_j - r_j0) >= 0
        # => -u_x * x_i + u_x * x_j - u_y * y_i + u_y * y_j - r_i - r_j >= -f0 - u_x*(x_i0 - x_j0) - u_y*(y_i0 - y_j0) - (r_i0 + r_j0 - dist0)
        # Actually simpler: 
        # f0 - u . (p_i - p_i0) + u . (p_j - p_j0) - (r_i - r_i0) - (r_j - r_j0) >= 0
        # f0 = dist0 - r_i0 - r_j0
        # -u . (p_i - p_i0) = -u_x*(x_i - x_i0) - u_y*(y_i - y_i0)
        # u . (p_j - p_j0) = u_x*(x_j - x_j0) + u_y*(y_j - y_j0)
        # So: dist0 - r_i0 - r_j0 - u_x*x_i + u_x*x_i0 - u_y*y_i + u_y*y_i0 + u_x*x_j - u_x*x_j0 + u_y*y_j - u_y*y_j0 - r_i + r_i0 - r_j + r_j0 >= 0
        # => -u_x*x_i - u_y*y_i + u_x*x_j + u_y*y_j - r_i - r_j >= -dist0 + u_x*(x_i - x_j) + u_y*(y_i - y_j) - (r_i0 + r_j0 - dist0)
        # Wait, I'm overcomplicating. Let me just write:
        # -u_x * x_i - u_y * y_i + u_x * x_j + u_y * y_j - r_i - r_j <= dist0 - u_x*(x_i0 - x_j0) - u_y*(y_i0 - y_j0) - (r_i0 + r_j0 - dist0)
        # No... let me just use the standard linearization:
        # The tangent plane to the feasible region at the current point.
        # For pair (i,j): 
        # If dist0 > 0:
        #   constraint: u . (p_j - p_i) - r_i - r_j >= -dist0 + (r_i0 + r_j0 - dist0)
        #   where u = (p_j0 - p_i0)/dist0
        #   => -u_x*x_i - u_y*y_i + u_x*x_j + u_y*y_j - r_i - r_j >= -(dist0 - r_i0 - r_j0)
        #   => u_x*x_i + u_y*y_i - u_x*x_j - u_y*y_j + r_i + r_j <= dist0 - r_i0 - r_j0
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist0 = np.sqrt(dx * dx + dy * dy)
                if dist0 < 1e-12:
                    # Degenerate: use a small perturbation
                    u_x, u_y = 1.0, 0.0
                    dist0 = 1e-12
                else:
                    u_x = dx / dist0
                    u_y = dy / dist0
                
                # Linearized constraint: u . (p_j - p_i) - r_i - r_j >= -(dist0 - r_i0 - r_j0)
                # => u_x*(x_j - x_i) + u_y*(y_j - y_i) - r_i - r_j >= -(dist0 - radii[i] - radii[j])
                # => -u_x*x_i - u_y*y_i + u_x*x_j + u_y*y_j - r_i - r_j >= -(dist0 - radii[i] - radii[j])
                # For A_ub x <= b_ub: multiply by -1
                # => u_x*x_i + u_y*y_i - u_x*x_j - u_y*y_j + r_i + r_j <= dist0 - radii[i] - radii[j]
                
                row = np.zeros(dim)
                row[i] = u_x
                row[n + i] = u_y
                row[j] = -u_x
                row[n + j] = -u_y
                row[2*n + i] = 1.0
                row[2*n + j] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(dist0 - radii[i] - radii[j])
        
        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_vals)
        else:
            A_ub = np.empty((0, dim))
            b_ub = np.empty(0)
        
        # No equality constraints
        A_eq = None
        b_eq = None
        
        # Bounds: x_i in [0, w], y_i in [0, h], r_i in [1e-8, min(w,h)]
        r_max = min(w, h) / 2.0
        bounds = []
        for i in range(n):
            bounds.append((0, w))
        for i in range(n):
            bounds.append((0, h))
        for i in range(n):
            bounds.append((1e-8, r_max))
        
        # Current solution
        x_curr = np.concatenate([centers[:, 0], centers[:, 1], radii])
        
        x_new, success, msg = _solve_qp(c, A_ub, b_ub, A_eq, b_eq, x_curr, bounds)
        
        if not success:
            # Try with slightly relaxed constraints
            A_ub_relaxed = A_ub.copy()
            b_ub_relaxed = b_ub + 1e-4
            x_new, success, msg = _solve_qp(c, A_ub_relaxed, b_ub_relaxed, A_eq, b_eq, x_curr, bounds)
        
        if not success:
            break
        
        new_centers = np.column_stack([x_new[:n], x_new[n:2*n]])
        new_radii = x_new[2*n:]
        
        new_obj = np.sum(new_radii)
        
        if new_obj > best_obj:
            best_obj = new_obj
            best_x = x_new.copy()
        
        # Check convergence
        if np.max(np.abs(x_new - x_curr)) < tol:
            break
        
        # Update for next sweep
        centers = new_centers
        radii = new_radii
    
    # Return best solution found
    if best_x is not None:
        cx = best_x[:n]
        cy = best_x[n:2*n]
        cr = best_x[2*n:]
        return np.column_stack([cx, cy, cr])
    else:
        # Fallback
        return np.column_stack([centers, radii])

def _polish(result, w, h, max_iters=100):
    """Local polishing step using gradient-based approach."""
    n = len(result)
    centers = result[:, :2].copy()
    radii = result[:, 2].copy()
    
    for _ in range(max_iters):
        improved = False
        
        # Try to increase each radius slightly
        for i in range(n):
            # Max possible radius given boundaries
            r_max_boundary = min(centers[i, 0], w - centers[i, 0], centers[i, 1], h - centers[i, 1])
            # Max possible radius given other circles
            r_max_overlap = np.inf
            for j in range(n):
                if j == i:
                    continue
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                r_max_overlap = min(r_max_overlap, (d - radii[j]) / 1.0)
            
            r_max = min(r_max_boundary, r_max_overlap)
            if r_max > radii[i] + 1e-10:
                # Try to increase
                new_r = (radii[i] + r_max) / 2
                # Check if valid
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                    if d < new_r + radii[j] - 1e-9:
                        valid = False
                        break
                if valid and new_r + 1e-9 <= centers[i,0] and new_r + 1e-9 <= w - centers[i,0] \
                   and new_r + 1e-9 <= centers[i,1] and new_r + 1e-9 <= h - centers[i,1]:
                    radii[i] = new_r
                    improved = True
        
        if not improved:
            break
    
    return np.column_stack([centers, radii])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second margin before 150s limit
    
    best_result = None
    best_score = -np.inf
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.9, 1.1, 0.75, 1.33]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Run SCR packing
        try:
            result = _scr_pack(w, h, n_circles=N, max_sweeps=40)
            
            if time.time() > deadline:
                break
            
            # Validate
            if _validate(result[:, :2], result[:, 2]):
                score = np.sum(result[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = result.copy()
            
            # Polish
            if best_result is not None and time.time() < deadline - 5:
                polished = _polish(best_result, w, h, max_iters=50)
                if _validate(polished[:, :2], polished[:, 2]):
                    p_score = np.sum(polished[:, 2])
                    if p_score > best_score:
                        best_score = p_score
                        best_result = polished.copy()
                        
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    else:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
