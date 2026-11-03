import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit

def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _init_hexagonal():
    """
    Generate a hexagonal packing initialization.
    This is a strong starting point for circle packing as it mimics the optimal
    infinite plane density.
    """
    best_pts = None
    best_r = -1
    
    # Try different scales and offsets to find the one that fits best in the square
    for scale in [0.14, 0.15, 0.16, 0.17, 0.18]:
        for dx in np.linspace(-0.05, 0.05, 5):
            for dy in np.linspace(-0.05, 0.05, 5):
                pts = []
                row_height = scale * np.sqrt(3) / 2
                # Generate a grid larger than needed
                for i in range(-6, 7):
                    for j in range(-6, 7):
                        x = j * scale + (i % 2) * scale / 2 + dx
                        y = i * row_height + dy
                        if abs(x) <= 0.49 and abs(y) <= 0.49:
                            pts.append([x, y])
                
                pts = np.array(pts)
                if len(pts) < N:
                    continue
                
                # Select N points: prefer those closest to the center to maintain symmetry/density
                dists = np.linalg.norm(pts, axis=1)
                idx = np.argsort(dists)[:N]
                pts_sel = pts[idx]
                
                # Ensure no overlaps (should be fine for hex grid if scale is right)
                # Check validity
                if np.any(np.abs(pts_sel) > 0.5):
                    continue
                    
                r = _true_score(pts_sel)
                if r > best_r:
                    best_r = r
                    best_pts = pts_sel.copy()
                    
    if best_pts is not None:
        return best_pts
    
    # Fallback: simple grid
    xs = (np.arange(8) + 0.5) / 8 - 0.5
    ys = (np.arange(7) + 0.5) / 7 - 0.5
    grid = np.array([(x, y) for y in ys for x in xs])
    # Take first 50
    return grid[:N]

def _force_relaxation(centers, time_budget=10.0):
    """
    Deterministic Force-Directed Relaxation.
    Uses a repulsive force that decays with distance squared to push circles apart,
    and an attractive force towards the center to keep them inside.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # Parameters
    repulsive_strength = 0.05
    attractive_strength = 0.01
    damping = 0.9
    velocity = np.zeros_like(centers)
    
    # Initial velocity is zero
    # We iterate until time budget is exhausted or convergence
    
    while time.time() - start_time < time_budget:
        forces = np.zeros_like(centers)
        
        # 1. Repulsive forces between pairs
        # O(N^2) is fine for N=50
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # If overlapping, add random jitter to break symmetry
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1
                    continue
                
                dist = np.sqrt(dist_sq)
                # Force magnitude ~ 1/d^2
                # We want to maximize minimum distance.
                # A common choice is F = k / d^2 * direction
                force_mag = repulsive_strength / dist_sq
                force_dir = diff / dist
                forces[i] += force_mag * force_dir
                forces[j] -= force_mag * force_dir
                
        # 2. Attractive forces to center (to keep them in the square)
        # Also add a soft boundary force
        for i in range(N):
            # Attraction to origin
            forces[i] -= attractive_strength * centers[i]
            
            # Boundary repulsion
            # If x > 0.5 - r_est, push left. But we don't know r_est yet.
            # Simple heuristic: if close to boundary, push inward.
            # Let's use a soft wall at 0.45
            wall = 0.45
            for k in range(2):
                if centers[i][k] > wall:
                    forces[i][k] -= 0.5 * (centers[i][k] - wall)
                elif centers[i][k] < -wall:
                    forces[i][k] += 0.5 * (centers[i][k] + wall)
            
        # Update velocity and position
        velocity = velocity * damping + forces
        centers += velocity
        
        # Clip to square strictly
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check convergence: if max velocity is small
        if np.max(np.linalg.norm(velocity, axis=1)) < 1e-6:
            break
            
    return centers

def _slsqp_polish(centers, time_budget=10.0):
    """
    Polish the configuration using SLSQP to maximize the minimum radius.
    We formulate this as minimizing -r, where r is the minimum of all constraints.
    To make it differentiable, we use a soft-min approximation or simply
    minimize the negative of the minimum distance directly (non-smooth but SLSQP can handle it sometimes).
    
    A better approach: Maximize r subject to d_ij >= 2r and boundary_i >= r.
    This is a non-linear programming problem.
    Variables: x_1...x_N, y_1...y_N, r
    Objective: Maximize r (or Minimize -r)
    Constraints:
    1. x_i^2 + y_i^2 ... no, distance constraints.
    For all i < j: (x_i - x_j)^2 + (y_i - y_j)^2 >= (2r)^2
    For all i: 0.5 - |x_i| >= r, 0.5 - |y_i| >= r
    
    SLSQP requires smooth constraints. The distance constraint is smooth.
    The boundary constraint 0.5 - |x| >= r is non-smooth at x=0.
    We can split x into positive and negative parts or just rely on the fact that
    optimal packings usually don't have centers on the axes (symmetry breaking).
    Or we can use the squared distance for pairs, and for boundaries:
    x_i <= 0.5 - r
    -x_i <= 0.5 - r
    y_i <= 0.5 - r
    -y_i <= 0.5 - r
    
    This is a set of linear inequalities for fixed r, but r is a variable.
    So: x_i + r <= 0.5
    -x_i + r <= 0.5
    etc.
    These are linear in (x_i, r).
    
    The pair constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (2r)^2 >= 0
    This is non-linear (quadratic).
    
    Let's define the vector v = [x_1, y_1, ..., x_N, y_N, r]
    Objective: -r (minimize)
    
    Constraints (h(v) >= 0 for SLSQP 'ineq' type which is h(v) >= 0? No, SLSQP uses h(v) >= 0 for 'ineq' if defined as such, usually scipy uses h(v) >= 0 for 'ineq' type? 
    Actually scipy.optimize.minimize SLSQP:
    type='ineq': h(v) >= 0
    type='eq': h(v) = 0
    
    So for pairs: (x_i - x_j)^2 + (y_i - y_j)^2 - 4r^2 >= 0
    For boundaries: 0.5 - x_i - r >= 0  => x_i + r - 0.5 <= 0 => 0.5 - x_i - r >= 0
    Similarly for -x_i, y_i, -y_i.
    Also r >= 0.
    """
    start_time = time.time()
    
    def objective(v):
        return -v[-1] # Minimize -r
    
    def constraints(v):
        cons = []
        r = v[-1]
        centers = v[:-1].reshape(N, 2)
        
        # Boundary constraints
        # 0.5 - x_i - r >= 0
        # 0.5 + x_i - r >= 0
        # 0.5 - y_i - r >= 0
        # 0.5 + y_i - r >= 0
        
        # We can vectorize these if we build the Jacobian, but for N=50,
        # returning a list of dictionaries might be slow.
        # Let's return a single array of constraint values.
        # SLSQP accepts a list of dicts or a single array if 'fun' returns array.
        # Let's return a list of dicts for clarity, though it's slower.
        # Better: return a single array and provide Jacobian? 
        # For simplicity and robustness, let's use the list of dicts first.
        
        # Actually, for speed, let's compute all pair constraints in one array.
        # Number of pairs: N*(N-1)/2 = 1225
        # Number of boundary constraints: 4*N = 200
        # Total: 1425 constraints.
        
        cons_vals = np.zeros(N*(N-1)//2 + 4*N + 1)
        idx = 0
        
        # Pair constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                cons_vals[idx] = dx*dx + dy*dy - 4*r*r
                idx += 1
        
        # Boundary constraints
        for i in range(N):
            x = centers[i, 0]
            y = centers[i, 1]
            cons_vals[idx] = 0.5 - x - r
            idx += 1
            cons_vals[idx] = 0.5 + x - r
            idx += 1
            cons_vals[idx] = 0.5 - y - r
            idx += 1
            cons_vals[idx] = 0.5 + y - r
            idx += 1
            
        # r >= 0
        cons_vals[idx] = r
        
        return cons_vals

    def jac(v):
        # Jacobian of constraints w.r.t v
        # Shape: (num_constraints, 2N + 1)
        num_cons = N*(N-1)//2 + 4*N + 1
        J = np.zeros((num_cons, 2*N + 1))
        r = v[-1]
        centers = v[:-1].reshape(N, 2)
        
        idx = 0
        # Pair constraints: d_ij^2 - 4r^2
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                
                # d/dx_i (dx^2 + dy^2) = 2*dx
                J[idx, 2*i] = 2*dx
                # d/dy_i = 2*dy
                J[idx, 2*i+1] = 2*dy
                # d/dx_j = -2*dx
                J[idx, 2*j] = -2*dx
                # d/dy_j = -2*dy
                J[idx, 2*j+1] = -2*dy
                # d/dr = -8*r
                J[idx, -1] = -8*r
                
                idx += 1
        
        # Boundary constraints
        for i in range(N):
            # 0.5 - x_i - r
            J[idx, 2*i] = -1
            J[idx, -1] = -1
            idx += 1
            
            # 0.5 + x_i - r
            J[idx, 2*i] = 1
            J[idx, -1] = -1
            idx += 1
            
            # 0.5 - y_i - r
            J[idx, 2*i+1] = -1
            J[idx, -1] = -1
            idx += 1
            
            # 0.5 + y_i - r
            J[idx, 2*i+1] = 1
            J[idx, -1] = -1
            idx += 1
            
        # r >= 0
        J[idx, -1] = 1
        
        return J

    # Initial guess
    v0 = np.zeros(2*N + 1)
    v0[:-1] = centers.flatten()
    v0[-1] = _true_score(centers) * 0.9 # Start slightly below current best to be feasible
    
    # Bounds
    # x, y in [-0.5, 0.5]
    # r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2*N) + [(0, 0.5)]
    
    # Run SLSQP
    try:
        res = minimize(objective, v0, method='SLSQP', jac=jac, 
                       bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False, 
                                'eps': 1e-4})
        
        if res.success or res.fun < objective(v0):
            new_centers = res.x[:-1].reshape(N, 2)
            new_r = res.x[-1]
            
            # Ensure validity
            if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                # Check if it's actually better
                true_r = _true_score(new_centers)
                if true_r > _true_score(centers):
                    return new_centers
    except Exception:
        pass
        
    return centers

def construct():
    """Main entry point combining FDR and SLSQP polish."""
    start_time = time.time()
    np.random.seed(42)

    # 1. Generate initial configuration
    init = _init_hexagonal()
    best_centers = init.copy()
    best_score = _true_score(init)
    
    # 2. Force Directed Relaxation
    # Allocate time based on remaining budget
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    t_fdr = min(remaining * 0.4, 20.0) # Use 40% of time or max 20s
    
    if t_fdr > 1:
        relaxed = _force_relaxation(init, time_budget=t_fdr)
        score_relaxed = _true_score(relaxed)
        if score_relaxed > best_score:
            best_score = score_relaxed
            best_centers = relaxed.copy()
        else:
            # If FDR didn't improve, maybe try a shorter FDR or just proceed?
            # Let's try a shorter FDR on the best so far
            pass

    # 3. SLSQP Polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    t_slsqp = min(remaining * 0.5, 30.0) # Use 50% of remaining or max 30s
    
    if t_slsqp > 5:
        polished = _slsqp_polish(best_centers, time_budget=t_slsqp)
        score_polished = _true_score(polished)
        if score_polished > best_score:
            best_score = score_polished
            best_centers = polished.copy()

    # 4. Final quick FDR + SLSQP if time permits
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    if remaining > 10:
        # Quick FDR
        t_fdr2 = min(remaining * 0.3, 5.0)
        if t_fdr2 > 1:
            relaxed2 = _force_relaxation(best_centers, time_budget=t_fdr2)
            score_r2 = _true_score(relaxed2)
            if score_r2 > best_score:
                best_score = score_r2
                best_centers = relaxed2.copy()
        
        # Quick SLSQP
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining > 5:
            t_slsqp2 = min(remaining * 0.8, 10.0)
            if t_slsqp2 > 1:
                polished2 = _slsqp_polish(best_centers, time_budget=t_slsqp2)
                score_p2 = _true_score(polished2)
                if score_p2 > best_score:
                    best_score = score_p2
                    best_centers = polished2.copy()

    # Ensure validity
    if best_centers is None:
        best_centers = _init_hexagonal()
    best_centers = np.clip(best_centers, -0.5, 0.5)

    return best_centers

if __name__ == "__main__":
    # Quick test
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
