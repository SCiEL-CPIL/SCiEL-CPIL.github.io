import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import sys

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

def _get_active_pairs(centers, current_r):
    """
    Identify the active constraints (pairs closer than 2*r + epsilon 
    and points closer than r + epsilon to boundary).
    Using Delaunay is overkill for N=50 if we just want pairs < 2r, 
    but for N=50, O(N^2) is fine. 
    However, the 'different strategy' is to use the structure of the packing.
    Let's stick to a robust O(N^2) but optimized with NumPy, 
    and focus the 'different strategy' on the Optimization Algorithm itself:
    We will use a custom 'Active Set' Newton-like method or SLSQP 
    restricted to the active constraints, and use Simulated Annealing 
    for global exploration.
    """
    # For N=50, O(N^2) is ~2500 ops, very fast.
    # The bottleneck in the current program is likely the SLSQP iterations 
    # with 1225 constraints and the slow expansion.
    # We will skip the slow expansion and go straight to high-level optimization.
    pass

def _slsqp_active(centers, r_init, time_budget=10.0):
    """
    SLSQP optimization on the epigraph formulation.
    Maximize t subject to:
    1. 0.5 - |c_i| >= t  (Boundary)
    2. ||c_i - c_j|| >= 2t (Pairs)
    
    We provide the Jacobian analytically to speed up SLSQP significantly.
    """
    start_time = time.time()
    
    def objective(x):
        return -x[-1]
    
    def gradient(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
        
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    # Constraints: g(x) >= 0
    # We define them as:
    # For each i, k: 0.5 - x[2i+k] - t >= 0  (if x[2i+k] >= 0) 
    # Actually, abs makes it non-smooth. SLSQP can handle it but slowly.
    # Let's use the smooth approximation for the boundary in the constraint?
    # No, SLSQP is good with inequalities. 
    # The non-smoothness of abs is at 0. If points are near 0, it's an issue.
    # But usually points are away from the center.
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        
        # Boundary: 0.5 - |c_ik| - t >= 0
        # We can split into 4 constraints per point? Or just use abs.
        # SLSQP handles abs poorly if derivative is undefined.
        # Let's use a smooth max for boundary distance in the constraint?
        # dist_b = 0.5 - |c|
        # Let's just use the raw abs. It's usually fine unless c=0.
        
        cons_b = 0.5 - np.abs(c) - t  # Shape (N, 2)
        
        # Pairs: ||c_i - c_j|| - 2t >= 0
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cons_p = dists - 2 * t
        
        return np.concatenate([cons_b.ravel(), cons_p])
        
    # Jacobian of constraints
    # This is the key speedup.
    # d/dx[2i+k] of (0.5 - |x[2i+k]| - t) = -sign(x[2i+k])
    # d/dt of (0.5 - |x[2i+k]| - t) = -1
    # d/dx[2i] of (||c_i - c_j|| - 2t) = (x[2i] - x[2j]) / dist
    # d/dt of (||c_i - c_j|| - 2t) = -2
    
    def jacobian(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        
        J = np.zeros((2*N + n_pairs, 2*N + 1))
        
        # Boundary part (first 2*N rows)
        # For point i, coord k:
        # J[2i+k, 2i+k] = -sign(c[i,k])
        # J[2i+k, -1] = -1
        for i in range(N):
            for k in range(2):
                idx = 2*i + k
                # Handle sign(0) as 1 to avoid NaNs, though rare
                s = np.sign(c[i, k])
                if s == 0: s = 1.0
                J[idx, idx] = -s
                J[idx, -1] = -1.0
        
        # Pair part (next n_pairs rows)
        # For pair (i, j) at index p:
        # J[2*N + p, 2*i] = (c[i,0] - c[j,0]) / dist
        # J[2*N + p, 2*i+1] = (c[i,1] - c[j,1]) / dist
        # J[2*N + p, 2*j] = -(c[i,0] - c[j,0]) / dist
        # J[2*N + p, 2*j+1] = -(c[i,1] - c[j,1]) / dist
        # J[2*N + p, -1] = -2
        
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        inv_dists = 1.0 / dists_safe
        
        # Vectorized assignment is tricky, let's loop for clarity as N is small
        for p in range(n_pairs):
            i = iu[0][p]
            j = iu[1][p]
            d = diffs[p]
            inv_d = inv_dists[p]
            
            row = 2*N + p
            J[row, 2*i] = d[0] * inv_d
            J[row, 2*i+1] = d[1] * inv_d
            J[row, 2*j] = -d[0] * inv_d
            J[row, 2*j+1] = -d[1] * inv_d
            J[row, -1] = -2.0
            
        return J
        
    t0 = _true_score(centers)
    x0 = np.concatenate([centers.ravel(), [t0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian}],
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        
        if result.success:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_t = result.x[-1]
            true_t = _true_score(new_centers)
            if true_t >= t0 - 1e-9:
                return new_centers, true_t
    except Exception:
        pass
        
    return centers, t0

def _simulated_annealing(centers, time_budget=20.0):
    """
    Simulated Annealing to escape local minima.
    Move random points by small random amounts.
    Accept if score improves, or with probability exp(-delta/T).
    """
    start_time = time.time()
    current_centers = centers.copy()
    current_score = _true_score(current_centers)
    best_centers = current_centers.copy()
    best_score = current_score
    
    T = 0.001  # Initial temperature
    T_min = 1e-6
    cooling = 0.99
    
    while time.time() - start_time < time_budget:
        if T < T_min:
            break
            
        # Pick a random point
        i = np.random.randint(N)
        # Random move
        move = np.random.randn(2) * 0.01
        new_centers = current_centers.copy()
        new_centers[i] += move
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        new_score = _true_score(new_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_centers = new_centers
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
                
        T *= cooling
        
    return best_centers, best_score

def _init_hexagonal():
    """
    Generate a good hexagonal packing initialization.
    """
    best_pts = None
    best_r = -1
    
    # Try different scales and offsets
    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        # Generate enough points
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.49 and abs(y) <= 0.49:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            continue
            
        # Select N points greedily: start with center, then pick furthest from existing set
        # This creates a space-filling pattern
        selected = []
        remaining = pts.copy()
        
        # Start with the point closest to center? Or random?
        # Let's try starting with the one closest to center to fill outwards
        center_dists = np.linalg.norm(remaining, axis=1)
        idx = np.argmin(center_dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        for _ in range(N - 1):
            if len(remaining) == 0:
                break
            sel = np.array(selected)
            # Calculate min distance to selected set for each remaining point
            dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            # Pick the one with max min distance (furthest from others)
            idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        if len(selected) == N:
            pts_sel = np.array(selected)
            r = _true_score(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel
                
    if best_pts is None:
        # Fallback grid
        rows, cols = 7, 8
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        # Take first N
        best_pts = grid[:N]
        
    return best_pts

def construct():
    """
    Main entry point.
    Strategy:
    1. High-quality Hexagonal Init.
    2. SLSQP with Analytic Jacobian (Fast Polish).
    3. Simulated Annealing (Global Search).
    4. SLSQP Polish again.
    5. Repeat SA/SLSQP if time permits.
    """
    start_time = time.time()
    np.random.seed(42)
    
    # 1. Initialization
    centers = _init_hexagonal()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    # 2. Initial Polish
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        centers, score = _slsqp_active(centers, best_score, time_budget=min(remaining*0.5, 15.0))
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # 3. Iterative SA + Polish Loop
    # We alternate between SA (to jump) and SLSQP (to polish)
    # SA is good for escaping, SLSQP for converging.
    
    # We will run a few rounds.
    rounds = 5
    for r in range(rounds):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
            
        # Simulated Annealing
        # Budget: 50% of remaining, capped at 20s
        sa_budget = min(remaining * 0.5, 20.0)
        if sa_budget > 5:
            centers_sa, score_sa = _simulated_annealing(best_centers, time_budget=sa_budget)
            if score_sa > best_score:
                best_score = score_sa
                best_centers = centers_sa.copy()
                
        # Polish with SLSQP
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        slsqp_budget = min(remaining * 0.4, 15.0)
        if slsqp_budget > 5:
            centers_polish, score_polish = _slsqp_active(best_centers, best_score, time_budget=slsqp_budget)
            if score_polish > best_score:
                best_score = score_polish
                best_centers = centers_polish.copy()
                
    # 4. Final Polish with tightest tolerances
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 5:
        centers_final, score_final = _slsqp_active(best_centers, best_score, time_budget=min(remaining*0.9, 10.0))
        if score_final > best_score:
            best_score = score_final
            best_centers = centers_final.copy()
            
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
