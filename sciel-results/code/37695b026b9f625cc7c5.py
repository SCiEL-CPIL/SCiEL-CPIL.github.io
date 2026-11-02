"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog
import time

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    
    Approach: Use Delaunay triangulation to identify relevant circle-circle
    constraints, solve LP for radii, then optimize centers via gradient ascent
    on the LP dual (shadow prices).
    """
    N = 21
    start_time = time.time()
    time_limit = 115  # Leave 5s margin for 120s limit
    
    def get_delaunay_pairs(centers):
        """Get Delaunay neighbor pairs from center positions."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    
    def solve_lp_with_dual(centers, W, H, delaunay_edges):
        """
        Solve LP for optimal radii given fixed centers and box size.
        Returns (radii, dual_values) where dual_values give shadow prices
        for each constraint, useful for gradient computation.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij (only Delaunay edges)
        for i, j in delaunay_edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
        
        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            # Left
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            # Right
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            # Bottom
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            # Top
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                # Extract dual values (shadow prices) for inequality constraints
                # In HiGHS, res.ineqlin.marginals gives the duals for A_ub constraints
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                return res.x, duals
        except Exception:
            pass
        return None, None

    def compute_forces(centers, radii, W, H, delaunay_edges, duals):
        """
        Compute forces on centers using LP dual values (shadow prices).
        The dual value for constraint r_i + r_j <= d_ij tells us how much
        increasing d_ij would improve the objective, which guides center movement.
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        
        if duals is not None and len(duals) > 0:
            # Pair constraints come first in A_ub
            num_pair_constraints = len(delaunay_edges)
            for k, (i, j) in enumerate(delaunay_edges):
                if k < len(duals):
                    dual_val = abs(duals[k])  # Shadow price (should be >= 0 for max)
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-10 and dual_val > 1e-8:
                        # Moving i away from j (and j away from i) increases d_ij
                        direction = d_vec / d
                        forces[i] += direction * dual_val
                        forces[j] -= direction * dual_val
            
            # Boundary constraints follow pair constraints
            offset = num_pair_constraints
            for i in range(n):
                # Left: r_i <= x_i, dual at index offset + 4*i
                idx = offset + 4 * i
                if idx < len(duals):
                    dual_val = abs(duals[idx])
                    if dual_val > 1e-8:
                        forces[i][0] += dual_val  # Push right
                # Right: r_i <= W - x_i, dual at index offset + 4*i + 1
                idx = offset + 4 * i + 1
                if idx < len(duals):
                    dual_val = abs(duals[idx])
                    if dual_val > 1e-8:
                        forces[i][0] -= dual_val  # Push left
                # Bottom: r_i <= y_i, dual at index offset + 4*i + 2
                idx = offset + 4 * i + 2
                if idx < len(duals):
                    dual_val = abs(duals[idx])
                    if dual_val > 1e-8:
                        forces[i][1] += dual_val  # Push up
                # Top: r_i <= H - y_i, dual at index offset + 4*i + 3
                idx = offset + 4 * i + 3
                if idx < len(duals):
                    dual_val = abs(duals[idx])
                    if dual_val > 1e-8:
                        forces[i][1] -= dual_val  # Push down
        else:
            # Fallback: use tight constraint detection
            for i, j in delaunay_edges:
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    continue
                if abs(radii[i] + radii[j] - d) < 1e-5:
                    direction = d_vec / d
                    forces[i] += direction
                    forces[j] -= direction
            
            for i in range(n):
                if abs(radii[i] - centers[i][0]) < 1e-5:
                    forces[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-5:
                    forces[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-5:
                    forces[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-5:
                    forces[i][1] -= 1.0
        
        return forces

    def optimize_packing(centers, W, H, max_iters=100, tol=1e-8):
        """
        Iteratively optimize centers to maximize sum of radii.
        Uses Delaunay-based constraint set and LP duals for force computation.
        """
        centers = centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        step_size = 0.03
        stall_count = 0
        delaunay_edges = get_delaunay_pairs(centers)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            radii, duals = solve_lp_with_dual(centers, W, H, delaunay_edges)
            if radii is None:
                break
            
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_radii = radii.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 20:
                # Shake to escape local minima
                noise_scale = 0.003 * (1.0 + it / max_iters)
                centers += np.random.normal(0, noise_scale, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
                # Recompute Delaunay
                delaunay_edges = get_delaunay_pairs(centers)
                stall_count = 0
                step_size = max(0.003, step_size * 0.7)
                continue
            
            # Compute forces using duals
            forces = compute_forces(centers, radii, W, H, delaunay_edges, duals)
            
            # Normalize and apply step
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm * step_size
                centers += forces
                
                # Clamp to box with small margin
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
                
                # Recompute Delaunay every 3 iterations
                if it % 3 == 0:
                    delaunay_edges = get_delaunay_pairs(centers)
                
            # Decay step size
            step_size *= 0.985
            
            if abs(val - best_val) < tol and it > 30:
                break

        if best_radii is not None:
            return best_centers, best_radii
        return None, None

    def generate_hexagonal_config(rows, cols, W, H):
        """Generate a hexagonal-like packing configuration."""
        pts = []
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) / cols + (0.5 / cols) * (row % 2)
                y = (row + 0.5) / rows
                x = min(max(x, 0.01), 0.99)
                y = min(max(y, 0.01), 0.99)
                pts.append([x * W, y * H])
        return np.array(pts[:N])
    
    def generate_grid_config(rows, cols, W, H):
        """Generate a regular grid configuration."""
        pts = []
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) / cols
                y = (row + 0.5) / rows
                pts.append([x * W, y * H])
        return np.array(pts[:N])
    
    def run_optimization(W, H, initial_centers, max_iters=80):
        """Run local optimization for a specific box size and initial config."""
        if time.time() - start_time > time_limit:
            return -np.inf, None, None
        centers, radii = optimize_packing(initial_centers, W, H, max_iters=max_iters)
        if radii is not None:
            return np.sum(radii), centers, radii
        return -np.inf, None, None

    # Finer aspect ratio search
    # Perimeter 2(W+H) = 4 => W+H=2
    aspects = np.arange(0.3, 3.3, 0.1)
    aspects = np.unique(np.round(aspects, 2))
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Generate diverse initial configurations in unit square
    configs = []
    
    # Grid patterns
    for rows, cols in [(5,5), (4,6), (6,4), (3,7), (7,3), (6,3), (3,6)]:
        configs.append(generate_grid_config(rows, cols, 1.0, 1.0))
    
    # Hexagonal patterns
    for rows, cols in [(5,5), (4,6), (6,4)]:
        configs.append(generate_hexagonal_config(rows, cols, 1.0, 1.0))
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        np.random.seed(seed)
        configs.append(np.random.rand(N, 2))
    
    # Sorted random (clustered) configurations
    np.random.seed(999)
    for _ in range(3):
        pts = np.random.rand(N, 2)
        pts[:, 0] = np.sort(pts[:, 0])
        configs.append(pts)
    
    # Boundary-biased configurations
    np.random.seed(555)
    for _ in range(2):
        pts = np.random.rand(N, 2)
        # Push some points toward boundaries
        mask = np.random.rand(N) < 0.3
        pts[mask, 0] = np.where(np.random.rand(np.sum(mask)) < 0.5, 0.05, 0.95)
        pts[mask, 1] = np.where(np.random.rand(np.sum(mask)) < 0.5, 0.05, 0.95)
        configs.append(pts)
    
    # Iterate through aspect ratios and configs
    for aspect in aspects:
        if time.time() - start_time > time_limit:
            break
            
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        for config in configs:
            if time.time() - start_time > time_limit:
                break
                
            # Scale config to box
            centers_init = config * np.array([W, H])
            
            val, centers, radii = run_optimization(W, H, centers_init)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((N, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    if best_result_global is not None:
        # Verify and clamp
        result = best_result_global.copy()
        # Ensure circles don't overlap and are within bounds
        # Re-center if needed
        if result[:, 0].min() < 0:
            result[:, 0] -= result[:, 0].min()
        if result[:, 1].min() < 0:
            result[:, 1] -= result[:, 1].min()
        
        # Verify perimeter constraint
        x_min = result[:, 0].min()
        x_max = result[:, 0].max()
        y_min = result[:, 1].min()
        y_max = result[:, 1].max()
        rect_W = x_max - x_min
        rect_H = y_max - y_min
        perimeter = 2 * (rect_W + rect_H)
        
        # If perimeter exceeds 4, scale down
        if perimeter > 4.0:
            scale = 4.0 / perimeter
            result[:, :2] = result[:, :2] * scale
            result[:, 2] = result[:, 2] * scale
        
        return result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
