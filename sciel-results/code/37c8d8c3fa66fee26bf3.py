"""Rewritten program for circle packing in perimeter-4 rectangle."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    
    Approach: Delaunay-based LP for radii with iterative center optimization
    using shadow prices from the LP dual.
    """
    N = 21
    
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
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    
    def solve_lp(centers, W, H, edges):
        """Solve LP for optimal radii given fixed centers and box size."""
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
        
        # Boundary constraints
        for i in range(n):
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                return res.x, duals
        except Exception:
            pass
        return None, None

    def compute_forces(centers, radii, W, H, edges, duals):
        """Compute forces on centers using LP dual values."""
        n = len(centers)
        forces = np.zeros_like(centers)
        
        if duals is not None and len(duals) > 0:
            num_pairs = len(edges)
            for k, (i, j) in enumerate(edges):
                if k < len(duals):
                    dual_val = abs(duals[k])
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-10 and dual_val > 1e-8:
                        direction = d_vec / d
                        forces[i] += direction * dual_val
                        forces[j] -= direction * dual_val
            
            offset = num_pairs
            for i in range(n):
                for side_idx, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1)]):
                    idx = offset + 4 * i + side_idx
                    if idx < len(duals):
                        dual_val = abs(duals[idx])
                        if dual_val > 1e-8:
                            forces[i, axis] += sign * dual_val
        else:
            # Fallback: tight constraint detection
            for i, j in edges:
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
                    forces[i, 0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-5:
                    forces[i, 0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-5:
                    forces[i, 1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-5:
                    forces[i, 1] -= 1.0
        
        return forces

    def optimize_packing(centers, W, H, max_iters=150):
        """Iteratively optimize centers to maximize sum of radii."""
        centers = centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        step_size = 0.05
        stall_count = 0
        edges = get_delaunay_pairs(centers)
        
        for it in range(max_iters):
            radii, duals = solve_lp(centers, W, H, edges)
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
            
            if stall_count > 25:
                centers += np.random.normal(0, 0.004, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
                edges = get_delaunay_pairs(centers)
                stall_count = 0
                step_size = max(0.005, step_size * 0.5)
                continue
            
            forces = compute_forces(centers, radii, W, H, edges, duals)
            
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm * step_size
                centers += forces
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
                
                if it % 5 == 0:
                    edges = get_delaunay_pairs(centers)
            
            step_size *= 0.99
            
            if abs(val - best_val) < 1e-7 and it > 50:
                break

        if best_radii is not None:
            return best_centers, best_radii
        return None, None

    def make_configs():
        """Generate diverse initial configurations."""
        cfgs = []
        
        # 5x5 grid
        gx, gy = 5, 5
        cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
        cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
        pts = np.array([[x, y] for y in cy for x in cx])
        cfgs.append(pts[:21])
        
        # 4x6 grid
        gx, gy = 4, 6
        cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
        cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
        pts = np.array([[x, y] for y in cy for x in cx])
        cfgs.append(pts[:21])
        
        # 7x3 grid
        gx, gy = 7, 3
        cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
        cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
        pts = np.array([[x, y] for y in cy for x in cx])
        cfgs.append(pts[:21])
        
        # 6x4 grid
        gx, gy = 6, 4
        cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
        cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
        pts = np.array([[x, y] for y in cy for x in cx])
        cfgs.append(pts[:21])
        
        # Hexagonal pattern
        hex_pts = []
        for row in range(5):
            for col in range(5):
                x = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
                y = (row + 0.5) / 5.0
                hex_pts.append([x, y])
        cfgs.append(np.array(hex_pts[:21]))
        
        # Random configs
        for seed in [42, 123, 456, 789, 101, 202]:
            rng = np.random.RandomState(seed)
            cfgs.append(rng.rand(21, 2))
        
        return cfgs
    
    # Aspect ratios to try (W/H)
    # Perimeter: 2(W+H) = 4 => W + H = 2
    aspect_ratios = np.linspace(0.3, 3.0, 20)
    
    best_global = -np.inf
    best_result = None
    configs = make_configs()
    
    for ar in aspect_ratios:
        H = 2.0 / (ar + 1.0)
        W = 2.0 * ar / (ar + 1.0)
        
        for cfg in configs:
            c0 = cfg * np.array([W, H])
            c0[:, 0] = np.clip(c0[:, 0], 0.01, W - 0.01)
            c0[:, 1] = np.clip(c0[:, 1], 0.01, H - 0.01)
            
            centers, radii = optimize_packing(c0, W, H, max_iter=120)
            
            if radii is not None:
                val = np.sum(radii)
                if val > best_global:
                    best_global = val
                    result = np.zeros((21, 3))
                    result[:, :2] = centers
                    result[:, 2] = radii
                    best_result = result
    
    if best_result is not None:
        result = best_result.copy()
        result[:, 0] -= result[:, 0].min()
        result[:, 1] -= result[:, 1].min()
        return result
    
    # Fallback
    side = 5
    cells = []
    for i in range(5):
        for j in range(5):
            cells.append([(i+0.5)/5, (j+0.5)/5, 0.5/5 - 0.001])
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
