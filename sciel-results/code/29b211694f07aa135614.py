"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    
    Approach: Delaunay-based LP formulation for radii, with gradient-based
    center optimization using dual shadow prices.
    """
    N = 21
    
    def get_delaunay_edges(centers):
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
    
    def solve_radii_lp(centers, W, H, edges):
        """
        Solve LP: max sum(r_i) s.t. r_i + r_j <= d_ij for edges,
        r_i <= dist_to_boundary, r_i >= 0.
        Returns (radii, duals) or (None, None) on failure.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub_rows = []
        b_ub_vals = []
        
        # Circle-circle constraints
        for i, j in edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(d)
        
        # Boundary constraints
        for i in range(n):
            row = np.zeros(n); row[i] = 1.0
            A_ub_rows.append(row); b_ub_vals.append(centers[i][0])
            row = np.zeros(n); row[i] = 1.0
            A_ub_rows.append(row); b_ub_vals.append(W - centers[i][0])
            row = np.zeros(n); row[i] = 1.0
            A_ub_rows.append(row); b_ub_vals.append(centers[i][1])
            row = np.zeros(n); row[i] = 1.0
            A_ub_rows.append(row); b_ub_vals.append(H - centers[i][1])
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        bounds = [(0.0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                return res.x, duals
        except Exception:
            pass
        return None, None
    
    def compute_center_forces(centers, radii, W, H, edges, duals):
        """
        Compute forces on centers from LP duals (shadow prices).
        Positive dual on r_i + r_j <= d_ij means increasing d_ij improves objective.
        """
        n = len(centers)
        forces = np.zeros_like(centers)
        
        if duals is None or len(duals) == 0:
            return forces
        
        n_edges = len(edges)
        
        # Circle-circle edge forces
        for k, (i, j) in enumerate(edges):
            if k >= len(duals):
                break
            dual_val = abs(duals[k])
            if dual_val < 1e-10:
                continue
            d_vec = centers[i] - centers[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                continue
            direction = d_vec / d
            forces[i] += direction * dual_val
            forces[j] -= direction * dual_val
        
        # Boundary forces
        offset = n_edges
        for i in range(n):
            base = offset + 4 * i
            if base + 3 < len(duals):
                # Left wall: r_i <= x_i -> push right if tight
                dl = abs(duals[base])
                if dl > 1e-10:
                    forces[i][0] += dl
                # Right wall: r_i <= W - x_i -> push left if tight
                dr = abs(duals[base + 1])
                if dr > 1e-10:
                    forces[i][0] -= dr
                # Bottom wall: r_i <= y_i -> push up if tight
                db = abs(duals[base + 2])
                if db > 1e-10:
                    forces[i][1] += db
                # Top wall: r_i <= H - y_i -> push down if tight
                dt = abs(duals[base + 3])
                if dt > 1e-10:
                    forces[i][1] -= dt
        
        return forces
    
    def optimize_centers(centers, W, H, max_iters=300):
        """
        Iteratively update centers using LP dual forces.
        Returns (best_centers, best_radii, best_value).
        """
        centers = centers.copy().astype(float)
        n = len(centers)
        
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        step = 0.03
        stall = 0
        edges = get_delaunay_edges(centers)
        
        for it in range(max_iters):
            radii, duals = solve_radii_lp(centers, W, H, edges)
            if radii is None:
                break
            
            val = np.sum(radii)
            if val > best_val + 1e-9:
                best_val = val
                best_centers = centers.copy()
                best_radii = radii.copy()
                stall = 0
            else:
                stall += 1
            
            if stall > 50:
                # Perturb to escape local minimum
                centers += np.random.normal(0, 0.003, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 1e-5, W - 1e-5)
                centers[:, 1] = np.clip(centers[:, 1], 1e-5, H - 1e-5)
                edges = get_delaunay_edges(centers)
                stall = 0
                step = max(0.003, step * 0.5)
                continue
            
            forces = compute_center_forces(centers, radii, W, H, edges, duals)
            
            # Normalize forces to avoid exploding steps
            norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(norms)
            if max_norm > 1e-12:
                forces = forces / max_norm * step
                centers += forces
                centers[:, 0] = np.clip(centers[:, 0], 1e-5, W - 1e-5)
                centers[:, 1] = np.clip(centers[:, 1], 1e-5, H - 1e-5)
            
            # Update Delaunay periodically
            if it % 8 == 0:
                edges = get_delaunay_edges(centers)
            
            # Adaptive step decay
            step *= 0.995
            
            if it > 100 and abs(val - best_val) < 1e-8:
                break
        
        if best_radii is not None:
            return best_centers, best_radii, best_val
        return None, None, -np.inf
    
    def make_hex_grid(n, W, H):
        """Create a hexagonal-like initial configuration scaled to box."""
        # Start with hexagonal packing in unit square, then scale
        pts = []
        row_h = np.sqrt(3) / 2
        for row in range(6):
            for col in range(6):
                x = col + 0.5 * (row % 2)
                y = row * row_h
                pts.append([x, y])
        pts = np.array(pts)
        # Normalize to [0,1]x[0,1]
        pts[:, 0] = (pts[:, 0] - pts[:, 0].min()) / (pts[:, 0].max() - pts[:, 0].min() + 1e-12)
        pts[:, 1] = (pts[:, 1] - pts[:, 1].min()) / (pts[:, 1].max() - pts[:, 1].min() + 1e-12)
        # Scale to box
        return pts * np.array([W, H])
    
    def make_grid(nx, ny, W, H):
        """Create grid initial configuration."""
        xs = np.linspace(0.5/nx, 1 - 0.5/nx, nx)
        ys = np.linspace(0.5/ny, 1 - 0.5/ny, ny)
        pts = np.array([[x, y] for y in ys for x in xs])
        return pts * np.array([W, H])
    
    # Focus on aspect ratios near optimal
    # Perimeter: 2(W+H) = 4 => W + H = 2
    aspects = [1.0, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.2]
    
    best_val_global = -np.inf
    best_result_global = None
    
    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        # Multiple initial configurations
        configs = []
        configs.append(make_grid(5, 5, W, H))
        configs.append(make_grid(4, 6, W, H))
        configs.append(make_grid(6, 4, W, H))
        configs.append(make_grid(7, 3, W, H))
        configs.append(make_hex_grid(21, W, H))
        
        # Random configurations
        for seed in [1, 7, 42]:
            np.random.seed(seed)
            configs.append(np.random.rand(21, 2) * np.array([W, H]))
        
        for cfg in configs:
            centers_init = cfg.copy()
            # Ensure within bounds
            centers_init[:, 0] = np.clip(centers_init[:, 0], 1e-5, W - 1e-5)
            centers_init[:, 1] = np.clip(centers_init[:, 1], 1e-5, H - 1e-5)
            
            centers, radii, val = optimize_centers(centers_init, W, H, max_iters=250)
            
            if radii is not None and val > best_val_global:
                best_val_global = val
                result = np.zeros((21, 3))
                result[:, 0] = centers[:, 0]
                result[:, 1] = centers[:, 1]
                result[:, 2] = radii
                best_result_global = result
    
    if best_result_global is not None:
        return best_result_global
    
    # Fallback: simple grid
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
