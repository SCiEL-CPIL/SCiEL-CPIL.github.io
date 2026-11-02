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
    """
    
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

    def solve_lp(centers, W, H, active_edges):
        """
        Solve LP for optimal radii given fixed centers, box, and active edges.
        Returns radii, duals, success.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints
        for i, j in active_edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
        
        # Boundary constraints
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
                duals = res.ineqlin.marginals if hasattr(res, 'ineqlin') else None
                return res.x, duals, True
        except Exception:
            pass
        return None, None, False

    def optimize_centers(W, H, initial_centers, max_iters=100):
        """
        Optimize centers for a fixed box size W, H.
        Uses dynamic active constraint set.
        """
        centers = initial_centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        # Initialize active edges with Delaunay
        active_edges = get_delaunay_edges(centers)
        
        step_size = 0.02
        stall_count = 0
        
        for it in range(max_iters):
            # 1. Solve LP with current active edges
            radii, duals, success = solve_lp(centers, W, H, active_edges)
            if not success:
                # If LP fails, shrink radii slightly or break
                break
            
            val = np.sum(radii)
            
            # 2. Check for overlaps with ALL pairs (feasibility check)
            # We need to ensure no pair overlaps: r_i + r_j <= d_ij
            # If any pair overlaps, it must be added to active_edges
            new_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if radii[i] + radii[j] > d + 1e-9:
                        # Check if already in active_edges
                        if (min(i,j), max(i,j)) not in active_edges:
                            new_edges.append((min(i,j), max(i,j)))
            
            if new_edges:
                active_edges.extend(new_edges)
                # Re-solve LP with new constraints to get valid radii
                radii, duals, success = solve_lp(centers, W, H, active_edges)
                if not success:
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
                # Shake
                centers += np.random.normal(0, 0.01, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
                active_edges = get_delaunay_edges(centers)
                stall_count = 0
                step_size = max(0.005, step_size * 0.5)
                continue
            
            # 3. Compute Forces
            forces = np.zeros_like(centers)
            
            if duals is not None and len(duals) > 0:
                num_pair_constraints = len(active_edges)
                
                # Pair forces
                for k, (i, j) in enumerate(active_edges):
                    if k < len(duals):
                        dual_val = abs(duals[k])
                        if dual_val > 1e-8:
                            d_vec = centers[i] - centers[j]
                            d = np.linalg.norm(d_vec)
                            if d > 1e-10:
                                direction = d_vec / d
                                forces[i] += direction * dual_val
                                forces[j] -= direction * dual_val
                
                # Boundary forces
                offset = num_pair_constraints
                for i in range(n):
                    # Left
                    idx = offset + 4 * i
                    if idx < len(duals):
                        dual_val = abs(duals[idx])
                        if dual_val > 1e-8:
                            forces[i][0] += dual_val
                    # Right
                    idx = offset + 4 * i + 1
                    if idx < len(duals):
                        dual_val = abs(duals[idx])
                        if dual_val > 1e-8:
                            forces[i][0] -= dual_val
                    # Bottom
                    idx = offset + 4 * i + 2
                    if idx < len(duals):
                        dual_val = abs(duals[idx])
                        if dual_val > 1e-8:
                            forces[i][1] += dual_val
                    # Top
                    idx = offset + 4 * i + 3
                    if idx < len(duals):
                        dual_val = abs(duals[idx])
                        if dual_val > 1e-8:
                            forces[i][1] -= dual_val
            
            # Add strong repulsion for any remaining overlaps (safety net)
            for i in range(n):
                for j in range(i + 1, n):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-10:
                        # Random push if coincident
                        forces[i] += np.random.normal(0, 0.1, 2)
                        forces[j] -= forces[i]
                        continue
                    if radii[i] + radii[j] > d:
                        overlap = radii[i] + radii[j] - d
                        direction = d_vec / d
                        # Push apart proportional to overlap
                        forces[i] += direction * overlap * 5.0
                        forces[j] -= direction * overlap * 5.0

            # Normalize and apply
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                forces = forces / max_norm * step_size
                centers += forces
                centers[:, 0] = np.clip(centers[:, 0], 1e-4, W - 1e-4)
                centers[:, 1] = np.clip(centers[:, 1], 1e-4, H - 1e-4)
            
            # Update Delaunay edges periodically to keep active set relevant
            if it % 10 == 0:
                del_edges = get_delaunay_edges(centers)
                # Merge: keep current active edges that are still tight or in Delaunay, plus new Delaunay
                # Simple strategy: Reset to Delaunay + any currently overlapping pairs
                new_active = set(del_edges)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if radii[i] + radii[j] > d:
                            new_active.add((min(i,j), max(i,j)))
                active_edges = list(new_active)
            
            step_size *= 0.95
            
            if val > best_val and abs(val - best_val) < 1e-7 and it > 50:
                break

        return best_val, best_centers, best_radii

    # Outer loop: Optimize Aspect Ratio
    # W + H = 2. W in (0, 2).
    # Symmetry: W and H are interchangeable. So we can search W in (1, 2) and assume W >= H.
    # Or just search W in (0.5, 1.5) to cover aspect ratios 0.33 to 3.
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Coarse search over W
    W_values = np.linspace(0.5, 1.5, 11) # 0.5 to 1.5 step 0.1
    
    initial_configs = []
    # Grid 5x5
    gx, gy = 5, 5
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    initial_configs.append(grid_pts[:21])
    
    # Hex
    hex_pts = []
    for row in range(5):
        for col in range(5):
            x = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
            y = (row + 0.5) / 5.0
            hex_pts.append([x, y])
    initial_configs.append(np.array(hex_pts[:21]))
    
    # Random
    np.random.seed(42)
    initial_configs.append(np.random.rand(21, 2))
    
    for W in W_values:
        H = 2.0 - W
        for config in initial_configs:
            centers_init = config * np.array([W, H])
            val, centers, radii = optimize_centers(W, H, centers_init, max_iters=80)
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((21, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    # Refine best W
    if best_result_global is not None:
        # Find the W that gave the best result? 
        # We didn't store it. Let's just do a finer search around the middle or re-run with more iters on the best few Ws.
        # For simplicity and speed, we'll just return the best found so far.
        # If we have time, we could refine. 120s limit. 
        # 11 Ws * 3 configs * 80 iters * (LP + Overlap Check) 
        # LP is small (21 vars, ~50-100 constraints). Overlap check is O(N^2) = 441.
        # Should be fast enough.
        
        pass

    if best_result_global is not None:
        result = best_result_global.copy()
        # Final verification and clamping
        # Shift to origin
        result[:, 0] -= result[:, 0].min()
        result[:, 1] -= result[:, 1].min()
        
        # Check perimeter
        W = result[:, 0].max() - result[:, 0].min()
        H = result[:, 1].max() - result[:, 1].min()
        
        # If perimeter > 4, scale down
        P = 2 * (W + H)
        if P > 4.0:
            scale = 4.0 / P
            result[:, 0] *= scale
            result[:, 1] *= scale
            result[:, 2] *= scale
            
        return result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
