import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import Voronoi

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_value = -np.inf
    best_config = None

    def _is_valid(pts, W, H, tol=1e-9):
        """Check if packing is valid: no overlaps, within bounds."""
        xs = pts[:, 0]
        ys = pts[:, 1]
        rs = pts[:, 2]
        
        # Boundary check
        for i in range(N):
            if xs[i] < rs[i] - tol:
                return False
            if xs[i] + rs[i] > W + tol:
                return False
            if ys[i] < rs[i] - tol:
                return False
            if ys[i] + rs[i] > H + tol:
                return False
        
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def _get_voronoi_inradii(pts, W, H):
        """Compute inradius of each Voronoi cell for the given points."""
        xs = pts[:, 0]
        ys = pts[:, 1]
        inradii = np.zeros(N)
        
        # Add boundary points for proper Voronoi
        margin = max(W, H) * 0.5
        boundary_pts = [
            [-margin, -margin], [W + margin, -margin],
            [-margin, H + margin], [W + margin, H + margin],
            [-margin, H/2], [W + margin, H/2],
            [W/2, -margin], [W/2, H + margin]
        ]
        
        all_pts = np.vstack([pts[:, :2], np.array(boundary_pts)])
        
        try:
            vor = Voronoi(all_pts)
        except Exception:
            return inradii
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0 or -1 in region:
                inradii[i] = 0.01
                continue
            
            # Compute inradius of the polygon
            vertices = vor.vertices[region]
            cx, cy = xs[i], ys[i]
            
            min_dist = np.inf
            for k in range(len(vertices)):
                v1 = vertices[k]
                v2 = vertices[(k + 1) % len(vertices)]
                # Distance from point to line segment
                dx = v2[0] - v1[0]
                dy = v2[1] - v1[1]
                if dx == 0 and dy == 0:
                    continue
                t = ((cx - v1[0]) * dx + (cy - v1[1]) * dy) / (dx*dx + dy*dy)
                t = np.clip(t, 0, 1)
                px = v1[0] + t * dx
                py = v1[1] + t * dy
                dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                min_dist = min(min_dist, dist)
            
            inradii[i] = min_dist * 0.95  # Slight safety margin
        
        return inradii

    def _newton_contact_solver(pts, W, H, max_iter=50, tol=1e-10):
        """
        Newton-Raphson solver for contact graph.
        Tries to increase radii by adjusting positions to maintain contacts.
        """
        pts = pts.copy()
        xs = pts[:, 0].copy()
        ys = pts[:, 1].copy()
        rs = pts[:, 2].copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Build contact graph: find pairs that are in contact or nearly in contact
            # A pair (i,j) is in contact if dist(i,j) <= rs[i] + rs[j] + eps
            eps = 1e-6
            contacts = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= rs[i] + rs[j] + eps:
                        contacts.append((i, j, dist))
            
            # Also check boundary contacts
            boundary_contacts = []
            for i in range(N):
                if abs(xs[i] - rs[i]) < eps:  # left wall
                    boundary_contacts.append((i, 'L'))
                if abs(W - xs[i] - rs[i]) < eps:  # right wall
                    boundary_contacts.append((i, 'R'))
                if abs(ys[i] - rs[i]) < eps:  # bottom wall
                    boundary_contacts.append((i, 'B'))
                if abs(H - ys[i] - rs[i]) < eps:  # top wall
                    boundary_contacts.append((i, 'T'))
            
            # Try to increase all radii slightly and solve for new positions
            dr = 1e-4  # Small radius increment
            
            # Build linear system: for each contact, enforce dist = r_i + r_j + dr
            # Variables: dx_i, dy_i for each circle
            n_vars = 2 * N
            A = np.zeros((len(contacts) + len(boundary_contacts), n_vars))
            b = np.zeros(len(contacts) + len(boundary_contacts))
            
            row = 0
            for (i, j, dist) in contacts:
                if dist < 1e-10:
                    continue
                # Gradient of (dist - r_i - r_j) w.r.t. positions
                dx_ij = xs[i] - xs[j]
                dy_ij = ys[i] - ys[j]
                A[row, 2*i] = dx_ij / dist
                A[row, 2*i+1] = dy_ij / dist
                A[row, 2*j] = -dx_ij / dist
                A[row, 2*j+1] = -dy_ij / dist
                b[row] = dr  # Want to increase distance by dr
                row += 1
            
            for (i, wall) in boundary_contacts:
                if wall == 'L':
                    A[row, 2*i] = 1
                    b[row] = dr
                elif wall == 'R':
                    A[row, 2*i] = -1
                    b[row] = dr
                elif wall == 'B':
                    A[row, 2*i+1] = 1
                    b[row] = dr
                elif wall == 'T':
                    A[row, 2*i+1] = -1
                    b[row] = dr
                row += 1
            
            if row == 0:
                # No contacts, just increase radii
                rs += dr
                continue
            
            # Solve the linear system using least squares
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(A[:row], b[:row], rcond=None)
            except Exception:
                break
            
            # Apply the solution
            for i in range(N):
                xs[i] += sol[2*i]
                ys[i] += sol[2*i+1]
            
            # Check if all constraints are satisfied
            valid = True
            for i in range(N):
                if xs[i] < rs[i] - 1e-8 or xs[i] + rs[i] > W + 1e-8:
                    valid = False
                    break
                if ys[i] < rs[i] - 1e-8 or ys[i] + rs[i] > H + 1e-8:
                    valid = False
                    break
            
            if valid:
                # Check overlaps
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < rs[i] + rs[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                rs += dr
                # Update pts
                pts[:, 0] = xs
                pts[:, 1] = ys
                pts[:, 2] = rs
            else:
                # Revert if invalid
                break
        
        return pts

    def _voronoi_initialization(W, H):
        """Create initial configuration using Voronoi-based radii."""
        # Use a grid as starting points
        cols = 5
        rows = 5
        dx = W / cols
        dy = H / rows
        
        pts = np.zeros((N, 3))
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                r = min(dx, dy) * 0.4
                pts[count] = [cx, cy, r]
                count += 1
        
        # Fill any remaining
        while count < N:
            pts[count] = [W/2, H/2, 0.05]
            count += 1
        
        # Compute Voronoi inradii and use as target radii
        inradii = _get_voronoi_inradii(pts, W, H)
        pts[:, 2] = np.minimum(pts[:, 2], inradii)
        
        return pts

    def _slsqp_optimize(pts, W, H):
        """SLSQP optimization to maximize sum of radii with hard constraints."""
        n = N
        x0 = pts.flatten()
        
        def objective(x):
            if time.time() > deadline:
                return 0
            pts_local = x.reshape(-1, 3)
            return -np.sum(pts_local[:, 2])
        
        # Inequality constraints: no overlaps
        def no_overlap(i, j):
            def constraint(x):
                if time.time() > deadline:
                    return 1.0
                pts_local = x.reshape(-1, 3)
                dx = pts_local[i, 0] - pts_local[j, 0]
                dy = pts_local[i, 1] - pts_local[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                return dist - pts_local[i, 2] - pts_local[j, 2]
            return constraint
        
        # Inequality constraints: within bounds
        def bound_constraint(i, axis, side):
            def constraint(x):
                if time.time() > deadline:
                    return 1.0
                pts_local = x.reshape(-1, 3)
                if axis == 0:
                    if side == 'min':
                        return pts_local[i, 0] - pts_local[i, 2]
                    else:
                        return W - pts_local[i, 0] - pts_local[i, 2]
                else:
                    if side == 'min':
                        return pts_local[i, 1] - pts_local[i, 2]
                    else:
                        return H - pts_local[i, 1] - pts_local[i, 2]
            return constraint
        
        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({'type': 'ineq', 'fun': no_overlap(i, j)})
            constraints.append({'type': 'ineq', 'fun': bound_constraint(i, 0, 'min')})
            constraints.append({'type': 'ineq', 'fun': bound_constraint(i, 0, 'max')})
            constraints.append({'type': 'ineq', 'fun': bound_constraint(i, 1, 'min')})
            constraints.append({'type': 'ineq', 'fun': bound_constraint(i, 1, 'max')})
        
        r_max = min(W, H) / 2
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            bounds.append((1e-6, r_max))
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-10})
            if res.success:
                return res.x.reshape(-1, 3)
        except Exception:
            pass
        
        return pts

    # Try different aspect ratios
    aspect_ratios = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    for x_ratio in aspect_ratios:
        if time.time() > deadline:
            break
        
        W = 2 * x_ratio
        H = 2 * (1 - x_ratio)
        
        # Get initial configuration
        pts = _voronoi_initialization(W, H)
        
        # Run SLSQP optimization
        pts_opt = _slsqp_optimize(pts, W, H)
        
        # Polish with Newton contact solver
        if time.time() < deadline - 20:
            pts_polished = _newton_contact_solver(pts_opt, W, H)
        else:
            pts_polished = pts_opt
        
        # Check validity
        if _is_valid(pts_polished, W, H):
            val = np.sum(pts_polished[:, 2])
            if val > best_value:
                best_value = val
                best_config = pts_polished.copy()
        
        # Also check the SLSQP result directly
        if _is_valid(pts_opt, W, H):
            val = np.sum(pts_opt[:, 2])
            if val > best_value:
                best_value = val
                best_config = pts_opt.copy()
    
    if best_config is None:
        # Fallback: simple grid
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        dx = W / cols
        dy = H / rows
        pts = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                cx = (i + 0.5) * dx
                cy = (j + 0.5) * dy
                r = min(dx, dy) * 0.4
                pts.append([cx, cy, r])
                count += 1
        best_config = np.array(pts[:N])
    
    return best_config
