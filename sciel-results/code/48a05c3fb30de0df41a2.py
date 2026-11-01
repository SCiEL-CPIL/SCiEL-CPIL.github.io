"""
Circle packing in perimeter-4 rectangle: Delaunay Topology Search + Rigid Body Relaxation.

Strategy:
1. Generate diverse initial configurations.
2. Use Delaunay Triangulation to identify the contact graph (adjacency).
3. Solve for positions and radii using a constraint-satisfaction approach (Rigid Body Relaxation)
   that respects the contact graph, rather than smooth penalties.
4. Iterate: Relax -> Recompute Triangulation -> Update Graph -> Relax.
5. Final polish with SLSQP on the best candidate found via topology search.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True

def _compute_delaunay_adjacency(x, y, r, w, h):
    """
    Compute adjacency matrix based on Delaunay triangulation of circle centers.
    Two circles are 'adjacent' if they share an edge in the Delaunay triangulation.
    This defines the contact graph topology.
    """
    points = np.column_stack((x, y))
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback to simple distance-based adjacency if Delaunay fails (degenerate points)
        adj = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < (r[i] + r[j]) * 1.1:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj
    
    adj = np.zeros((N, N), dtype=bool)
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a = simplex[i]
                b = simplex[j]
                adj[a, b] = True
                adj[b, a] = True
    
    # Ensure we don't have false positives if circles are far apart but triangulated
    # Filter edges where distance > (r_i + r_j) + margin
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j]:
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d > (r[i] + r[j]) + 0.1: # Heuristic margin
                    adj[i, j] = False
                    adj[j, i] = False
                    
    return adj

def _rigid_body_relax(x, y, r, w, h, adj, max_iter=200, dt=0.01):
    """
    Relax positions and radii to satisfy:
    1. Adjacent circles touch (distance = r_i + r_j)
    2. All circles are within bounds.
    
    This is a simplified force-directed method targeting the contact graph.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(max_iter):
        fx = np.zeros(N)
        fy = np.zeros(N)
        fr = np.zeros(N)
        
        # 1. Contact forces: Push apart if overlapping, pull together if adjacent but separated
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                target_d = r[i] + r[j]
                
                if adj[i, j]:
                    # Adjacent: try to make d == target_d
                    error = d - target_d
                    # If error is large, it's likely a topology change or bad init
                    if abs(error) > 0.5:
                        continue
                    
                    force = error * 0.5
                    fx[i] += force * (dx/d)
                    fy[i] += force * (dy/d)
                    fx[j] -= force * (dx/d)
                    fy[j] -= force * (dy/d)
                    # Adjust radii to help satisfy contact if positions are stuck?
                    # For now, keep radii fixed in this step, optimize later.
                else:
                    # Non-adjacent: must not overlap
                    if d < target_d:
                        error = target_d - d
                        force = error * 2.0 # Stronger repulsion for non-contacts to clear space
                        fx[i] += force * (dx/d)
                        fy[i] += force * (dy/d)
                        fx[j] -= force * (dx/d)
                        fy[j] -= force * (dy/d)
        
        # 2. Boundary forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        fx -= 5.0 * left_viol
        fx += 5.0 * right_viol
        fy -= 5.0 * bottom_viol
        fy += 5.0 * top_viol
        
        # Update
        x += dt * fx
        y += dt * fy
        
        # Clamp to bounds (soft clamp to avoid singularities)
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
        # Check convergence
        if np.max(np.abs(fx)) < 1e-6 and np.max(np.abs(fy)) < 1e-6:
            break
            
    return x, y

def _expand_radii(x, y, r, w, h, adj, max_iter=50):
    """
    Try to expand radii while maintaining validity.
    Uses a greedy approach: expand the circle with the most 'room' (lowest contact pressure).
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_iter):
        improved = False
        # Calculate 'slack' for each circle: min distance to neighbor - (r_i + r_j)
        # If slack > 0, we can expand.
        slacks = np.zeros(N)
        for i in range(N):
            min_slack = 1e9
            for j in range(N):
                if i == j: continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                current_dist = d
                required_dist = r[i] + r[j]
                slack = current_dist - required_dist
                min_slack = min(min_slack, slack)
            
            # Also check bounds
            min_slack = min(min_slack, x[i] - r[i])
            min_slack = min(min_slack, w - r[i] - x[i])
            min_slack = min(min_slack, y[i] - r[i])
            min_slack = min(min_slack, h - r[i] - y[i])
            
            slacks[i] = min_slack
        
        # Expand the circle with the largest positive slack
        idx = np.argmax(slacks)
        if slacks[idx] > 1e-6:
            # Expand by a fraction of the slack
            delta_r = slacks[idx] * 0.5
            r[idx] += delta_r
            improved = True
            
            # Re-relax positions slightly to accommodate
            x, y = _rigid_body_relax(x, y, r, w, h, adj, max_iter=20, dt=0.005)
            
            # Validate
            if not _is_valid(x, y, r, w, h):
                r[idx] -= delta_r
                improved = False
        
        if not improved:
            break
            
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))
        
        # Diverse aspect ratios
        aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3: continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = get_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break
            
        x = z0[0::3][:N].copy()
        y = z0[1::3][:N].copy()
        r = z0[2::3][:N].copy()
        w = z0[-1]
        h = 2.0 - w
        
        # Phase 1: Topology Search Loop
        for topo_iter in range(10):
            if time.time() > deadline - 20:
                break
                
            # 1. Compute adjacency from current state
            adj = _compute_delaunay_adjacency(x, y, r, w, h)
            
            # 2. Relax positions to satisfy contact graph
            x, y = _rigid_body_relax(x, y, r, w, h, adj, max_iter=100)
            
            # 3. Expand radii
            x, y, r = _expand_radii(x, y, r, w, h, adj, max_iter=20)
            
            # 4. Check validity and update best
            z_curr = np.zeros(3*N + 1)
            z_curr[0::3][:N] = x
            z_curr[1::3][:N] = y
            z_curr[2::3][:N] = r
            z_curr[-1] = w
            
            valid, val = check_validity(z_curr)
            if valid and val > best_value:
                best_value = val
                best_result = z_curr.copy()
        
        # Phase 2: Local Refinement with SLSQP on the best topology-found state
        if best_result is not None and time.time() < deadline - 10:
            z_opt = best_result.copy()
            
            def objective(z_):
                x_ = z_[0::3][:N]
                y_ = z_[1::3][:N]
                r_ = z_[2::3][:N]
                w_ = z_[-1]
                h_ = 2.0 - w_
                
                if w_ < 0.2 or h_ < 0.2 or np.any(r_ < 0):
                    return 1e6
                
                obj = -np.sum(r_)
                
                # Penalty for overlaps
                dx = x_[:, None] - x_[None, :]
                dy = y_[:, None] - y_[None, :]
                d_sq = dx * dx + dy * dy
                r_sum = r_[:, None] + r_[None, :]
                
                iu = np.triu_indices(N, k=1)
                d_pairs = np.sqrt(np.maximum(d_sq[iu], 0.0))
                overlap = np.maximum(r_sum[iu] - d_pairs, 0.0)
                obj += 1000.0 * np.sum(overlap * overlap)
                
                # Penalty for boundaries
                left_viol = np.maximum(r_ - x_, 0.0)
                right_viol = np.maximum(x_ + r_ - w_, 0.0)
                bottom_viol = np.maximum(r_ - y_, 0.0)
                top_viol = np.maximum(y_ + r_ - h_, 0.0)
                obj += 1000.0 * (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                                 np.sum(bottom_viol**2) + np.sum(top_viol**2))
                
                return obj
            
            try:
                res = minimize(objective, z_opt, method='SLSQP',
                               bounds=[(0, 2)] * (3*N) + [(0.3, 1.7)],
                               options={'maxiter': 200, 'ftol': 1e-12})
                
                valid, val = check_validity(res.x)
                if valid and val > best_value:
                    best_value = val
                    best_result = res.x.copy()
            except Exception:
                pass

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
