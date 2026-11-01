"""Circle packing in perimeter-4 rectangle: Contact Graph + Newton-Raphson."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if the packing is valid."""
    if W <= 0.0 or H <= 0.0:
        return False
    # Check bounds
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Check overlaps
    dists = cdist(np.column_stack((cx, cy)), np.column_stack((cx, cy)))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    if np.any(min_dists < 2 * r - tol):
        return False
    return True

def _compute_max_radii(cx, cy, W, H):
    """
    Compute the maximum possible radius for each circle given fixed centers.
    r_i = min(
        cx_i, W - cx_i, cy_i, H - cy_i,
        min_{j != i} (dist(i, j) - r_j)  # This is circular, so we use a fixed-point or iterative approach?
    )
    Actually, for a contact graph, if we assume all circles have the same radius r,
    then r is determined by the minimum distance to walls and neighbors.
    For variable radii, this is harder.
    However, the optimal packing usually has equal radii or a few distinct sizes.
    Let's assume equal radii first, as it simplifies the contact graph to unit distances.
    If we fix r, the problem is to place N circles of radius r.
    The objective is to maximize r.
    """
    # This function is a placeholder for the logic that determines radii from centers
    # For the Newton solver, we will solve for positions given a target radius r,
    # or solve for r and positions simultaneously.
    pass

def _newton_solve_contact_graph(nodes, edges, W, H, r_target, max_iter=50, tol=1e-10):
    """
    Solve for positions of circles such that:
    1. Distance between connected nodes is exactly 2 * r_target.
    2. Nodes are within [r_target, W-r_target] x [r_target, H-r_target].
    
    We use a penalty-based Newton step on the residual of the distance constraints.
    """
    n = len(nodes)
    x = np.array(nodes, dtype=float).flatten() # 2n
    
    def residual(x_vec):
        pos = x_vec.reshape(-1, 2)
        res = np.zeros(len(edges))
        for k, (i, j) in enumerate(edges):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            res[k] = dist - 2 * r_target
        # Also add soft bounds? No, Newton is for equality constraints.
        # We will handle bounds by projecting or using a barrier in the objective if we switch to minimization.
        # For a pure Newton step on distances, we need the Jacobian.
        return res, pos

    def jacobian(x_vec):
        pos = x_vec.reshape(-1, 2)
        J = np.zeros((len(edges), 2 * n))
        for k, (i, j) in enumerate(edges):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                dist = 1e-12
            J[k, 2*i] = dx / dist
            J[k, 2*i+1] = dy / dist
            J[k, 2*j] = -dx / dist
            J[k, 2*j+1] = -dy / dist
        return J

    for _ in range(max_iter):
        r, pos = residual(x)
        if np.max(np.abs(r)) < tol:
            break
        J = jacobian(x)
        # Solve J dx = -r
        try:
            dx = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        x += dx
        # Project to bounds? Newton doesn't respect bounds directly.
        # We will check validity after.
        
    return x.reshape(-1, 2)

def _generate_triangular_layouts():
    """
    Generate contact graphs for triangular lattice packings.
    For N=21, a 5x4 grid or similar might work, but triangular is denser.
    Rows of 5, 4, 5, 4, 3? Sum = 21.
    Or 5, 5, 5, 3, 3?
    Let's try a few row configurations that sum to 21.
    """
    configs = [
        [5, 4, 5, 4, 3],
        [4, 5, 4, 5, 3],
        [5, 5, 5, 3, 3],
        [3, 5, 5, 5, 3],
        [4, 4, 5, 4, 4],
        [5, 3, 5, 3, 5],
    ]
    
    graphs = []
    for row_counts in configs:
        if sum(row_counts) != N:
            continue
        nodes = []
        edges = []
        # Create nodes
        for row_idx, count in enumerate(row_counts):
            for col_idx in range(count):
                nodes.append([col_idx, row_idx]) # Logical coords
        # Create edges
        # 1. Horizontal neighbors
        for row_idx, count in enumerate(row_counts):
            for col_idx in range(count - 1):
                i = sum(row_counts[:row_idx]) + col_idx
                j = i + 1
                edges.append((i, j))
        # 2. Diagonal neighbors (triangular lattice)
        # Row r, col c is connected to Row r+1, col c and Row r+1, col c+1 (if they exist)
        # This depends on the offset. In a triangular lattice, row r+1 is shifted by 0.5.
        # Let's assume standard triangular packing where row r+1 sits in the "valleys" of row r.
        # If row r has k circles, row r+1 has k-1 or k+1.
        # Let's just generate all potential edges based on distance < 1.1 in logical coords.
        # This is safer.
        for i in range(N):
            for j in range(i + 1, N):
                dx = nodes[i][0] - nodes[j][0]
                dy = nodes[i][1] - nodes[j][1]
                if dx*dx + dy*dy < 1.2: # Close enough to be a neighbor
                    edges.append((i, j))
        graphs.append((nodes, edges, row_counts))
    return graphs

def _evaluate_config(nodes_logical, edges, W, H):
    """
    Given a contact graph (topology), find the optimal rectangle aspect ratio W
    and radius r that fits the graph.
    
    We scale the logical graph to fit in a rectangle of width W and height H.
    The distance between neighbors is 2r.
    In the logical graph, neighbor distance is 1.
    So, 2r = scale_factor * 1.
    W = scale_factor * width_logical
    H = scale_factor * height_logical
    
    Perimeter = 2(W + H) = 4 => W + H = 2.
    scale_factor * (width_logical + height_logical) = 2
    scale_factor = 2 / (width_logical + height_logical)
    r = scale_factor / 2 = 1 / (width_logical + height_logical)
    
    This assumes the bounding box of the logical graph is width_logical x height_logical.
    """
    xs = [n[0] for n in nodes_logical]
    ys = [n[1] for n in nodes_logical]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width_logical = max_x - min_x + 1 # +1 because distance is 1 between centers, so span is (max-min)*1 + 2r?
    # Wait. If centers are at 0, 1, 2... distance is 1.
    # The physical width is (max_x - min_x) * (2r).
    # The bounding box of the circles extends from min_x - r to max_x + r.
    # Total width W = (max_x - min_x) * 2r + 2r = (max_x - min_x + 1) * 2r.
    # Similarly H = (max_y - min_y + 1) * 2r.
    
    wl = max_x - min_x + 1
    hl = max_y - min_y + 1
    
    # W + H = 2
    # 2r * wl + 2r * hl = 2
    # 2r (wl + hl) = 2
    # r = 1 / (wl + hl)
    
    r = 1.0 / (wl + hl)
    W = wl * 2 * r
    H = hl * 2 * r
    
    # Now we have a candidate (W, H, r).
    # We need to check if this configuration is valid.
    # The logical positions are integers. We need to map them to physical space.
    # x_phys = (x_log - min_x) * 2r + r
    # y_phys = (y_log - min_y) * 2r + r
    
    cx = np.array([(n[0] - min_x) * 2 * r + r for n in nodes_logical])
    cy = np.array([(n[1] - min_y) * 2 * r + r for n in nodes_logical])
    
    # Check validity
    if not _is_valid(cx, cy, np.full(N, r), W, H):
        return None, 0.0
    
    # Try to improve by allowing the rectangle to be slightly larger or circles to grow?
    # The formula r = 1/(wl+hl) is exact for the bounding box.
    # However, if the graph is not "tight" (i.e. some nodes don't touch the boundary),
    # we might be able to increase r.
    # But for a triangular lattice, the bounding box is tight.
    
    # Let's try a local optimization to see if we can squeeze more out.
    # We will optimize W (aspect ratio) and scale (radius) jointly.
    # Actually, if we change W, H changes (W+H=2).
    # We can try to stretch the configuration.
    
    # Objective: Maximize sum(r_i). Here all r_i = r. So maximize r.
    # Constraints:
    # 1. Centers in [r, W-r] x [r, H-r]
    # 2. dist(i,j) >= 2r for all i,j
    
    # We have a valid starting point. Let's try to increase r.
    # This is a 1D search on r, adjusting positions to maintain contact?
    # Or just a simple local optimization on positions and W.
    
    z0 = np.concatenate([cx, cy, [W]])
    
    def obj(z):
        cx_l = z[0::2][:N]
        cy_l = z[1::2][:N]
        W_l = z[-1]
        H_l = 2.0 - W_l
        if W_l < 0.1 or H_l < 0.1:
            return 1e6
        # Max possible r for this position is limited by walls and neighbors.
        # We want to maximize the MINIMUM possible radius.
        # Let's just compute the max r for fixed positions.
        r_min = np.min(np.minimum(
            np.minimum(cx_l, W_l - cx_l),
            np.minimum(cy_l, H_l - cy_l)
        ))
        
        # Also limited by neighbors
        dists = cdist(np.column_stack((cx_l, cy_l)), np.column_stack((cx_l, cy_l)))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        r_min = min(r_min, np.min(min_dists) / 2.0)
        
        return -r_min # Minimize negative r

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    bounds.append((0.5, 1.5)) # W
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
        if res.success:
            cx_opt = res.x[0::2][:N]
            cy_opt = res.x[1::2][:N]
            W_opt = res.x[-1]
            H_opt = 2.0 - W_opt
            
            # Compute final radii (should be uniform)
            r_opt = -res.fun
            
            if _is_valid(cx_opt, cy_opt, np.full(N, r_opt), W_opt, H_opt):
                return (cx_opt, cy_opt, np.full(N, r_opt), W_opt, H_opt), r_opt * N
    except Exception:
        pass
        
    # Fallback to the analytical solution if optimization fails
    return (cx, cy, np.full(N, r), W, H), r * N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_value = 0.0
    best_result = None
    
    # 1. Generate contact graphs
    graphs = _generate_triangular_layouts()
    
    # 2. Evaluate each graph
    for nodes, edges, row_counts in graphs:
        if time.time() > deadline - 10:
            break
            
        result, val = _evaluate_config(nodes, edges, 0, 0) # W, H are derived inside
        if result is not None and val > best_value:
            best_value = val
            best_result = result
            
    # 3. If no good result, try a simple grid as fallback
    if best_result is None:
        # 5x4 grid + 1
        cx = []
        cy = []
        W = 1.0
        H = 1.0
        # This is a rough fallback
        for i in range(N):
            row, col = divmod(i, 5)
            cx.append((col + 0.5) / 5.0 * W)
            cy.append((row + 0.5) / 5.0 * H)
        cx = np.array(cx)
        cy = np.array(cy)
        r = 0.1
        best_result = (cx, cy, np.full(N, r), W, H)
        best_value = r * N
        
    # 4. Polish the best result with a general optimizer
    cx, cy, r, W, H = best_result
    z0 = np.concatenate([cx, cy, [W]])
    
    def obj_polish(z):
        cx_l = z[0::2][:N]
        cy_l = z[1::2][:N]
        W_l = z[-1]
        H_l = 2.0 - W_l
        if W_l < 0.1 or H_l < 0.1:
            return 1e6
        # Max r
        r_min = np.min(np.minimum(
            np.minimum(cx_l, W_l - cx_l),
            np.minimum(cy_l, H_l - cy_l)
        ))
        dists = cdist(np.column_stack((cx_l, cy_l)), np.column_stack((cx_l, cy_l)))
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        r_min = min(r_min, np.min(min_dists) / 2.0)
        return -r_min

    bounds = []
    for i in range(N):
        bounds.append((0, W))
    for i in range(N):
        bounds.append((0, H))
    bounds.append((0.5, 1.5))
    
    try:
        res = minimize(obj_polish, z0, method='SLSQP', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14})
        if res.success and res.fun < -best_value / N:
            cx_opt = res.x[0::2][:N]
            cy_opt = res.x[1::2][:N]
            W_opt = res.x[-1]
            H_opt = 2.0 - W_opt
            r_opt = -res.fun
            if _is_valid(cx_opt, cy_opt, np.full(N, r_opt), W_opt, H_opt):
                best_result = (cx_opt, cy_opt, np.full(N, r_opt), W_opt, H_opt)
                best_value = r_opt * N
    except Exception:
        pass

    cx, cy, r, W, H = best_result
    
    # Final safety shrink
    for _ in range(10):
        r_trial = r * (1 - 1e-8)
        if _is_valid(cx, cy, r_trial, W, H):
            r = r_trial
        else:
            break
            
    return np.column_stack((cx, cy, r))

if __name__ == "__main__":
    res = construct()
    print(res)
