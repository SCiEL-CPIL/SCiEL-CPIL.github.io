"""Circle packing in perimeter-4 rectangle: Voronoi-based radius maximization + SLSQP."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _voronoi_inradii(cx, cy, W, H):
    """
    Compute the maximum radius for each circle given its position,
    considering neighbors and walls. This is the inradius of the
    Voronoi cell for each site.
    """
    n = N
    r_max = np.zeros(n)
    
    for i in range(n):
        # Wall constraints
        r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
        
        # Neighbor constraints
        r_neigh = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            # The max radius for circle i given circle j's position
            # (assuming j has some radius, but for inradius we consider
            # the perpendicular bisector, so r_i + r_j <= d)
            # For the inradius computation, we want the distance from
            # site i to the nearest edge of its Voronoi cell.
            # The Voronoi edge between i and j is the perpendicular bisector.
            # Distance from i to this edge is d/2.
            # So r_i <= d/2 (if all circles have equal radius in the cell)
            # But actually, the constraint is r_i + r_j <= d.
            # For the inradius (max r_i such that circle i fits in its cell),
            # we need r_i <= d/2 for each neighbor j (when r_j = r_i).
            # More precisely, the distance from site i to the Voronoi edge
            # between i and j is d/2. So r_i <= d/2.
            r_neigh = min(r_neigh, d / 2.0)
        
        r_max[i] = min(r_wall, r_neigh)
    
    return r_max


def _objective(z, penalty_weight=500.0):
    """Penalty-based objective for SLSQP with smooth soft-min penalties."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Pairwise overlap penalty with smooth soft-min
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    # Smooth penalty: use softplus-like approximation of max(0, x)^2
    # softplus(x) = log(1+exp(x)) approximates max(0,x) smoothly
    overlap = r_sum_pairs - d_pairs
    # Smooth squared penalty via softplus
    beta = 50.0  # sharpness of smooth approximation
    smooth_overlap_sq = (1.0 / beta) * np.log(1.0 + np.exp(beta * overlap)) ** 2
    # For large negative overlap, this is ~0; for positive, ~overlap^2
    obj += penalty_weight * np.sum(smooth_overlap_sq)

    # Boundary penalty with smooth soft-min
    left_viol = r - x
    right_viol = x + r - w
    bottom_viol = r - y
    top_viol = y + r - h

    all_viol = np.concatenate([left_viol, right_viol, bottom_viol, top_viol])
    smooth_viol_sq = (1.0 / beta) * np.log(1.0 + np.exp(beta * all_viol)) ** 2
    obj += penalty_weight * np.sum(smooth_viol_sq)

    return obj


def _slsqp_refine(z, time_budget, maxiter=500):
    """SLSQP refinement with penalty objective."""
    def obj_func(zv):
        return _objective(zv)

    bounds = [(0, 2)] * (3 * N) + [(0.3, 1.7)]
    
    try:
        res = minimize(obj_func, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 200})
        return res.x
    except Exception:
        return z


def _high_precision_polish(cx, cy, r, W, H, time_budget):
    """
    Final high-precision polish: iteratively adjust positions and radii
    to minimize constraint violations and maximize total radius.
    Uses a smooth objective with very tight tolerances.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    deadline = time.time() + max(0, time_budget - (time.time() - time.time()))
    
    # Phase 1: Fine-tune positions using gradient descent on smooth objective
    for phase in range(3):
        if time.time() > deadline - 2:
            break
        
        lr = 0.001 * (0.5 ** phase)
        for it in range(200):
            if time.time() > deadline - 2:
                break
            
            # Compute gradients numerically (small perturbation)
            eps = 1e-6
            grad_cx = np.zeros(N)
            grad_cy = np.zeros(N)
            grad_r = np.zeros(N)
            
            # Build state vector
            def _build_state(dx, dy, dr):
                s = np.zeros(3 * N + 1)
                s[0::3][:N] = cx + dx
                s[1::3][:N] = cy + dy
                s[2::3][:N] = r + dr
                s[-1] = W
                return s
            
            f0 = _objective(_build_state(np.zeros(N), np.zeros(N), np.zeros(N)))
            
            for i in range(N):
                dx = np.zeros(N); dx[i] = eps
                dy = np.zeros(N); dy[i] = eps
                dr = np.zeros(N); dr[i] = eps
                grad_cx[i] = (_objective(_build_state(dx, np.zeros(N), np.zeros(N))) - f0) / eps
                grad_cy[i] = (_objective(_build_state(np.zeros(N), dy, np.zeros(N))) - f0) / eps
                grad_r[i] = (_objective(_build_state(np.zeros(N), np.zeros(N), dr)) - f0) / eps
            
            # Update
            cx -= lr * grad_cx
            cy -= lr * grad_cy
            r -= lr * grad_r
            
            # Clamp to valid range
            cx = np.clip(cx, 0.01, W - 0.01)
            cy = np.clip(cy, 0.01, H - 0.01)
            r = np.clip(r, 0.001, 0.5)
            
            # Check validity and shrink if needed
            if not _is_valid(cx, cy, r, W, H):
                # Shrink radii slightly
                r *= 0.999
                cx = np.clip(cx, r, W - r)
                cy = np.clip(cy, r, H - r)
        
        # Re-expand radii after position adjustments
        r_expanded = _expand_radii(cx, cy, W, H, deadline)
        if _is_valid(cx, cy, r_expanded, W, H):
            r = r_expanded
    
    # Phase 2: Binary search for optimal uniform radius scaling
    if time.time() < deadline - 1:
        # Try to scale all radii up slightly
        for scale in [1.001, 1.0005, 1.0001]:
            r_test = r * scale
            if _is_valid(cx, cy, r_test, W, H):
                r = r_test
    
    return cx, cy, r


def _expand_radii(cx, cy, W, H, time_budget):
    """
    Iteratively expand each circle's radius to its maximum possible value
    given the current positions of all other circles and the walls.
    Uses the Voronoi inradius concept.
    """
    r = np.zeros(N)
    
    # Initialize with small radii
    for i in range(N):
        r[i] = 0.01
    
    max_rounds = 100
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(N):
            # Compute max radius for circle i
            r_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
            
            r_neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_neigh = min(r_neigh, d - r[j])
            
            r_max = max(0.0, min(r_wall, r_neigh))
            
            if r_max > r[i] + 1e-9:
                # Try to expand
                r_new = r.copy()
                r_new[i] = r_max * 0.999
                if _is_valid(cx, cy, r_new, W, H):
                    r = r_new
                    improved = True
        
        if not improved:
            break
    
    return r


def _jiggle_polish(cx, cy, r, W, H, time_budget):
    """
    Jiggle positions to allow radius expansion.
    Move circles that are constrained, then re-expand radii.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    max_rounds = 200
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        # Find circles that are close to their constraint
        min_margin = np.full(N, np.inf)
        for i in range(N):
            margin_wall = min(cx[i] - r[i], W - cx[i] - r[i], cy[i] - r[i], H - cy[i] - r[i])
            margin_neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                margin_neigh = min(margin_neigh, d - r[i] - r[j])
            min_margin[i] = min(margin_wall, margin_neigh)
        
        # Try to move circles with small margins
        moved = False
        for i in range(N):
            if min_margin[i] > 0.02:
                continue
            
            # Determine direction to move
            ml = cx[i] - r[i]
            mr = W - cx[i] - r[i]
            mb = cy[i] - r[i]
            mt = H - cy[i] - r[i]
            
            # Find nearest neighbor
            best_j = -1
            best_d = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < best_d:
                    best_d = d
                    best_j = j
            
            mg = best_d - r[i] - r[best_j] if best_j >= 0 else np.inf
            
            if mg <= ml and mg <= mr and mg <= mb and mg <= mt and best_j >= 0:
                dx = cx[i] - cx[best_j]
                dy = cy[i] - cy[best_j]
                d = np.sqrt(dx * dx + dy * dy)
                if d > 1e-8:
                    move_x = dx / d
                    move_y = dy / d
                else:
                    move_x = 1.0
                    move_y = 0.0
            elif ml <= mr and ml <= mb and ml <= mt:
                move_x = 1.0
                move_y = 0.0
            elif mr <= ml and mr <= mb and mr <= mt:
                move_x = -1.0
                move_y = 0.0
            elif mb <= ml and mb <= mr and mb <= mt:
                move_x = 0.0
                move_y = 1.0
            else:
                move_x = 0.0
                move_y = -1.0
            
            # Try small moves
            for step in [0.01, 0.005, 0.002, 0.001]:
                new_cx = cx[i] + move_x * step
                new_cy = cy[i] + move_y * step
                if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                    continue
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    dx = new_cx - cx[j]
                    dy = new_cy - cy[j]
                    if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                        valid = False
                        break
                if valid:
                    cx[i] = new_cx
                    cy[i] = new_cy
                    moved = True
                    break
        
        if not moved:
            # Try random jiggles
            rng = np.random.default_rng(rnd)
            for i in range(N):
                if min_margin[i] > 0.01:
                    continue
                angle = rng.uniform(0, 2 * np.pi)
                for step in [0.01, 0.005, 0.002]:
                    new_cx = cx[i] + np.cos(angle) * step
                    new_cy = cy[i] + np.sin(angle) * step
                    if new_cx < r[i] or new_cx > W - r[i] or new_cy < r[i] or new_cy > H - r[i]:
                        continue
                    valid = True
                    for j in range(N):
                        if i == j:
                            continue
                        dx = new_cx - cx[j]
                        dy = new_cy - cy[j]
                        if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - 1e-9:
                            valid = False
                            break
                    if valid:
                        cx[i] = new_cx
                        cy[i] = new_cy
                        moved = True
                        break
                if moved:
                    break
        
        if not moved:
            break
        
        # Re-expand radii
        r_new = _expand_radii(cx, cy, W, H, time_budget)
        if np.sum(r_new) > np.sum(r) + 1e-9:
            r = r_new
    
    return cx, cy, r


def _get_initial_guesses():
    """Generate diverse initial guesses using hexagonal and square grids."""
    guesses = []
    
    # Hexagonal packing initial guesses
    for w0 in [1.0, 0.9, 1.1, 0.8, 1.2, 0.7, 1.3, 0.65]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        z = np.zeros(3 * N + 1)
        grid_cols = 7
        grid_rows = 4
        
        x_spacing = w0 / (grid_cols + 0.5)
        y_spacing = x_spacing * np.sqrt(3) / 2.0
        
        total_height = grid_rows * y_spacing
        if total_height > h0:
            scale = h0 / total_height
            x_spacing *= scale
            y_spacing *= scale
        
        idx = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if idx >= N:
                    break
                cx = (col + 0.5) * x_spacing
                cy = (row + 0.5) * y_spacing
                if row % 2 == 1:
                    cx += x_spacing / 2.0
                
                if cx < 0 or cx > w0 or cy < 0 or cy > h0:
                    continue
                
                r0 = min(x_spacing, y_spacing) * 0.45
                z[3 * idx] = cx
                z[3 * idx + 1] = cy
                z[3 * idx + 2] = r0
                idx += 1
        
        if idx < N:
            for i in range(idx, N):
                z[3 * i] = w0 / 2.0
                z[3 * i + 1] = h0 / 2.0
                z[3 * i + 2] = 0.05
        
        z[-1] = w0
        guesses.append(z)

    # Square grid initial guesses
    side = int(np.ceil(np.sqrt(N)))
    for w0 in [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55]:
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
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
    for _ in range(10):
        w0 = rng.uniform(0.6, 1.4)
        h0 = 2.0 - w0
        z = np.zeros(3 * N + 1)
        for i in range(N):
            r0 = rng.uniform(0.04, 0.14)
            cx = rng.uniform(r0, w0 - r0)
            cy = rng.uniform(r0, h0 - r0)
            z[3 * i] = cx
            z[3 * i + 1] = cy
            z[3 * i + 2] = r0
        z[-1] = w0
        guesses.append(z)

    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    guesses = _get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break

        x = z0[0::3][:N]
        y = z0[1::3][:N]
        r = z0[2::3][:N]
        w = z0[-1]
        h = 2.0 - w

        # Force-directed relaxation to resolve overlaps
        dt = 0.005
        for it in range(500):
            if time.time() > deadline - 25:
                break
            
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Wall repulsion
            left_viol = np.maximum(r - x, 0.0)
            right_viol = np.maximum(x + r - w, 0.0)
            bottom_viol = np.maximum(r - y, 0.0)
            top_viol = np.maximum(y + r - h, 0.0)
            fx -= left_viol * 100
            fx += right_viol * 100
            fy -= bottom_viol * 100
            fy += top_viol * 100
            
            # Pairwise repulsion
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx * dx + dy * dy
            r_sum = r[:, None] + r[None, :]
            
            iu = np.triu_indices(N, k=1)
            d_sq_pairs = d_sq[iu]
            r_sum_pairs = r_sum[iu]
            
            d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
            overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
            
            if np.max(overlap) > 1e-8:
                f_pairs = overlap * 15.0
                dx_pairs = dx[iu]
                dy_pairs = dy[iu]
                ux = dx_pairs / d_pairs
                uy = dy_pairs / d_pairs
                
                for idx in range(len(iu[0])):
                    i_idx = iu[0][idx]
                    j_idx = iu[1][idx]
                    fx[i_idx] += ux[idx] * f_pairs[idx]
                    fy[i_idx] += uy[idx] * f_pairs[idx]
                    fx[j_idx] -= ux[idx] * f_pairs[idx]
                    fy[j_idx] -= uy[idx] * f_pairs[idx]
            
            x += dt * fx
            y += dt * fy
            x = np.clip(x, r, w - r)
            y = np.clip(y, r, h - r)
        
        # SLSQP refinement
        z_relaxed = np.zeros(3 * N + 1)
        z_relaxed[0::3][:N] = x
        z_relaxed[1::3][:N] = y
        z_relaxed[2::3][:N] = r
        z_relaxed[-1] = w
        
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            z_opt = _slsqp_refine(z_relaxed, time.time() + 10, maxiter=300)
            
            x_opt = z_opt[0::3][:N]
            y_opt = z_opt[1::3][:N]
            r_opt = z_opt[2::3][:N]
            w_opt = z_opt[-1]
            h_opt = 2.0 - w_opt
            
            # Expand radii
            r_expanded = _expand_radii(x_opt, y_opt, w_opt, h_opt, deadline - 5)
            
            if _is_valid(x_opt, y_opt, r_expanded, w_opt, h_opt):
                val = float(np.sum(r_expanded))
                if val > best_value:
                    best_value = val
                    best_result = np.zeros(3 * N + 1)
                    best_result[0::3][:N] = x_opt
                    best_result[1::3][:N] = y_opt
                    best_result[2::3][:N] = r_expanded
                    best_result[-1] = w_opt
        
        # Jiggle polish
        if best_result is not None and time.time() < deadline - 10:
            x_j, y_j, r_j = _jiggle_polish(
                best_result[0::3][:N], 
                best_result[1::3][:N], 
                best_result[2::3][:N], 
                best_result[-1], 
                2.0 - best_result[-1], 
                deadline - 5
            )
            
            if _is_valid(x_j, y_j, r_j, best_result[-1], 2.0 - best_result[-1]):
                val = float(np.sum(r_j))
                if val > best_value:
                    best_value = val
                    best_result[0::3][:N] = x_j
                    best_result[1::3][:N] = y_j
                    best_result[2::3][:N] = r_j
        
        # High-precision polish on best result
        if best_result is not None and time.time() < deadline - 3:
            x_hp, y_hp, r_hp = _high_precision_polish(
                best_result[0::3][:N],
                best_result[1::3][:N],
                best_result[2::3][:N],
                best_result[-1],
                2.0 - best_result[-1],
                deadline - 2
            )
            
            if _is_valid(x_hp, y_hp, r_hp, best_result[-1], 2.0 - best_result[-1]):
                val = float(np.sum(r_hp))
                if val > best_value:
                    best_value = val
                    best_result[0::3][:N] = x_hp
                    best_result[1::3][:N] = y_hp
                    best_result[2::3][:N] = r_hp

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Final tiny radius expansion attempt
    for scale in [1.0 + 1e-8, 1.0 + 5e-9, 1.0 + 1e-9]:
        r_test = r * scale
        if _is_valid(x, y, r_test, w, h):
            r = r_test
            break

    return np.column_stack((x, y, r))
