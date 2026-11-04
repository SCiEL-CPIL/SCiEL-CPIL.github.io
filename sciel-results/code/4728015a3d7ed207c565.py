"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 21

def _validate(circles, w, h, tol=1e-7):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    # Vectorized overlap check
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    for i in range(n):
        dx = cx[i] - cx[i+1:]
        dy = cy[i] - cy[i+1:]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i] + r[i+1:] - tol):
            return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _relax_circles(circles, w, h, iterations=200, tol=1e-8):
    """
    Relaxation algorithm to resolve overlaps and boundary violations.
    Uses a more careful sequential update with proper handling.
    """
    n = len(circles)
    pos = np.array(circles[:, :2], dtype=float)
    rad = np.array(circles[:, 2], dtype=float)

    for _ in range(iterations):
        moved = False

        # 1. Resolve boundary violations
        for i in range(n):
            if pos[i, 0] - rad[i] < 0:
                pos[i, 0] = rad[i]
                moved = True
            if pos[i, 0] + rad[i] > w:
                pos[i, 0] = w - rad[i]
                moved = True
            if pos[i, 1] - rad[i] < 0:
                pos[i, 1] = rad[i]
                moved = True
            if pos[i, 1] + rad[i] > h:
                pos[i, 1] = h - rad[i]
                moved = True

        # 2. Resolve overlaps
        for i in range(n):
            dx = pos[i+1:, 0] - pos[i, 0]
            dy = pos[i+1:, 1] - pos[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rad[i] + rad[i+1:]
            overlap = min_dist - dist
            
            for k in range(len(overlap)):
                if overlap[k] > 1e-10:
                    if dist[k] < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = overlap[k] / 2.0
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[i+1+k, 0] += np.cos(angle) * push
                        pos[i+1+k, 1] += np.sin(angle) * push
                    else:
                        push = overlap[k] / 2.0
                        ux = dx[k] / dist[k]
                        uy = dy[k] / dist[k]
                        pos[i, 0] -= ux * push
                        pos[i, 1] -= uy * push
                        pos[i+1+k, 0] += ux * push
                        pos[i+1+k, 1] += uy * push
                    moved = True

        if not moved:
            break

    # Final boundary clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])

    new_circles = np.column_stack((pos, rad))

    if _validate(new_circles, w, h, tol=1e-6):
        return new_circles

    return circles

def _logsumexp_penalty(circles, w, h, epsilon=0.01, beta=1000.0):
    """
    Compute log-sum-exp penalty for overlaps and boundaries.
    Returns (penalty_value, gradient) where gradient is w.r.t. all 3*n variables.
    """
    n = len(circles)
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    
    total_pen = 0.0
    grad = np.zeros(3 * n)
    
    # Collect all constraint violations into a list for LSE
    violations = []  # list of (value, grad_contrib) where grad_contrib is (3n,)
    
    # Pairwise overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d_sq = dx*dx + dy*dy
            if d_sq < 1e-24:
                d_sq = 1e-24
                d = 1e-12
            else:
                d = np.sqrt(d_sq)
            req = r[i] + r[j]
            a = req - d  # positive if overlapping
            
            if a > -epsilon * 2:  # only consider near-violations
                # Gradient of a = req - d
                # da/dcxi = -dx/d, da/cxj = dx/d
                # da/dyi = -dy/d, da/dyj = dy/d
                # da/dri = 1, da/drj = 1
                
                inv_d = 1.0 / d
                g_vec = np.zeros(3*n)
                g_vec[i] = -dx * inv_d
                g_vec[n+i] = -dy * inv_d
                g_vec[j] = dx * inv_d
                g_vec[n+j] = dy * inv_d
                g_vec[2*n+i] = 1.0
                g_vec[2*n+j] = 1.0
                
                violations.append((a, g_vec))
    
    # Boundary constraints
    for i in range(n):
        # Left: max(0, r_i - cx_i)
        a = r[i] - cx[i]
        if a > -epsilon * 2:
            g_vec = np.zeros(3*n)
            g_vec[i] = -1.0
            g_vec[2*n+i] = 1.0
            violations.append((a, g_vec))
        
        # Right: max(0, cx_i + r_i - w)
        a = cx[i] + r[i] - w
        if a > -epsilon * 2:
            g_vec = np.zeros(3*n)
            g_vec[i] = 1.0
            g_vec[2*n+i] = 1.0
            violations.append((a, g_vec))
        
        # Bottom: max(0, r_i - cy_i)
        a = r[i] - cy[i]
        if a > -epsilon * 2:
            g_vec = np.zeros(3*n)
            g_vec[n+i] = -1.0
            g_vec[2*n+i] = 1.0
            violations.append((a, g_vec))
        
        # Top: max(0, cy_i + r_i - h)
        a = cy[i] + r[i] - h
        if a > -epsilon * 2:
            g_vec = np.zeros(3*n)
            g_vec[n+i] = 1.0
            g_vec[2*n+i] = 1.0
            violations.append((a, g_vec))
    
    if not violations:
        return 0.0, grad
    
    # Compute LSE: epsilon * log(sum(exp(v_i / epsilon)))
    # For numerical stability, use log-sum-exp trick
    vals = np.array([v[0] for v in violations])
    max_val = np.max(vals)
    if max_val > epsilon * 50:
        # Overflow protection
        vals_shifted = vals / epsilon - max_val / epsilon
        log_sum = max_val / epsilon + np.log(np.sum(np.exp(vals_shifted)))
    else:
        vals_shifted = vals / epsilon
        log_sum = np.log(np.sum(np.exp(vals_shifted)))
    
    total_pen = beta * epsilon * log_sum
    
    # Gradient: d/dx [beta * epsilon * log(sum(exp(v_i/epsilon)))]
    # = beta * sum(exp(v_i/epsilon) / sum(exp(v_j/epsilon))) * grad(v_i)
    # = beta * sum(sigmoid-like weights) * grad(v_i)
    
    exp_vals = np.exp(vals_shifted)
    weights = exp_vals / np.sum(exp_vals)
    
    for idx, (val, g_vec) in enumerate(violations):
        grad += beta * weights[idx] * g_vec
    
    return total_pen, grad

def _high_precision_polish(circles, w, h, deadline):
    """
    High-precision polish using log-sum-exp penalty with progressively smaller epsilon.
    """
    if time.time() > deadline - 3.0:
        return circles
    
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
    
    best_circles = circles.copy()
    best_value = _total_radius(circles)
    
    # Progressive epsilon reduction with increasing beta
    epsilons = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    betas = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    
    for eps, beta in zip(epsilons, betas):
        if time.time() > deadline - 2.0:
            break
        
        def obj_and_grad(x):
            cx = x[:n]
            cy = x[n:2*n]
            r = np.maximum(x[2*n:], 1e-6)
            
            pen, pen_grad = _logsumexp_penalty(
                np.column_stack([cx, cy, r]), w, h, eps, beta
            )
            
            obj = -np.sum(r) + pen
            grad = np.zeros(3*n)
            grad[2*n:] = -1.0
            grad += pen_grad
            
            return obj, grad
        
        try:
            result = minimize(
                lambda x: obj_and_grad(x)[0],
                x0,
                jac=lambda x: obj_and_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_circles = np.column_stack([
                result.x[:n],
                result.x[n:2*n],
                np.maximum(result.x[2*n:], 1e-6)
            ])
            
            if _validate(new_circles, w, h, tol=1e-7):
                val = _total_radius(new_circles)
                if val > best_value:
                    best_value = val
                    best_circles = new_circles
                x0 = result.x
        except Exception:
            pass
    
    return best_circles

def _make_hexagonal_grid(w, h, rows=None):
    """Create initial hexagonal-like arrangement for 21 circles."""
    if rows is None:
        rows = [5, 4, 5, 4, 3]
    circles = []
    total_rows = len(rows)
    
    for row_i, count in enumerate(rows):
        y = (row_i + 0.5) * h / total_rows
        for col_j in range(count):
            avg_w = w / 5.0 
            offset = 0.5 * avg_w if row_i % 2 == 1 else 0
            x = (col_j + 0.5) * w / 5.0 + offset
            r = 0.35 * min(w/5.0, h/total_rows)
            circles.append([x, y, r])
    
    return np.array(circles[:N])

def _make_grid(w, h, nc, nr, stagger=False):
    """Generate initial circle positions in a grid pattern."""
    positions = []
    for j in range(nr):
        for i in range(nc):
            if len(positions) >= N:
                break
            if stagger and j % 2 == 1:
                x = (i + 0.5) * w / nc + 0.3 * w / nc
            else:
                x = (i + 0.5) * w / nc
            y = (j + 0.5) * h / nr
            positions.append([x, y])
    while len(positions) < N:
        idx = len(positions)
        x = w / 2 + (idx - N / 2) * w / N * 0.5
        y = h / 2 + (idx - N / 2) * h / N * 0.5
        positions.append([x, y])
    
    pos = np.array(positions[:N])
    r = 0.35 * min(w/nc, h/nr)
    rad = np.full(N, r)
    return np.column_stack((pos, rad))

def _penalty_optimize(circles, w, h, deadline, beta_start=10.0, beta_end=1000.0, n_beta_steps=8):
    """
    Penalty-based optimization using L-BFGS-B with quadratic penalties.
    """
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))

    def objective_and_grad(x, beta):
        cx = x[:n]
        cy = x[n:2*n]
        r = np.maximum(x[2*n:], 1e-6)
        
        obj = -np.sum(r)
        grad = np.zeros_like(x)
        grad[2*n:] = -1.0
        
        # Boundary penalty
        for i in range(n):
            if cx[i] - r[i] < 0:
                val = cx[i] - r[i]
                obj += beta * val**2
                grad[i] += 2 * beta * val
                grad[2*n+i] -= 2 * beta * val
            if cx[i] + r[i] > w:
                val = cx[i] + r[i] - w
                obj += beta * val**2
                grad[i] += 2 * beta * val
                grad[2*n+i] += 2 * beta * val
            if cy[i] - r[i] < 0:
                val = cy[i] - r[i]
                obj += beta * val**2
                grad[n+i] += 2 * beta * val
                grad[2*n+i] -= 2 * beta * val
            if cy[i] + r[i] > h:
                val = cy[i] + r[i] - h
                obj += beta * val**2
                grad[n+i] += 2 * beta * val
                grad[2*n+i] += 2 * beta * val
        
        # Overlap penalty (vectorized)
        for i in range(n):
            dx = cx[i] - cx[i+1:]
            dy = cy[i] - cy[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[i+1:]
            overlap = np.maximum(0, min_dist - dist)
            
            mask = overlap > 0
            if np.any(mask):
                obj += beta * np.sum(overlap[mask]**2)
                
                d_dist_dx = dx[mask] / dist[mask]
                d_dist_dy = dy[mask] / dist[mask]
                
                grad[i] += 2 * beta * np.sum(overlap[mask] * d_dist_dx)
                grad[n+i] += 2 * beta * np.sum(overlap[mask] * d_dist_dy)
                grad[2*n+i] += 2 * beta * np.sum(overlap[mask])
                
                idxs = np.arange(i+1, n)[mask]
                np.add.at(grad, idxs, 2 * beta * overlap[mask] * d_dist_dx)
                np.add.at(grad, n + idxs, 2 * beta * overlap[mask] * d_dist_dy)
                np.add.at(grad, 2*n + idxs, 2 * beta * overlap[mask])

        return obj, grad

    best_result = None
    best_val = np.inf
    
    for step in range(n_beta_steps):
        if time.time() > deadline - 2.0:
            break
        beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_beta_steps - 1))
        
        try:
            result = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                x0,
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 80}
            )
            val = result.fun
            if val < best_val:
                best_val = val
                best_result = result
            x0 = result.x
        except Exception:
            pass

    if best_result is not None:
        pos = best_result.x[:2*n].reshape(n, 2)
        rad = np.maximum(best_result.x[2*n:], 1e-6)
        return np.column_stack((pos, rad))
    return None

def _contact_graph_optimize(circles, w, h, deadline):
    """
    Contact-graph based optimization: identify contact pairs and use
    constrained optimization to maximize radii subject to contact constraints.
    """
    if time.time() > deadline - 5.0:
        return circles
    
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Identify contact pairs (distance <= r_i + r_j + tol)
    contact_tol = 1e-4
    contacts = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            if d <= rad[i] + rad[j] + contact_tol:
                contacts.append((i, j))
    
    # Also identify boundary contacts
    boundary_contacts = []
    for i in range(n):
        if pos[i, 0] - rad[i] < contact_tol:
            boundary_contacts.append((i, 'left'))
        if pos[i, 0] + rad[i] > w - contact_tol:
            boundary_contacts.append((i, 'right'))
        if pos[i, 1] - rad[i] < contact_tol:
            boundary_contacts.append((i, 'bottom'))
        if pos[i, 1] + rad[i] > h - contact_tol:
            boundary_contacts.append((i, 'top'))
    
    # Use SLSQP to maximize sum of radii subject to non-overlap constraints
    x0 = np.concatenate([pos.ravel(), rad])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
    for i in range(n):
        bounds.append((1e-6, min(w, h)/2))
    
    def objective(x):
        return -np.sum(x[2*n:])
    
    def grad_objective(x):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g
    
    # Inequality constraints: dist_ij >= r_i + r_j
    def make_constraint(i, j):
        def constraint(x):
            dx = x[i] - x[j]
            dy = x[n+i] - x[n+j]
            d = np.sqrt(dx*dx + dy*dy)
            return d - x[2*n+i] - x[2*n+j]
        return constraint
    
    constraints = [{'type': 'ineq', 'fun': make_constraint(i, j)} for i, j in contacts]
    
    # Boundary constraints
    def make_boundary_constraint(i, side):
        def constraint(x):
            if side == 'left':
                return x[i] - x[2*n+i]
            elif side == 'right':
                return w - x[i] - x[2*n+i]
            elif side == 'bottom':
                return x[n+i] - x[2*n+i]
            elif side == 'top':
                return h - x[n+i] - x[2*n+i]
        return constraint
    
    for i, side in boundary_contacts:
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, side)})
    
    try:
        result = minimize(
            objective, x0, jac=grad_objective,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success:
            new_circles = np.column_stack([
                result.x[:n],
                result.x[n:2*n],
                np.maximum(result.x[2*n:], 1e-6)
            ])
            if _validate(new_circles, w, h, tol=1e-6):
                return new_circles
    except Exception:
        pass
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_value = 0.0
    best_result = None
    best_w = None
    best_h = None

    # Strategy 1: Fixed aspect ratios with multiple initializations
    aspect_ratios = [1.0, 1.5, 2.0/3, 0.75, 1.25, 1.333, 0.667, 0.5, 2.0, 1.1, 0.9, 1.4, 0.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15.0:
            break
        
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try hexagonal init
        if time.time() < deadline - 8.0:
            circles = _make_hexagonal_grid(w, h)
            circles = _relax_circles(circles, w, h, iterations=100)
            result = _penalty_optimize(circles, w, h, deadline)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result
                        best_w = w
                        best_h = h

        # Try grid inits
        layouts = [
            (5, 4, True),
            (4, 5, True),
            (5, 4, False),
            (4, 5, False),
            (6, 4, True),
        ]
        for (nc, nr, stagger) in layouts:
            if time.time() > deadline - 8.0:
                break
            circles = _make_grid(w, h, nc, nr, stagger)
            circles = _relax_circles(circles, w, h, iterations=100)
            result = _penalty_optimize(circles, w, h, deadline)
            if result is not None:
                if _validate(result, w, h):
                    value = _total_radius(result)
                    if value > best_value:
                        best_value = value
                        best_result = result
                        best_w = w
                        best_h = h

    # Strategy 2: Contact-graph optimization on best result
    if best_result is not None and time.time() < deadline - 10.0:
        cg_result = _contact_graph_optimize(best_result, best_w, best_h, deadline)
        if cg_result is not None:
            if _validate(cg_result, best_w, best_h, tol=1e-6):
                value = _total_radius(cg_result)
                if value > best_value:
                    best_value = value
                    best_result = cg_result

    # Strategy 3: High-precision polish with log-sum-exp
    if best_result is not None and time.time() < deadline - 8.0:
        polished = _high_precision_polish(best_result, best_w, best_h, deadline)
        if polished is not None:
            if _validate(polished, best_w, best_h, tol=1e-7):
                value = _total_radius(polished)
                if value > best_value:
                    best_value = value
                    best_result = polished

    # Strategy 4: Final micro-polish
    if best_result is not None and time.time() < deadline - 5.0:
        micro = _penalty_optimize(best_result, best_w, best_h, deadline,
                                  beta_start=1000.0, beta_end=100000.0, n_beta_steps=3)
        if micro is not None:
            if _validate(micro, best_w, best_h, tol=1e-7):
                value = _total_radius(micro)
                if value > best_value:
                    best_value = value
                    best_result = micro

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
