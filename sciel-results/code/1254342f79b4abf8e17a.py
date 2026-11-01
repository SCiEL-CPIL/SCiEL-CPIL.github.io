import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 32
_DEADLINE = 142.0  # wall-clock safety margin (evaluator kills at 150s)

def _validity_check(arr, eps=1e-9):
    """Return True if all circles are disjoint (interiors) and inside [0,1]^2."""
    if arr is None or len(arr) != N:
        return False
    x = arr[:, 0]; y = arr[:, 1]; r = arr[:, 2]
    if np.any(r <= 0):
        return False
    # inside square
    if np.any(x - r < -eps) or np.any(x + r > 1 + eps) or np.any(y - r < -eps) or np.any(y + r > 1 + eps):
        return False
    # disjoint interiors
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            if d2 < (r[i] + r[j]) ** 2 - eps:
                return False
    return True

def _score(arr):
    """Sum of radii if valid, else -inf."""
    if _validity_check(arr):
        return float(np.sum(arr[:, 2]))
    return -np.inf

def _hex_seed():
    """Staggered hexagonal-ish lattice trimmed to N circles."""
    pts = []
    row_h = 0.8660254  # sqrt(3)/2
    cols = 0
    centers = []
    while len(centers) < N + 10:
        y = row_h * cols
        xoff = 0.5 if (cols % 2) else 0.0
        for i in range(0, 20):
            centers.append((xoff + i, y))
            if len(centers) >= N + 10:
                break
        cols += 1
    centers = centers[:N]
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    # normalize into [0,1] with margin
    cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-12)
    cy = (cy - cy.min()) / (cy.max() - cy.min() + 1e-12)
    # base radius: min distance to boundary /2, and half min pair distance
    r = np.full(N, 0.0)
    for i in range(N):
        dmin = min(cx[i], 1 - cx[i], cy[i], 1 - cy[i])
        r[i] = dmin * 0.5
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            r[i] = min(r[i], d * 0.5)
    return np.column_stack([cx, cy, r])

def _random_seed(rng):
    """Generate random positions and small initial radii."""
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.01)
    return np.column_stack([x, y, r])

def _get_contact_pairs(x, y):
    """Get contact graph edges using Delaunay triangulation."""
    pts = np.column_stack([x, y])
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback: all pairs within some distance
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < 0.3:
                    edges.append((i, j))
        return edges

def _objective(v, edges):
    """
    Objective: -sum(r) + smooth penalties for overlaps and wall violations.
    Uses soft-min (log-sum-exp) for smoothness.
    """
    x = v[0:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for overlaps using soft hinge
    # For each pair, penalty = max(0, r_i + r_j - d_ij)^2
    # Use smooth approximation: softplus(beta*(r_i+r_j-d))^2 / beta^2
    beta = 50.0
    pen_pair = 0.0
    for (i, j) in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy) + 1e-12
        s = r[i] + r[j]
        overlap = s - d
        # softplus(beta * overlap)
        if overlap > 0:
            sp = np.log1p(np.exp(beta * overlap)) / beta
        else:
            sp = overlap + np.log1p(np.exp(beta * overlap)) / beta
        pen_pair += sp * sp
    
    # Penalty for wall violations
    pen_wall = 0.0
    for i in range(N):
        dl = x[i]
        dr = 1.0 - x[i]
        db = y[i]
        dt = 1.0 - y[i]
        # r_i should be <= min(dl, dr, db, dt)
        # penalty = max(0, r_i - min(dl,dr,db,dt))^2
        # Use soft-min for the min
        vals = np.array([dl, dr, db, dt])
        # soft-min: -logsumexp(-beta * vals) / beta
        m = np.min(vals)
        logsumexp_term = np.log(np.sum(np.exp(-beta * (vals - m)))) + (-beta * m)
        soft_min = -logsumexp_term / beta
        excess = r[i] - soft_min
        if excess > 0:
            sp = np.log1p(np.exp(beta * excess)) / beta
        else:
            sp = excess + np.log1p(np.exp(beta * excess)) / beta
        pen_wall += sp * sp
    
    lambda_pair = 200.0
    lambda_wall = 200.0
    
    return obj + lambda_pair * pen_pair + lambda_wall * pen_wall

def _gradient(v, edges):
    """
    Gradient of the objective function.
    """
    x = v[0:N].copy()
    y = v[N:2*N].copy()
    r = v[2*N:3*N].copy()
    
    grad = np.zeros(3*N)
    
    # Gradient of -sum(r)
    grad[2*N:3*N] = -1.0
    
    beta = 50.0
    lambda_pair = 200.0
    lambda_wall = 200.0
    
    # Gradient of pair penalties
    for (i, j) in edges:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy) + 1e-12
        s = r[i] + r[j]
        overlap = s - d
        
        # d/d(overlap) of softplus(beta*overlap)^2
        # softplus(t) = log(1+exp(t))
        # d/dt softplus(t) = sigmoid(t) = 1/(1+exp(-t))
        t = beta * overlap
        if t > 50:
            sig = 1.0
        elif t < -50:
            sig = 0.0
        else:
            sig = 1.0 / (1.0 + np.exp(-t))
        
        # d/d(overlap) of [softplus(beta*overlap)]^2
        # = 2 * softplus(beta*overlap) * beta * sig
        if overlap > 0:
            sp = np.log1p(np.exp(t)) / beta
        else:
            sp = overlap + np.log1p(np.exp(t)) / beta
        d_sp_d_overlap = 2 * sp * sig
        
        # d(overlap)/d(x_i) = -dx/d, d(overlap)/d(x_j) = dx/d
        # d(overlap)/d(y_i) = -dy/d, d(overlap)/d(y_j) = dy/d
        # d(overlap)/d(r_i) = 1, d(overlap)/d(r_j) = 1
        
        if d > 1e-12:
            grad[0:N][i] += lambda_pair * d_sp_d_overlap * (-dx / d)
            grad[N:2*N][i] += lambda_pair * d_sp_d_overlap * (-dy / d)
            grad[0:N][j] += lambda_pair * d_sp_d_overlap * (dx / d)
            grad[N:2*N][j] += lambda_pair * d_sp_d_overlap * (dy / d)
        grad[2*N:3*N][i] += lambda_pair * d_sp_d_overlap
        grad[2*N:3*N][j] += lambda_pair * d_sp_d_overlap
    
    # Gradient of wall penalties
    for i in range(N):
        dl = x[i]
        dr = 1.0 - x[i]
        db = y[i]
        dt = 1.0 - y[i]
        vals = np.array([dl, dr, db, dt])
        m = np.min(vals)
        # soft-min derivative: weights are softmax(-beta * (vals - m))
        exp_vals = np.exp(-beta * (vals - m))
        weights = exp_vals / np.sum(exp_vals)
        # d(soft_min)/d(val_k) = -weights[k]
        
        excess = r[i] - (-np.log(np.sum(exp_vals)) / beta - m)  # this is approximate
        # Actually, let me recompute soft_min properly
        # soft_min = -log(sum(exp(-beta*vals)))/beta
        # = -(log(sum(exp(-beta*(vals-m)))) + (-beta*m)) / beta
        # = -log(sum(exp(-beta*(vals-m))))/beta + m
        log_sum = np.log(np.sum(exp_vals))
        soft_min = m - log_sum / beta
        
        excess = r[i] - soft_min
        t = beta * excess
        if t > 50:
            sig = 1.0
        elif t < -50:
            sig = 0.0
        else:
            sig = 1.0 / (1.0 + np.exp(-t))
        
        if excess > 0:
            sp = np.log1p(np.exp(t)) / beta
        else:
            sp = excess + np.log1p(np.exp(t)) / beta
        d_sp_d_excess = 2 * sp * sig
        
        # d(excess)/d(r_i) = 1
        grad[2*N:3*N][i] += lambda_wall * d_sp_d_excess
        
        # d(excess)/d(x_i): 
        # d(soft_min)/d(dl) = -weights[0], d(dl)/d(x_i) = 1
        # d(soft_min)/d(dr) = -weights[1], d(dr)/d(x_i) = -1
        # d(soft_min)/d(db) = -weights[2], d(db)/d(x_i) = 0
        # d(soft_min)/d(dt) = -weights[3], d(dt)/d(x_i) = 0
        # d(excess)/d(x_i) = -d(soft_min)/d(x_i) = -(d(soft_min)/d(dl)*1 + d(soft_min)/d(dr)*(-1))
        # = -(-weights[0] + weights[1]) = weights[0] - weights[1]
        grad[0:N][i] += lambda_wall * d_sp_d_excess * (weights[0] - weights[1])
        
        # d(excess)/d(y_i) = weights[2] - weights[3]
        grad[N:2*N][i] += lambda_wall * d_sp_d_excess * (weights[2] - weights[3])
    
    return grad

def _shrink_to_valid(arr):
    """Given a candidate, shrink radii minimally to guarantee validity."""
    x = arr[:, 0].copy(); y = arr[:, 1].copy(); r = arr[:, 2].copy()
    # ensure inside
    for i in range(N):
        r[i] = min(r[i], x[i], 1 - x[i], y[i], y[i], 1 - y[i])
    # ensure disjoint: iteratively shrink
    for _ in range(200):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                s = r[i] + r[j]
                if d < s:
                    need = s - d
                    tot = r[i] + r[j]
                    if tot <= 0:
                        continue
                    new_tot = max(0.0, d)
                    r[i] = r[i] * new_tot / tot
                    r[j] = r[j] * new_tot / tot
                    changed = True
        if not changed:
            break
    return np.column_stack([x, y, r])

def _polish(arr):
    """High-precision polish pass using tighter tolerances."""
    if not _validity_check(arr):
        arr = _shrink_to_valid(arr)
    
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    # Get contact pairs
    edges = _get_contact_pairs(x, y)
    if len(edges) < 10:
        # Fallback: use all pairs within reasonable distance
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(x[i]-x[j], y[i]-y[j])
                if d < 0.25:
                    edges.append((i, j))
    
    v = np.concatenate([x, y, r])
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    try:
        res = minimize(
            lambda v: _objective(v, edges),
            v,
            method='L-BFGS-B',
            jac=lambda v: _gradient(v, edges),
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        v_opt = res.x
        x_opt = v_opt[0:N]
        y_opt = v_opt[N:2*N]
        r_opt = v_opt[2*N:3*N]
        arr_opt = np.column_stack([x_opt, y_opt, r_opt])
        arr_valid = _shrink_to_valid(arr_opt)
        sc = _score(arr_valid)
        if sc > _score(arr):
            return arr_valid
    except Exception:
        pass
    return arr

def construct():
    t0 = time.time()
    global _DEADLINE
    
    # Baseline from hex seed
    base = _hex_seed()
    base = _shrink_to_valid(base)
    best = base
    best_score = _score(best)
    
    rng = np.random.default_rng(12345)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
    
    restart = 0
    while time.time() - t0 < _DEADLINE - 10:
        # Choose initial configuration
        if restart % 4 == 0:
            # Hex seed with jitter
            x0 = base[:, 0].copy()
            y0 = base[:, 1].copy()
            jit = 0.03
            x0 += rng.uniform(-jit, jit, N)
            y0 += rng.uniform(-jit, jit, N)
            x0 = np.clip(x0, 0.0, 1.0)
            y0 = np.clip(y0, 0.0, 1.0)
            r0 = np.full(N, 0.04)
        elif restart % 4 == 1:
            # Random positions
            arr_init = _random_seed(rng)
            x0 = arr_init[:, 0]
            y0 = arr_init[:, 1]
            r0 = arr_init[:, 2]
        elif restart % 4 == 2:
            # Hex seed, shifted
            x0 = base[:, 0].copy()
            y0 = base[:, 1].copy()
            x0 = (x0 + rng.uniform(0, 0.15)) % 1.0
            y0 = (y0 + rng.uniform(0, 0.15)) % 1.0
            r0 = np.full(N, 0.04)
        else:
            # Best so far with small perturbation
            x0 = best[:, 0].copy()
            y0 = best[:, 1].copy()
            r0 = best[:, 2].copy()
            x0 += rng.normal(0, 0.02, N)
            y0 += rng.normal(0, 0.02, N)
            x0 = np.clip(x0, 0.0, 1.0)
            y0 = np.clip(y0, 0.0, 1.0)
            r0 = np.clip(r0 * rng.uniform(0.9, 1.1), 0.01, 0.5)
        
        v0 = np.concatenate([x0, y0, r0])
        
        # Get contact pairs for this configuration
        edges = _get_contact_pairs(x0, y0)
        
        # Run L-BFGS-B
        try:
            res = minimize(
                lambda v: _objective(v, edges),
                v0,
                method='L-BFGS-B',
                jac=lambda v: _gradient(v, edges),
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            
            if res.success or res.fun < _objective(v0, edges):
                v_opt = res.x
                x_opt = v_opt[0:N]
                y_opt = v_opt[N:2*N]
                r_opt = v_opt[2*N:3*N]
                
                arr_opt = np.column_stack([x_opt, y_opt, r_opt])
                arr_valid = _shrink_to_valid(arr_opt)
                sc = _score(arr_valid)
                
                if sc > best_score:
                    best_score = sc
                    best = arr_valid.copy()
        except Exception:
            pass
        
        restart += 1
    
    # Final polish pass
    if time.time() - t0 < _DEADLINE - 3:
        best = _polish(best)
        best_score = _score(best)
    
    # Final safety: ensure validity
    best = _shrink_to_valid(best)
    if not _validity_check(best):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best = np.array(cells[:N])
    
    return best
