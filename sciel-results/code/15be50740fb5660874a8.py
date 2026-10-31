"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 26

# EVOLVE-BLOCK-START

def _softplus(x, beta=50.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 20.0/beta, x, np.log1p(np.exp(beta * x)) / beta)

def _is_valid(pos, radii, tol=1e-7):
    """Check feasibility vectorized."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """
    Structural change: Projection-based repair.
    Instead of just shrinking, this iteratively projects radii to satisfy
    pairwise distance constraints and boundary constraints.
    """
    pos = np.asarray(pos, dtype=float).copy()
    radii = np.asarray(radii, dtype=float).copy()
    
    # 1. Boundary clamp
    max_b = np.minimum(np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                       np.minimum(pos[:, 1], 1.0 - pos[:, 1]))
    radii = np.minimum(radii, max_b * 0.9999)
    radii = np.maximum(radii, 1e-8)
    
    # 2. Pairwise projection (Gauss-Seidel style but vectorized inner loop)
    # We want r_i + r_j <= dist_ij
    # Iterate: for each i, r_i <= min_j (dist_ij - r_j)
    for _ in range(100):
        changed = False
        # Compute distances
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        # For each circle i, the max allowed radius given current others is:
        # min( boundary_max_i, min_{j!=i} (dist_ij - r_j) )
        # Note: dist_ij - r_j is the space available for i if j stays fixed.
        allowed = dist - radii[np.newaxis, :]  # allowed[i,j] = dist_ij - r_j
        np.fill_diagonal(allowed, np.inf)
        max_pair_allowed = np.min(allowed, axis=1)
        
        new_radii = np.minimum(radii, max_pair_allowed * 0.9999)
        new_radii = np.minimum(new_radii, max_b * 0.9999)
        new_radii = np.maximum(new_radii, 1e-8)
        
        if np.allclose(new_radii, radii, atol=1e-10):
            break
        radii = new_radii
        changed = True
        
    return pos, radii

def _solve_lp_radii(pos):
    """
    Solve LP to maximize sum of radii for fixed positions.
    This is the exact optimal radii for a given configuration.
    """
    n = len(pos)
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = pos[i]
        for bound_val in [x, y, 1.0-x, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bound_val)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 1e-8)
        else:
            return None
    except Exception:
        return None

def _energy(pos, beta=50.0):
    """
    Smooth energy for position optimization.
    Penalizes boundary violations and overlaps using a reference radius.
    """
    n = len(pos)
    energy = 0.0
    grad = np.zeros((n, 2))
    r_ref = 0.11  # Adjusted based on typical packing density
    
    for i in range(n):
        x, y = pos[i]
        v1 = r_ref - x
        v2 = r_ref - y
        v3 = x - (1.0 - r_ref)
        v4 = y - (1.0 - r_ref)
        
        e_b = _softplus(v1, beta) + _softplus(v2, beta) + _softplus(v3, beta) + _softplus(v4, beta)
        energy += e_b
        
        s1 = 1.0 / (1.0 + np.exp(-beta * v1))
        s2 = 1.0 / (1.0 + np.exp(-beta * v2))
        s3 = 1.0 / (1.0 + np.exp(-beta * v3))
        s4 = 1.0 / (1.0 + np.exp(-beta * v4))
        
        grad[i, 0] += -s1 + s3
        grad[i, 1] += -s2 + s4
    
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy) + 1e-10
            v = 2.0 * r_ref - dist
            e_p = _softplus(v, beta)
            energy += e_p
            
            s = 1.0 / (1.0 + np.exp(-beta * v))
            grad[i, 0] += s * (dx / dist)
            grad[i, 1] += s * (dy / dist)
            grad[j, 0] += -s * (dx / dist)
            grad[j, 1] += -s * (dy / dist)
    
    return energy, grad

def _initial_hexagonal():
    """Hexagonal grid initialization."""
    pos = np.zeros((N, 2))
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N: break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k, 0] = x
                pos[k, 1] = y
                k += 1
        if k >= N: break
    # Fill remaining
    while k < N:
        pos[k, 0] = np.random.uniform(0.1, 0.9)
        pos[k, 1] = np.random.uniform(0.1, 0.9)
        k += 1
    return pos

def _initial_grid():
    """Grid initialization."""
    pos = np.zeros((N, 2))
    side = 6
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            pos[idx, 0] = (i + 0.5) / side
            pos[idx, 1] = (j + 0.5) / side
            idx += 1
    return pos

def _initial_best_fit():
    """Greedy best-fit initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # Start with a large central circle? No, just random greedy.
    rng = np.random.RandomState(42)
    for i in range(N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(2000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            max_r = min(x, 1.0-x, y, 1.0-y)
            for j in range(i):
                d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                max_r = min(max_r, d - radii[j])
            if max_r > best_r:
                best_r = max_r
                best_x, best_y = x, y
        pos[i, 0] = best_x
        pos[i, 1] = best_y
        radii[i] = max(best_r, 0.01)
    return pos, radii

def _optimize_positions(pos_init, beta=50.0, maxiter=500):
    """L-BFGS-B optimization."""
    def obj_and_grad(flat):
        pos = flat.reshape(N, 2)
        e, g = _energy(pos, beta)
        return e, g.flatten()
    
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(obj_and_grad, pos_init.flatten(), method='L-BFGS-B',
                       jac=True, bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000})
        return res.x.reshape(N, 2)
    except Exception:
        return pos_init

def _anneal_positions(pos, beta_start=20.0, beta_end=100.0, steps=300, temp_start=0.05, temp_end=0.001, deadline=0):
    """Simulated Annealing."""
    current = pos.copy()
    current_energy, _ = _energy(current, beta_start)
    best = current.copy()
    best_energy = current_energy
    
    for step in range(steps):
        if time.time() > deadline:
            break
        t = step / max(steps - 1, 1)
        beta = beta_start + (beta_end - beta_start) * t
        temp = temp_start + (temp_end - temp_start) * t
        
        new_pos = current.copy()
        i = np.random.randint(N)
        new_pos[i, 0] += np.random.normal(0, temp)
        new_pos[i, 1] += np.random.normal(0, temp)
        new_pos[i, 0] = np.clip(new_pos[i, 0], 0.0, 1.0)
        new_pos[i, 1] = np.clip(new_pos[i, 1], 0.0, 1.0)
        
        new_energy, _ = _energy(new_pos, beta)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            current = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
                
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    np.random.seed(12345)
    
    best_pos = None
    best_radii = None
    best_value = 0.0
    
    # Initializations
    inits = [
        ("hex", _initial_hexagonal()),
        ("grid", _initial_grid()),
    ]
    
    # Add best-fit init
    pos_bf, rad_bf = _initial_best_fit()
    inits.append(("bf", pos_bf))
    
    for name, pos_init in inits:
        if time.time() > deadline - 20:
            break
            
        # 1. Gradient Descent
        pos_opt = _optimize_positions(pos_init, beta=50.0, maxiter=300)
        
        # 2. Annealing
        pos_ann = _anneal_positions(pos_opt, deadline=deadline - 15)
        
        # Choose better
        e_opt, _ = _energy(pos_opt, beta=50.0)
        e_ann, _ = _energy(pos_ann, beta=50.0)
        pos_best = pos_opt if e_opt <= e_ann else pos_ann
        
        # 3. Solve LP for optimal radii
        radii = _solve_lp_radii(pos_best)
        
        if radii is not None:
            # 4. Post-process (Projection) to ensure strict feasibility if LP was slightly off
            pos_pp, radii_pp = _post_process(pos_best, radii)
            
            if _is_valid(pos_pp, radii_pp, tol=1e-6):
                val = np.sum(radii_pp)
                if val > best_value:
                    best_value = val
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
        
        # Also try post-processing the LP result directly without re-optimizing positions
        # (Sometimes LP gives a slightly infeasible solution due to numerical issues,
        #  and simple projection fixes it while retaining most value)
        if radii is not None:
             pos_pp2, radii_pp2 = _post_process(pos_best, radii)
             if _is_valid(pos_pp2, radii_pp2, tol=1e-6):
                 val2 = np.sum(radii_pp2)
                 if val2 > best_value:
                     best_value = val2
                     best_pos = pos_pp2.copy()
                     best_radii = radii_pp2.copy()

    # Fallback if nothing worked
    if best_pos is None:
        pos_init = _initial_grid()
        radii = _solve_lp_radii(pos_init)
        if radii is None:
            radii = np.full(N, 0.05)
            pos_init, radii = _post_process(pos_init, radii)
        best_pos = pos_init
        best_radii = radii
        best_value = np.sum(radii)
        
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[i, 0]
        result[i, 1] = best_pos[i, 1]
        result[i, 2] = best_radii[i]
        
    return result

# EVOLVE-BLOCK-END
