import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26
EPS = 1e-8
DEADLINE = 145.0  # safety margin before 150s kill

def _objective(params):
    """Maximize sum of radii, so minimize -sum(r)."""
    r = params[2::3]
    return -np.sum(r)

def _objective_grad(params):
    """Gradient of objective."""
    g = np.zeros_like(params)
    g[2::3] = -1.0
    return g

def _constraints(params):
    """
    Returns a list of constraint dictionaries for SLSQP.
    Inequality constraints: fun(x) >= 0
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    constraints = []
    
    def wall_fun(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        c1 = xx - rr
        c2 = 1.0 - xx - rr
        c3 = yy - rr
        c4 = 1.0 - yy - rr
        return np.concatenate([c1, c2, c3, c4])
        
    def wall_jac(p):
        jac = np.zeros((4*N, 3*N))
        for i in range(N):
            jac[i, 3*i] = 1.0
            jac[i, 3*i+2] = -1.0
        for i in range(N):
            jac[N+i, 3*i] = -1.0
            jac[N+i, 3*i+2] = -1.0
        for i in range(N):
            jac[2*N+i, 3*i+1] = 1.0
            jac[2*N+i, 3*i+2] = -1.0
        for i in range(N):
            jac[3*N+i, 3*i+1] = -1.0
            jac[3*N+i, 3*i+2] = -1.0
        return jac

    constraints.append({'type': 'ineq', 'fun': wall_fun, 'jac': wall_jac})
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    def pair_fun(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        
        dx = xx[iu[0]] - xx[iu[1]]
        dy = yy[iu[0]] - yy[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        dist_safe = np.maximum(dist, 1e-8)
        return dist_safe - (rr[iu[0]] + rr[iu[1]])
        
    def pair_jac(p):
        pp = p.reshape(N, 3)
        xx = pp[:, 0]
        yy = pp[:, 1]
        rr = pp[:, 2]
        
        dx = xx[iu[0]] - xx[iu[1]]
        dy = yy[iu[0]] - yy[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        dist_safe = np.maximum(dist, 1e-8)
        
        jac = np.zeros((n_pairs, 3*N))
        
        idx_i = iu[0]
        idx_j = iu[1]
        
        jac[:, 3*idx_i] = dx / dist_safe
        jac[:, 3*idx_j] = -dx / dist_safe
        jac[:, 3*idx_i + 1] = dy / dist_safe
        jac[:, 3*idx_j + 1] = -dy / dist_safe
        jac[:, 3*idx_i + 2] = -1.0
        jac[:, 3*idx_j + 2] = -1.0
        
        return jac

    constraints.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})
    
    def min_r_fun(p):
        return p[2::3] - 1e-6
        
    def min_r_jac(p):
        jac = np.zeros((N, 3*N))
        for i in range(N):
            jac[i, 3*i+2] = 1.0
        return jac
        
    constraints.append({'type': 'ineq', 'fun': min_r_fun, 'jac': min_r_jac})
    
    return constraints

def _init_voronoi(seed):
    rng = np.random.RandomState(seed)
    points = rng.uniform(0.1, 0.9, size=(N, 2))
    
    radii = np.zeros(N)
    for i in range(N):
        dw = np.min([points[i,0], 1-points[i,0], points[i,1], 1-points[i,1]])
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        dn = np.min(dists)
        radii[i] = 0.5 * np.min([dw, dn]) * 0.9
        
    params = np.zeros(3*N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-5, 1e-5, size=params.size)
    return params

def _init_grid(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    xs, ys = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), 
                         np.linspace(0.5/side, 1-0.5/side, side))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    points = points[:N]
    
    r = 0.5 / side * 0.95
    radii = np.full(N, r)
    
    params = np.zeros(3*N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params

def _init_hex(seed):
    """Hexagonal packing initialization."""
    rng = np.random.RandomState(seed)
    # Hexagonal arrangement for 26 circles
    # 5 rows: 5+5+5+5+6 = 26
    rows = [5, 5, 5, 5, 6]
    spacing = 0.19
    points = []
    for row_idx, n_in_row in enumerate(rows):
        y = 0.1 + row_idx * spacing * np.sqrt(3) / 2
        offset = spacing / 2 if row_idx % 2 else 0
        for col in range(n_in_row):
            x = 0.1 + offset + col * spacing
            points.append([x, y])
    points = np.array(points[:N])
    
    # Adjust to fit in unit square
    if points.max() > 0.9:
        scale = 0.8 / points.max()
        points = points * scale + 0.1
    
    r = spacing * 0.48
    radii = np.full(N, r)
    
    params = np.zeros(3*N)
    params[0::3] = points[:, 0]
    params[1::3] = points[:, 1]
    params[2::3] = radii
    
    params += rng.uniform(-1e-4, 1e-4, size=params.size)
    return params

def _enforce_validity(params):
    """
    Post-process to ensure strict validity.
    If SLSQP leaves slight violations, we fix them by shrinking radii.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0, 1)
    p[:, 1] = np.clip(p[:, 1], 0, 1)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    wall_dist = np.minimum(np.minimum(p[:, 0], 1-p[:, 0]), 
                           np.minimum(p[:, 1], 1-p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], wall_dist)
    
    for _ in range(50):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx*dx + dy*dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-9
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] < p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
    
    return p

def _energy(params):
    """
    Compute the energy (negative sum of radii + penalty for violations).
    Lower energy is better.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]
    
    # Base objective: negative sum of radii
    e = -np.sum(r)
    
    # Wall violations
    wall_pen = 0.0
    for i in range(N):
        if r[i] > x[i]:
            wall_pen += (r[i] - x[i]) ** 2
        if r[i] > 1 - x[i]:
            wall_pen += (r[i] - (1 - x[i])) ** 2
        if r[i] > y[i]:
            wall_pen += (r[i] - y[i]) ** 2
        if r[i] > 1 - y[i]:
            wall_pen += (r[i] - (1 - y[i])) ** 2
    
    # Pairwise overlap violations
    pair_pen = 0.0
    iu = np.triu_indices(N, k=1)
    dx = x[iu[0]] - x[iu[1]]
    dy = y[iu[0]] - y[iu[1]]
    dist = np.sqrt(dx*dx + dy*dy)
    overlap = (r[iu[0]] + r[iu[1]]) - dist
    overlap = np.maximum(overlap, 0)
    pair_pen = np.sum(overlap ** 2)
    
    return e + 100.0 * (wall_pen + pair_pen)

def _parallel_tempering(rng, start_params, n_replicas=8, n_sweeps=2000, 
                        base_temp=0.05, temp_factor=1.5, time_limit=120.0):
    """
    Parallel tempering Monte Carlo for circle packing.
    """
    start_time = time.time()
    
    # Create replicas at different temperatures
    replicas = [start_params.copy() for _ in range(n_replicas)]
    energies = [_energy(replicas[i]) for i in range(n_replicas)]
    temps = [base_temp * (temp_factor ** i) for i in range(n_replicas)]
    
    # Track best solution
    best_params = start_params.copy()
    best_energy = _energy(start_params)
    
    swap_acc = 0
    move_acc = 0
    total_swaps = 0
    total_moves = 0
    
    for sweep in range(n_sweeps):
        # Check time
        if time.time() - start_time > time_limit - 5:
            break
        
        # Perform moves in each replica
        for rep in range(n_replicas):
            if time.time() - start_time > time_limit - 5:
                break
            
            p = replicas[rep]
            E_old = energies[rep]
            
            # Choose random move type
            move_type = rng.random()
            
            if move_type < 0.4:
                # Position move: move a random circle
                i = rng.randint(N)
                dx = rng.normal(0, 0.01)
                dy = rng.normal(0, 0.01)
                
                old_x, old_y = p[3*i], p[3*i+1]
                p[3*i] = np.clip(old_x + dx, 0, 1)
                p[3*i+1] = np.clip(old_y + dy, 0, 1)
                
                E_new = _energy(p)
                dE = E_new - E_old
                
                if dE < 0 or rng.random() < np.exp(-dE / temps[rep]):
                    total_moves += 1
                else:
                    p[3*i] = old_x
                    p[3*i+1] = old_y
                    E_new = E_old
            
            elif move_type < 0.8:
                # Radius move: change radius of a random circle
                i = rng.randint(N)
                dr = rng.normal(0, 0.005)
                
                old_r = p[3*i+2]
                p[3*i+2] = np.clip(old_r + dr, 1e-6, 0.5)
                
                E_new = _energy(p)
                dE = E_new - E_old
                
                if dE < 0 or rng.random() < np.exp(-dE / temps[rep]):
                    total_moves += 1
                else:
                    p[3*i+2] = old_r
                    E_new = E_old
            
            else:
                # Swap two circles' positions
                i, j = rng.choice(N, size=2, replace=False)
                
                old_xi, old_yi = p[3*i], p[3*i+1]
                old_xj, old_yj = p[3*j], p[3*j+1]
                
                p[3*i], p[3*i+1] = old_xj, old_yj
                p[3*j], p[3*j+1] = old_xi, old_yi
                
                E_new = _energy(p)
                dE = E_new - E_old
                
                if dE < 0 or rng.random() < np.exp(-dE / temps[rep]):
                    total_moves += 1
                else:
                    p[3*i], p[3*i+1] = old_xi, old_yi
                    p[3*j], p[3*j+1] = old_xj, old_yj
                    E_new = E_old
            
            energies[rep] = E_new
            
            # Update best
            if E_new < best_energy:
                best_energy = E_new
                best_params = p.copy()
        
        # Attempt swap between adjacent replicas
        if n_replicas > 1:
            swap_idx = rng.randint(n_replicas - 1)
            i, j = swap_idx, swap_idx + 1
            
            Ei, Ej = energies[i], energies[j]
            Ti, Tj = temps[i], temps[j]
            
            # Metropolis criterion for swap
            dE_swap = (Ei / Tj + Ej / Ti) - (Ei / Ti + Ej / Tj)
            
            if dE_swap < 0 or rng.random() < np.exp(-dE_swap):
                # Swap replicas
                replicas[i], replicas[j] = replicas[j], replicas[i]
                energies[i], energies[j] = energies[j], energies[i]
                swap_acc += 1
                total_swaps += 1
        
        # Every 50 sweeps, try to refine the best replica with SLSQP
        if (sweep + 1) % 50 == 0 and time.time() - start_time < time_limit - 10:
            # Find the best replica
            best_rep = np.argmin(energies)
            p_ref = _enforce_validity(replicas[best_rep]).flatten()
            
            try:
                res = minimize(
                    _objective,
                    p_ref,
                    method='SLSQP',
                    jac=_objective_grad,
                    bounds=[(0, 1), (0, 1), (1e-6, 0.5)] * N,
                    constraints=_constraints(np.zeros(3*N)),
                    options={'maxiter': 50, 'ftol': 1e-8}
                )
                
                p_res = _enforce_validity(res.x)
                E_res = _energy(p_res.flatten())
                
                if E_res < best_energy:
                    best_energy = E_res
                    best_params = p_res.flatten()
                    # Update the best replica with the refined solution
                    replicas[best_rep] = p_res.flatten()
                    energies[best_rep] = E_res
                    
            except Exception:
                pass
    
    return best_params, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_score = -np.inf
    
    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    cons = _constraints(np.zeros(3*N))
    
    # List of initializers
    initializers = [
        lambda s: _init_voronoi(s),
        lambda s: _init_grid(s),
        lambda s: _init_hex(s),
    ]
    
    # Phase 1: Parallel tempering with multiple seeds
    pt_seeds = [42, 123, 7, 999, 555]
    
    for seed in pt_seeds:
        if time.time() > deadline - 20:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 20:
                break
                
            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()
            
            try:
                # Run parallel tempering
                time_left = deadline - time.time() - 5
                if time_left < 10:
                    break
                    
                pt_params, pt_energy = _parallel_tempering(
                    np.random.RandomState(seed),
                    x0,
                    n_replicas=6,
                    n_sweeps=1500,
                    base_temp=0.03,
                    temp_factor=1.4,
                    time_limit=time_left
                )
                
                p = _enforce_validity(pt_params)
                s = np.sum(p[:, 2])
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception as e:
                continue
    
    # Phase 2: SLSQP refinement from multiple starting points
    slsqp_seeds = [42, 123, 7, 999, 555, 2024, 31415, 271828, 1234, 5678]
    
    for seed in slsqp_seeds:
        if time.time() > deadline - 10:
            break
            
        for init_fn in initializers:
            if time.time() > deadline - 10:
                break
                
            x0 = init_fn(seed)
            x0 = _enforce_validity(x0).flatten()
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='SLSQP',
                    jac=_objective_grad,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 100, 'ftol': 1e-9}
                )
                
                p = res.x.reshape(N, 3)
                p = _enforce_validity(p)
                s = np.sum(p[:, 2])
                
                if s > best_score:
                    best_score = s
                    best_params = p.copy()
                    
            except Exception as e:
                continue
    
    # Phase 3: Final local improvement
    if best_params is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 3.0:
            p = best_params.copy()
            for _ in range(5):
                if time.time() > deadline - 2.0:
                    break
                improved = False
                for i in range(N):
                    new_r = p[i, 2] * 1.001
                    if new_r > 0.5:
                        continue
                    p[i, 2] = new_r
                    
                    valid = True
                    if p[i, 2] > p[i, 0] or p[i, 2] > 1-p[i, 0] or \
                       p[i, 2] > p[i, 1] or p[i, 2] > 1-p[i, 1]:
                        valid = False
                    if valid:
                        for j in range(N):
                            if i == j:
                                continue
                            dx = p[i, 0] - p[j, 0]
                            dy = p[i, 1] - p[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < p[i, 2] + p[j, 2]:
                                valid = False
                                break
                                
                    if valid:
                        improved = True
                    else:
                        p[i, 2] /= 1.001
                        
                if not improved:
                    break
                    
            s = np.sum(p[:, 2])
            if s > best_score:
                best_score = s
                best_params = p
    
    if best_params is None:
        best_params = _enforce_validity(_init_grid(42)).reshape(N, 3)
        best_score = np.sum(best_params[:, 2])
        
    return best_params
