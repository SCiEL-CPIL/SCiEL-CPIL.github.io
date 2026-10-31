"""Circle packing in unit square - maximize sum of radii for 26 circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # safety margin before 150s kill

def _soft_min_penalty(dists, radii, beta):
    """Smooth penalty for overlap: sum over pairs of max(0, r_i + r_j - d_ij)^2
    Using soft-max approximation via log-sum-exp for differentiability."""
    n = len(radii)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            # d_ij is the distance, constraint: d_ij >= r_i + r_j
            # violation = r_i + r_j - d_ij (positive when overlapping)
            # Use smooth max: log-sum-exp trick
            # max(0, v) ≈ (1/beta) * log(1 + exp(beta*v))
            v = radii[i] + radii[j] - dists[i, j]
            # Smooth max(0, v)
            if v > 50.0 / beta:
                penalty += v * v
            else:
                penalty += (v * v) * (1.0 + np.exp(beta * v)) / (1.0 + np.exp(-beta * v)) ** 2
    return penalty

def _boundary_penalty(x, y, r):
    """Penalty for circles going outside [0,1]^2."""
    # x - r >= 0, y - r >= 0, x + r <= 1, y + r <= 1
    pen = 0.0
    for i in range(N):
        vi = x[i] - r[i]
        if vi < 0:
            pen += vi * vi
        vi = y[i] - r[i]
        if vi < 0:
            pen += vi * vi
        vi = x[i] + r[i] - 1.0
        if vi > 0:
            pen += vi * vi
        vi = y[i] + r[i] - 1.0
        if vi > 0:
            pen += vi * vi
    return pen

def _objective(params, beta):
    """Objective: -sum(radii) + penalty terms. Minimize this."""
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Compute pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    
    # Overlap penalty
    pen = _soft_min_penalty(dists, r, beta)
    # Boundary penalty
    pen += _boundary_penalty(x, y, r)
    
    return -np.sum(r) + pen

def _is_valid(params, tol=1e-6):
    """Check if solution is valid: all circles inside unit square, no overlaps."""
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Boundary check
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > 1.0 + tol) or np.any(y + r > 1.0 + tol):
        return False
    if np.any(r < -tol):
        return False
    
    # Overlap check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    for i in range(N):
        for j in range(i+1, N):
            if dists[i, j] < r[i] + r[j] - tol:
                return False
    return True

def _lattice_init():
    """Hexagonal lattice initialization."""
    # For 26 circles, use a 5x6 grid with hexagonal offset
    # or 6x5
    # Let's use a hexagonal packing pattern
    # 26 circles: try 5 columns, 6 rows with hex offset
    cols = 5
    rows = 6
    # Hexagonal spacing
    # In hex packing, horizontal spacing = 2r, vertical = sqrt(3)*r
    # For unit square, fit as many as possible
    # Let's just place them on a grid with slight hexagonal perturbation
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for j in range(rows):
        for i in range(cols):
            if idx >= N:
                break
            # Hexagonal offset: odd rows shifted
            offset = 0.5 if (j % 2 == 1) else 0.0
            x[idx] = (i + offset + 0.5) / (cols + 0.5)
            y[idx] = (j + 0.5) / rows
            r[idx] = 0.5 / (cols + 0.5) * 0.9
            idx += 1
        if idx >= N:
            break
    
    # Scale radii to be consistent
    # Actually, let's just use a simpler approach: grid with small perturbation
    # and let the optimizer figure out the radii
    side = int(np.ceil(np.sqrt(N)))
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    idx = 0
    for j in range(side):
        for i in range(side):
            if idx >= N:
                break
            x[idx] = (i + 0.5) / side
            y[idx] = (j + 0.5) / side
            r[idx] = 0.5 / side * 0.8
            idx += 1
        if idx >= N:
            break
    
    return np.concatenate([x, y, r])

def _random_init():
    """Random initialization within bounds."""
    x = np.random.uniform(0.1, 0.9, N)
    y = np.random.uniform(0.1, 0.9, N)
    r = np.random.uniform(0.05, 0.15, N)
    return np.concatenate([x, y, r])

def _optimize_single(params0, beta_start, beta_end, n_stages, time_limit):
    """Multi-stage L-BFGS-B optimization."""
    params = params0.copy()
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0.01, 0.5)] * N
    
    for stage in range(n_stages):
        if time.time() > time_limit:
            break
        beta = beta_start * (beta_end / beta_start) ** (stage / max(n_stages - 1, 1))
        res = minimize(
            _objective, params, args=(beta,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        params = res.x
    
    return params

def _differential_evolution(params0, time_limit, pop_size=50, max_iter=100):
    """Local Differential Evolution polishing."""
    from scipy.optimize import differential_evolution
    
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0.01, 0.5)] * N
    
    # Use a local DE around the current best
    # We'll do a simplified DE manually to control time
    pop = np.tile(params0, (pop_size, 1))
    # Perturb
    for i in range(pop_size):
        pop[i] = pop[i] + np.random.normal(0, 0.02, 3*N)
        pop[i] = np.clip(pop[i], [b[0] for b in bounds], [b[1] for b in bounds])
    
    best_params = pop[0].copy()
    best_val = _objective(pop[0], 100.0)
    
    for gen in range(max_iter):
        if time.time() > time_limit:
            break
        for i in range(pop_size):
            if time.time() > time_limit:
                break
            # Select three distinct random individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            # Mutation
            mutant = pop[a] + 0.8 * (pop[b] - pop[c])
            mutant = np.clip(mutant, [bo[0] for bo in bounds], [bo[1] for bo in bounds])
            # Evaluate
            val_m = _objective(mutant, 100.0)
            val_i = _objective(pop[i], 100.0)
            # Selection
            if val_m < val_i:
                pop[i] = mutant
                if val_m < best_val:
                    best_val = val_m
                    best_params = mutant.copy()
    
    return best_params

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_params = None
    best_value = -np.inf
    
    # Try multiple initializations
    inits = [_lattice_init()]
    # Add a few random inits
    for _ in range(3):
        inits.append(_random_init())
    
    for init in inits:
        if time.time() > deadline - 30:  # Leave 30s for polishing
            break
        
        # Multi-stage L-BFGS-B
        params = _optimize_single(init, beta_start=5.0, beta_end=100.0, n_stages=3, time_limit=deadline - 20)
        
        # Check validity
        if _is_valid(params):
            val = np.sum(params[2*N:3*N])
            if val > best_value:
                best_value = val
                best_params = params.copy()
        
        # DE polishing if time allows
        if time.time() < deadline - 15:
            params_de = _differential_evolution(params, time_limit=deadline - 10, pop_size=30, max_iter=50)
            if _is_valid(params_de):
                val_de = np.sum(params_de[2*N:3*N])
                if val_de > best_value:
                    best_value = val_de
                    best_params = params_de.copy()
    
    # If no valid solution found, use the lattice init as fallback
    if best_params is None:
        best_params = _lattice_init()
    
    # Format output: (x, y, radius)
    x = best_params[0:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    return np.column_stack([x, y, r])
