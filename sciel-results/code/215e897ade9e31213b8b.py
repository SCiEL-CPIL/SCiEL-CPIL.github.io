"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _hex_initial(n, jitter=0.0, seed=0):
    """Create a hexagonal close-packed initial layout."""
    rng = np.random.RandomState(seed)
    positions = []
    # Try to fit n circles in a hex pattern
    for cols in range(3, 9):
        rows = int(np.ceil(n / cols))
        if cols * rows < n:
            continue
        sx = 1.0 / cols
        sy = 1.0 / rows
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= n:
                    break
                x = (c + 0.5) * sx
                y = (r + 0.5) * sy
                if r % 2 == 1:
                    x += sx * 0.5
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                positions.append([x, y])
        if len(positions) >= n:
            break
    positions = np.array(positions[:n])
    if jitter > 0:
        positions = positions + rng.uniform(-jitter, jitter, size=positions.shape)
        positions = np.clip(positions, 0.01, 0.99)
    return positions


def _random_initial(n, seed):
    """Random initial positions with minimum separation."""
    rng = np.random.RandomState(seed)
    positions = []
    min_dist = 0.1
    attempts = 0
    while len(positions) < n and attempts < 3000:
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.08, 0.92)
        ok = True
        for px, py in positions:
            if (x - px)**2 + (y - py)**2 < min_dist**2:
                ok = False
                break
        if ok:
            positions.append([x, y])
        attempts += 1
    while len(positions) < n:
        positions.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])
    return np.array(positions[:n])


def _objective(params, n):
    """Objective: -sum(radii) + penalty for violations."""
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Maximize sum of radii
    obj = -np.sum(radii)
    
    # Boundary penalties (smooth)
    vl = np.maximum(0.0, radii - pos[:, 0])
    vr = np.maximum(0.0, pos[:, 0] + radii - 1.0)
    vb = np.maximum(0.0, radii - pos[:, 1])
    vt = np.maximum(0.0, pos[:, 1] + radii - 1.0)
    boundary_pen = np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    
    # Overlap penalties (smooth)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = radii[:, np.newaxis] + radii[np.newaxis, :]
    iu = np.triu_indices(n, k=1)
    overlap_pen = np.sum(np.maximum(0.0, min_dists[iu] - dists[iu])**2)
    
    return obj + 500.0 * (boundary_pen + overlap_pen)


def _solve_one(positions, radii, n, deadline):
    """Run a full optimization sequence for one initial layout."""
    if time.time() > deadline - 30:
        return None
    
    params = np.zeros(3 * n)
    params[:2*n] = positions.flatten()
    params[2*n:] = radii
    
    bounds = [(0.001, 0.999)] * (2 * n) + [(0.001, 0.45)] * n
    
    # Phase 1: L-BFGS-B
    try:
        res = minimize(
            _objective, params, args=(n,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
        )
        params = res.x
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        pass
    else:
        # Phase 2: Local search with random perturbations
        best = params.copy()
        best_score = _objective(best, n)
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)
        
        for it in range(800):
            if time.time() > deadline - 15:
                break
            trial = best.copy()
            k = rng.randint(1, 4)
            idx = rng.choice(len(trial), k, replace=False)
            for i in idx:
                trial[i] += rng.uniform(-0.005, 0.005)
            trial[:2*n] = np.clip(trial[:2*n], 0.001, 0.999)
            trial[2*n:] = np.clip(trial[2*n:], 0.001, 0.45)
            s = _objective(trial, n)
            if s < best_score:
                best = trial
                best_score = s
        params = best
    
    return params


def _fix_and_validate(params, n, tol=1e-9):
    """Ensure no violations; return (result_array, sum_radii)."""
    pos = params[:2*n].reshape(n, 2).copy()
    radii = params[2*n:].copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    # Iteratively fix overlaps
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - tol:
                    excess = md - d
                    # Reduce radii proportionally
                    f = 0.5
                    radii[i] -= excess * f
                    radii[j] -= excess * (1 - f)
                    radii[i] = max(0.0, radii[i])
                    radii[j] = max(0.0, radii[j])
                    changed = True
        if not changed:
            break
    
    # Final boundary check
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    # Validate
    valid = True
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 1] - radii[i] < -tol:
            valid = False
        if pos[i, 0] + radii[i] > 1 + tol or pos[i, 1] + radii[i] > 1 + tol:
            valid = False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            if d < radii[i] + radii[j] - tol:
                valid = False
    
    result = np.zeros(3 * n)
    result[:2*n] = pos.flatten()
    result[2*n:] = radii
    
    return result, valid, np.sum(radii)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    # Generate diverse initial layouts
    initial_configs = []
    
    # Hexagonal layouts
    for seed in range(5):
        for jitter in [0.0, 0.02, 0.05]:
            pos = _hex_initial(N, jitter=jitter, seed=seed)
            # Initial radii: small
            r = np.full(N, 0.02)
            initial_configs.append((pos, r))
    
    # Random layouts
    for seed in range(15):
        pos = _random_initial(N, seed=1000 + seed)
        r = np.full(N, 0.02)
        initial_configs.append((pos, r))
    
    # Grid-like layouts
    for cols in [4, 5, 6]:
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            continue
        sx = 1.0 / cols
        sy = 1.0 / rows
        pos = []
        for rr in range(rows):
            for cc in range(cols):
                if len(pos) >= N:
                    break
                x = (cc + 0.5) * sx
                y = (rr + 0.5) * sy
                pos.append([x, y])
        pos = np.array(pos[:N])
        r = np.full(N, 0.03)
        initial_configs.append((pos, r))
    
    # Process each initial config
    np.random.seed(42)
    order = np.random.permutation(len(initial_configs))
    
    for idx in order:
        if time.time() > deadline - 25:
            break
        
        pos, radii = initial_configs[idx]
        
        try:
            params = _solve_one(pos, radii, N, deadline)
            if params is None:
                continue
            
            result, valid, score = _fix_and_validate(params, N)
            if valid and score > best_score:
                best_score = score
                best_result = result.reshape(N, 3)
                
        except Exception:
            continue
    
    # Fallback if nothing worked
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = min(x, y, 1-x, 1-y) * 0.9
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
