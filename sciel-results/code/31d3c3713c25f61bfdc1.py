"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _bounds():
    """Variable bounds for [x1,y1,...,xN,yN,r1,...,rN]."""
    b = []
    for _ in range(N):
        b.append((0.005, 0.995))
        b.append((0.005, 0.995))
    for _ in range(N):
        b.append((0.005, 0.45))
    return b

def _objective(x, penalty=1000.0):
    """Minimize -sum(r) + penalty * (boundary_viol + overlap_viol)."""
    pos = x[:2*N].reshape(N, 2)
    r = x[2*N:]

    # Boundary violations: each circle must satisfy x>=r, x<=1-r, y>=r, y<=1-r
    bv = (np.maximum(0.0, r - pos[:, 0]) +
          np.maximum(0.0, pos[:, 0] - (1.0 - r)) +
          np.maximum(0.0, r - pos[:, 1]) +
          np.maximum(0.0, pos[:, 1] - (1.0 - r)))
    boundary_pen = np.sum(bv * bv)

    # Overlap violations: dist < r_i + r_j
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = np.maximum(0.0, min_dist - dist)
    overlap_pen = np.sum(viol[mask] * viol[mask])

    return -np.sum(r) + penalty * (boundary_pen + overlap_pen)

def _objective_smooth(x, penalty=1000.0, beta=40.0):
    """Smooth objective with softplus penalties."""
    pos = x[:2*N].reshape(N, 2)
    r = np.maximum(x[2*N:], 1e-6)

    # Smooth boundary
    d = np.stack([pos[:, 0] - r, 1.0 - pos[:, 0] - r,
                  pos[:, 1] - r, 1.0 - pos[:, 1] - r], axis=1)
    # soft-min of d (i.e., -soft-max of -d)
    neg_d = -d
    m = np.max(neg_d, axis=1, keepdims=True)
    neg_d_clipped = np.clip(neg_d, -40.0, 40.0)
    soft_max_neg = m + np.log(np.sum(np.exp(neg_d_clipped - m), axis=1, keepdims=True))
    soft_min_d = -soft_max_neg[:, 0]
    boundary_pen = np.sum(np.log1p(np.exp(beta * (r - soft_min_d))) / beta)

    # Smooth overlap
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    min_dist = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    viol = min_dist - dist
    overlap_pen = np.sum(np.log1p(np.exp(beta * viol[mask])) / beta)

    return -np.sum(r) + penalty * (boundary_pen + overlap_pen)

def _is_valid(pos, radii, tol=1e-7):
    """Check strict validity."""
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

def _post_process(pos, radii, iters=300):
    """Shrink radii to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(iters):
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
                sr = radii[i] + radii[j]
                if d < sr:
                    k = d / (sr + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.11
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        y = row * dy + r
        for col in range(6):
            if k >= N:
                break
            x = r + offset + col * dx
            if 0 <= x - r and x + r <= 1.0 and 0 <= y - r and y + r <= 1.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        mr = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            mr = min(mr, d - radii[j])
        if mr > 0.01:
            pos[k] = [x, y]
            radii[k] = mr * 0.95
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_corner():
    """Corner-anchored initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        pos[i] = [cx, cy]
        radii[i] = 0.15
    rng = np.random.RandomState(7)
    for i in range(4, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(3000):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            mr = min(x, 1 - x, y, 1 - y)
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                mr = min(mr, d - radii[j])
            if mr > best_r:
                best_r = mr
                best_x, best_y = x, y
        pos[i] = [best_x, best_y]
        radii[i] = max(best_r, 0.01) * 0.95
    return pos, radii

def _init_best_fit():
    """Greedy best-fit with center first."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.25
    rng = np.random.RandomState(123)
    for i in range(1, N):
        best_x, best_y, best_r = 0.5, 0.5, 0.0
        for _ in range(5000):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            mr = min(x, 1 - x, y, 1 - y)
            for j in range(i):
                d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
                mr = min(mr, d - radii[j])
            if mr > best_r and mr > 0.003:
                best_r = mr
                best_x, best_y = x, y
        pos[i] = [best_x, best_y]
        radii[i] = max(best_r, 0.003) * 0.97
    return pos, radii

def _init_random(seed):
    """Random initialization with post-processing."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.04, 0.12, N)
    return _post_process(pos, radii)

def _slsqp_polish(pos, radii, deadline):
    """Multi-stage SLSQP polishing with increasing penalty."""
    x = np.concatenate([pos.flatten(), radii])
    bnds = _bounds()
    best_x = x.copy()
    best_val = _objective(x, 1000.0)

    for p_w in [100.0, 500.0, 2000.0, 8000.0]:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(_objective, x, method='SLSQP', bounds=bnds,
                           args=(p_w,),
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        x = best_x

    return best_x[:2*N].reshape(N, 2), np.maximum(best_x[2*N:], 1e-4)

def _lbfgs_polish(pos, radii, deadline):
    """L-BFGS-B polishing."""
    x = np.concatenate([pos.flatten(), radii])
    bnds = _bounds()
    for p_w in [500.0, 3000.0]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(_objective, x, method='L-BFGS-B', bounds=bnds,
                           args=(p_w,),
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < _objective(x, p_w):
                x = res.x
        except Exception:
            pass
    return x[:2*N].reshape(N, 2), np.maximum(x[2*N:], 1e-4)

def _simulated_annealing(pos, radii, deadline, seed=0):
    """Vectorized simulated annealing."""
    rng = np.random.RandomState(seed)
    pos = pos.copy().astype(float)
    radii = radii.copy().astype(float)
    n = N
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)

    x = np.concatenate([pos.flatten(), radii])
    current_energy = _objective(x, 100.0)
    n_steps = 8000

    for step in range(n_steps):
        if time.time() > deadline - 10:
            break
        frac = step / max(n_steps - 1, 1)
        temp = 0.04 * (0.0005 / 0.04) ** frac

        i = rng.randint(n)
        dx = rng.normal(0, temp * 0.015)
        dy = rng.normal(0, temp * 0.015)
        dr = rng.normal(0, temp * 0.008)

        old_x, old_y, old_r = pos[i, 0], pos[i, 1], radii[i]
        pos[i, 0] = np.clip(pos[i, 0] + dx, 0.01, 0.99)
        pos[i, 1] = np.clip(pos[i, 1] + dy, 0.01, 0.99)
        radii[i] = np.clip(radii[i] + dr, 0.01, 0.4)

        x_new = np.concatenate([pos.flatten(), radii])
        new_energy = _objective(x_new, 100.0)
        delta_e = new_energy - current_energy

        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
            current_energy = new_energy
        else:
            pos[i, 0] = old_x
            pos[i, 1] = old_y
            radii[i] = old_r

        if step % 200 == 0:
            pp, rr = _post_process(pos, radii)
            if _is_valid(pp, rr):
                s = np.sum(rr)
                if s > best_score:
                    best_score = s
                    best_pos = pp.copy()
                    best_radii = rr.copy()

    return best_pos, best_radii

def _differential_evolution(pos_init, radii_init, deadline):
    """Differential evolution for global search."""
    if time.time() > deadline - 30:
        return pos_init, radii_init
    bnds = _bounds()
    try:
        res = differential_evolution(
            _objective, bnds, args=(200.0,),
            seed=42, maxiter=200, popsize=15,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.7,
            polish=True, init='latinhypercube',
            updating='immediate'
        )
        if res.success or res.fun < _objective(np.concatenate([pos_init.flatten(), radii_init]), 200.0):
            x = res.x
            return x[:2*N].reshape(N, 2), np.maximum(x[2*N:], 1e-4)
    except Exception:
        pass
    return pos_init, radii_init

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Generate initializations
    inits = []
    pos_hex, rad_hex = _init_hexagonal()
    inits.append((pos_hex, rad_hex))

    pos_cor, rad_cor = _init_corner()
    inits.append((pos_cor, rad_cor))

    pos_bf, rad_bf = _init_best_fit()
    inits.append((pos_bf, rad_bf))

    pos_r1, rad_r1 = _init_random(100)
    inits.append((pos_r1, rad_r1))

    pos_r2, rad_r2 = _init_random(200)
    inits.append((pos_r2, rad_r2))

    best_pos = pos_bf.copy()
    best_radii = rad_bf.copy()
    best_score = np.sum(best_radii)

    def _update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        pp, rr = _post_process(pos, radii)
        if _is_valid(pp, rr):
            s = np.sum(rr)
            if s > best_score:
                best_score = s
                best_pos = pp.copy()
                best_radii = rr.copy()

    # Phase 1: Differential evolution on the best init (most robust)
    if time.time() < deadline - 40:
        pos_de, rad_de = _differential_evolution(pos_bf, rad_bf, deadline)
        _update_best(pos_de, rad_de)

    # Phase 2: SLSQP + L-BFGS-B polishing from multiple inits
    for idx, (pos_init, rad_init) in enumerate(inits):
        if time.time() > deadline - 15:
            break
        # SLSQP
        if time.time() < deadline - 12:
            pos_s, rad_s = _slsqp_polish(pos_init, rad_init, deadline)
            _update_best(pos_s, rad_s)
        # L-BFGS-B
        if time.time() < deadline - 8:
            pos_l, rad_l = _lbfgs_polish(pos_s, rad_s, deadline)
            _update_best(pos_l, rad_l)

    # Phase 3: Simulated annealing from best
    if time.time() < deadline - 15:
        pos_sa, rad_sa = _simulated_annealing(best_pos, best_radii, deadline, seed=42)
        _update_best(pos_sa, rad_sa)

    # Phase 4: Final polish
    if time.time() < deadline - 8:
        pos_f, rad_f = _slsqp_polish(best_pos, best_radii, deadline)
        _update_best(pos_f, rad_f)
    if time.time() < deadline - 5:
        pos_f, rad_f = _lbfgs_polish(best_pos, best_radii, deadline)
        _update_best(pos_f, rad_f)

    # Final validation
    best_pos, best_radii = _post_process(best_pos, best_radii)
    if not _is_valid(best_pos, best_radii):
        # Fallback: 5x5 grid
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        k = 0
        for i in range(5):
            for j in range(5):
                if k < N:
                    pos[k] = [(i + 0.5) / 5.0, (j + 0.5) / 5.0]
                    radii[k] = 0.095
                    k += 1
        best_pos, best_radii = _post_process(pos, radii)

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
