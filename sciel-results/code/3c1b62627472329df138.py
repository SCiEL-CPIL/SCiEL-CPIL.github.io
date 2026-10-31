"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize

H = np.sqrt(3) / 2
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all C(11,3)=165 triangle index triplets
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_N_TRI = len(_TRI_IDX)

def _all_areas(pts):
    """Vectorized computation of all 165 triangle areas."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum area among all triangles."""
    return float(np.min(_all_areas(pts)))

def _clamp_point(p):
    """Clamp point to triangle using barycentric coordinates."""
    x, y = p
    c = 2.0 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1.0 - b - c
    eps = 1e-9
    a = max(eps, min(1.0 - 2*eps, a))
    b = max(eps, min(1.0 - 2*eps, b))
    c = max(eps, min(1.0 - 2*eps, c))
    s = a + b + c
    a /= s; b /= s; c /= s
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points inside triangle."""
    return np.array([_clamp_point(p) for p in pts])

def _soft_min(pts, beta=100.0):
    """Smooth approximation of min area for gradient-based methods."""
    areas = _all_areas(pts)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -400, 400)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _obj_smooth(flat, beta=100.0):
    """Objective for L-BFGS-B: negative soft-min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_soft_min(pts, beta)

def _obj_hard(flat):
    """Objective for Nelder-Mead: negative hard min area."""
    pts = _clamp_all(flat.reshape(-1, 2))
    return -_min_area(pts)

def _init_structured():
    """3-fold symmetric structured initialization."""
    pts = []
    m = 0.10
    pts.append(m * V0 + (1-m)/2 * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + m * V1 + (1-m)/2 * V2)
    pts.append((1-m)/2 * V0 + (1-m)/2 * V1 + m * V2)
    m2 = 0.12
    pts.append(0.5*(1-m2)*V0 + 0.5*m2*V1 + 0.5*(1-m2)*V2)
    pts.append(m2*V0 + 0.5*(1-m2)*V1 + 0.5*(1-m2)*V2)
    pts.append(0.5*(1-m2)*V0 + 0.5*(1-m2)*V1 + m2*V2)
    pts.append((V0 + V1 + V2) / 3.0)
    cx, cy = 1/3.0, H/3.0
    r = 0.14
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        pts.append([cx + r * np.cos(angle), cy + r * np.sin(angle)])
    return np.array(pts[:N])

def _init_grid():
    """Triangular grid initialization."""
    pts = []
    for i, y in enumerate([0.06, 0.34, 0.60]):
        n_row = 4 - i
        x_start = 0.06 + i * 0.08
        x_span = 0.88 - i * 0.16
        for j in range(n_row):
            x = x_start + j * (x_span / max(n_row - 1, 1))
            pts.append([x, y])
    while len(pts) < N:
        pts.append([0.5, 0.45])
    return np.array(pts[:N])

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        pts.append(r1 * V1 + r2 * V2)
    return np.array(pts)

def _lbfgs_polish(pts, beta, maxiter, deadline):
    """L-BFGS-B polish with soft-min objective."""
    try:
        res = minimize(
            _obj_smooth, pts.flatten(), args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        opt = _clamp_all(res.x.reshape(-1, 2))
        return opt, _min_area(opt)
    except Exception:
        return pts, _min_area(pts)

def _nm_polish(pts, deadline):
    """Nelder-Mead polish with hard objective."""
    try:
        res = minimize(
            _obj_hard, pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        opt = _clamp_all(res.x.reshape(-1, 2))
        return opt, _min_area(opt)
    except Exception:
        return pts, _min_area(pts)

def _coord_descent(pts, deadline, max_rounds=200):
    """Coordinate descent: optimize each point with small steps, adaptive refinement."""
    current = pts.copy()
    score = _min_area(current)
    steps = [0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in steps:
        for _ in range(max_rounds // len(steps)):
            if time.time() > deadline:
                return current, score
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        trial = current.copy()
                        trial[i] = _clamp_point(current[i] + np.array([dx, dy]))
                        s = _min_area(trial)
                        if s > score + 1e-15:
                            current = trial
                            score = s
                            improved = True
            if not improved:
                break
    return current, score

def _differential_evolution_step(pts, pop_size, deadline, max_iter=50):
    """Differential evolution step to escape local optima."""
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    pop = [pts.copy() for _ in range(pop_size)]
    for i in range(1, pop_size):
        pop[i] = _clamp_all(pts + rng.normal(0, 0.05, size=(N, 2)))
    
    best = pts.copy()
    best_score = _min_area(best)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        for i in range(pop_size):
            r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
            mutant = pop[r1] + 0.8 * (pop[r2] - pop[r3])
            mutant = _clamp_all(mutant)
            trial = mutant if rng.random() < 0.9 else pop[i]
            trial_score = _min_area(trial)
            if trial_score > _min_area(pop[i]):
                pop[i] = trial
            if trial_score > best_score:
                best_score = trial_score
                best = trial.copy()
    return best, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    def tleft():
        return deadline - time.time()
    
    # Generate diverse initializations
    inits = [_init_structured(), _init_grid()]
    for s in [1, 7, 13, 29, 42, 100, 200, 300, 500, 700, 900, 1200, 1500, 2000]:
        inits.append(_init_random(s))
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Multi-start L-BFGS with multiple betas
    for init in inits:
        if tleft() < 12:
            break
        for beta in [60.0, 120.0, 250.0]:
            if tleft() < 5:
                break
            pts, sc = _lbfgs_polish(init, beta, 800, deadline)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _init_structured()
        best_score = _min_area(best_pts)
    
    # Phase 2: Nelder-Mead polish on best
    if tleft() > 15:
        pts, sc = _nm_polish(best_pts, deadline)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    # Phase 3: Differential evolution to escape local optima
    if tleft() > 30:
        for _ in range(5):
            if tleft() < 20:
                break
            pts, sc = _differential_evolution_step(best_pts, 15, deadline, max_iter=40)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
            # Polish the result
            if tleft() > 10:
                pts, sc = _nm_polish(best_pts, deadline)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts.copy()
    
    # Phase 4: Perturbation multi-start around best
    rng = np.random.default_rng(99999)
    for trial in range(20):
        if tleft() < 10:
            break
        scale = 0.008 + 0.004 * trial
        pert = best_pts + rng.normal(0, scale, size=(N, 2))
        pert = _clamp_all(pert)
        for beta in [80.0, 160.0]:
            if tleft() < 5:
                break
            pts, sc = _lbfgs_polish(pert, beta, 600, deadline)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        # NM polish on perturbation if time allows
        if tleft() > 8:
            pts, sc = _nm_polish(pert, deadline)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    
    # Phase 5: Final coordinate descent refinement
    if tleft() > 10:
        pts, sc = _coord_descent(best_pts, deadline)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    # Phase 6: One more NM + CD cycle if time remains
    if tleft() > 15:
        pts, sc = _nm_polish(best_pts, deadline)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
        if tleft() > 8:
            pts, sc = _coord_descent(best_pts, deadline)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
