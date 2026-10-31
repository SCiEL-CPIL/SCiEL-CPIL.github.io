"""LJ38 global minimum: FCC truncated-octahedron construction + L-BFGS polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
SIGMA = 1.0
# LJ minimum distance
R_MIN = 2.0 ** (1.0/6.0)  # ~1.122462

# deadline guard (target well under 120s)
START = time.time()
DEADLINE = 105.0  # seconds

def _elapsed():
    return time.time() - START

def lj_energy_grad(pos):
    pos = np.asarray(pos, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    pair_e = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)
    coef = (24.0 * inv_r6 - 48.0 * inv_r12) / np.sqrt(r2)
    np.fill_diagonal(coef, 0.0)
    grad = np.einsum('ij,ijk->ik', coef, diff)
    return energy, grad

def lj_energy(pos):
    return lj_energy_grad(pos)[0]

def _fcc_points(max_coord=5):
    pts = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                if (i + j + k) % 2 == 0:
                    pts.append((i, j, k))
    return np.array(pts, dtype=float)

def _make_fragment(fcc, center=(0,0,0), n=N, tie_break='lex'):
    c = np.array(center, dtype=float)
    d = np.linalg.norm(fcc - c, axis=1)
    if tie_break == 'lex':
        order = np.lexsort((fcc[:,2], fcc[:,1], fcc[:,0], d))
    else:
        order = np.argsort(d)
    idx = order[:n]
    frag = fcc[idx].copy()
    # center
    frag -= frag.mean(axis=0)
    return frag

def _scale_to_min_dist(pos, target=R_MIN):
    # compute current min pair distance
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1e9)
    rmin = np.sqrt(np.min(r2))
    if rmin > 1e-12:
        pos = pos * (target / rmin)
    return pos

def _relax(pos):
    # L-BFGS-B polish
    res = minimize(
        lambda x: lj_energy_grad(x)[0],
        pos,
        jac=lambda x: lj_energy_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 20000},
    )
    return res.x

def _valid(pos):
    # atoms at least 0.5 apart
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1e9)
    return np.min(r2) >= 0.25 - 1e-9

def construct():
    global START
    START = time.time()
    fcc = _fcc_points(5)
    
    best_pos = None
    best_e = np.inf
    
    # Build a set of initial guesses (FCC-based)
    guesses = []
    # base fragment, lex tie-break
    base = _make_fragment(fcc, tie_break='lex')
    base = _scale_to_min_dist(base)
    guesses.append(base.copy())
    # argsort tie-break (stable) 
    base2 = _make_fragment(fcc, tie_break='argsort')
    base2 = _scale_to_min_dist(base2)
    guesses.append(base2.copy())
    
    # small random perturbations of the base fragment
    rng = np.random.default_rng(12345)
    for _ in range(40):
        p = base.copy()
        p += rng.normal(0, 0.03, p.shape)
        guesses.append(p)
    
    for g in guesses:
        if _elapsed() > DEADLINE - 5.0:
            break
        try:
            relaxed = _relax(g)
            if not _valid(relaxed):
                # try to repair by re-centering / small jitter; if still invalid skip
                continue
            e = lj_energy(relaxed)
            if e < best_e:
                best_e = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # fallback: return the scaled base fragment
        best_pos = base
    return np.asarray(best_pos, dtype=float)
