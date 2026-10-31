import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 100
TARGET = N / 2.0
DEADLINE = time.time() + 132.0   # explicit wall-clock guard (kill at 150s)

# ---------------------------------------------------------------------------
# Core linear structure
# ---------------------------------------------------------------------------
def _overlaps(h):
    """overlap[k] = sum_i h[i] * (1 - h[(i-k)%N])  for all circular shifts k."""
    R = np.fft.ifft(np.abs(np.fft.fft(h)) ** 2).real  # R[k]=sum_i h[i]h[(i-k)%N]
    return TARGET - R

def _grad_for_weights(h, w):
    """grad_j = sum_k w[k] * (1 - h[(j-k)%N] - h[(j+k)%N])  (d overlap_k / d h_j)."""
    H = np.fft.fft(h)
    W = np.fft.fft(w)
    conv1 = np.fft.ifft(W * H).real                 # sum_k w[k] h[(j-k)%N]
    wn = np.zeros(N); wn[0] = w[0]; wn[1:] = w[:0:-1]
    conv2 = np.fft.ifft(np.fft.fft(wn) * H).real    # sum_k w[k] h[(j+k)%N]
    return np.sum(w) - conv1 - conv2

def _proj(h):
    """Project onto {h in [0,1]^N : sum h = TARGET} (bisection on the level)."""
    h = np.asarray(h, dtype=float)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = np.sum(np.clip(h - mid, 0.0, 1.0))
        if s > TARGET:
            lo = mid
        else:
            hi = mid
    return np.clip(h - 0.5 * (lo + hi), 0.0, 1.0)

def _active_weights(ov, tol=0.05):
    m = np.max(ov)
    d = m - ov
    w = np.exp(-d / tol)
    return w / np.sum(w)

# ---------------------------------------------------------------------------
# Initializations (diverse, all projected to valid)
# ---------------------------------------------------------------------------
def _zadoff(root):
    n = np.arange(N)
    x = np.cos(-np.pi * root * n * (n + 1) / N)
    x = (x - x.mean()) / (x.max() - x.min() + 1e-12)
    return _proj(0.5 + 0.5 * x)

def _sinusoid(freq, amp, phase):
    n = np.arange(N)
    return _proj(0.5 + amp * np.cos(2 * np.pi * freq * n / N + phase))

def _two_block(frac, lo, hi):
    h = np.zeros(N)
    k = int(N * frac)
    h[:k] = lo; h[k:] = hi
    return _proj(h)

def _inits():
    L = [np.full(N, 0.5)]
    for r in range(1, 31):
        L.append(_zadoff(r))
    for f in range(1, 16):
        for ph in [0.0, np.pi / 2, np.pi]:
            L.append(_sinusoid(f, 0.5, ph))
    for fr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for lo, hi in [(0.0, 1.0), (0.2, 0.8), (0.1, 0.9), (0.15, 0.85)]:
            L.append(_two_block(fr, lo, hi))
    rng = np.random.default_rng(42)
    for _ in range(25):
        L.append(_proj(rng.random(N)))
    return L

# ---------------------------------------------------------------------------
# Active-set LP solver (exact convex min-max)
# ---------------------------------------------------------------------------
def _solve_lp(h, tol=0.05, max_iter=8):
    ov = _overlaps(h)
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        m = np.max(ov)
        idx = np.where(m - ov <= 1e-5 + tol)[0]
        if len(idx) == 0:
            idx = np.where(m - ov <= 1e-4)[0]
        if len(idx) == 0:
            idx = np.array([int(np.argmax(ov))])
        A = np.vstack([_grad_for_weights(h, np.zeros(N)) * 0.0 + 0 for _ in range(0)])  # placeholder
        # Build A_eq: rows = overlap_k for active k (coefficient of h_j = d overlap_k/d h_j)
        A_eq = np.vstack([_grad_for_weights(h, np.zeros(N)) for _ in idx])  # zero-weight trick won't work
        # Recompute rows properly:
        A_eq = np.zeros((len(idx), N))
        for r, k in enumerate(idx):
            e = np.zeros(N); e[k] = 1.0
            A_eq[r] = _grad_for_weights(h, e)
        b_eq = np.full(len(idx), m)
        b_eq[-1] = TARGET
        A_eq[-1] = 1.0
        res = linprog(np.zeros(N), A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, 1.0)] * N,
                      method='highs')
        if not res.success:
            break
        h_new = _proj(res.x)
        if np.max(np.abs(h_new - h)) < 1e-9:
            h = h_new
            break
        h = h_new
        ov = _overlaps(h)
    return h

# ---------------------------------------------------------------------------
# Projected subgradient over near-max shifts
# ---------------------------------------------------------------------------
def _subgrad_polish(h, steps=200, tol=0.05):
    ov = _overlaps(h)
    lr0 = 0.5 / np.linalg.norm(_grad_for_weights(h, _active_weights(ov, tol)))
    for s in range(steps):
        if time.time() > DEADLINE:
            break
        ov = _overlaps(h)
        w = _active_weights(ov, tol)
        g = _grad_for_weights(h, w)
        g = g - np.mean(g)  # keep sum(h) = TARGET
        lr = lr0 / (1.0 + 0.02 * s)
        h = _proj(h - lr * g)
    return h

# ---------------------------------------------------------------------------
# Smooth-max polish (keep as final local refinement)
# ---------------------------------------------------------------------------
def _smooth_max(h, beta):
    ov = _overlaps(h)
    m = np.max(ov)
    sh = np.clip(ov - m, -50.0, 50.0)
    e = np.exp(beta * sh)
    lse = m + np.log(np.sum(e)) / beta
    w = e / np.sum(e)
    return lse, _grad_for_weights(h, w)

def _smooth_polish(h, betas=(40.0, 20.0, 10.0)):
    for beta in betas:
        if time.time() > DEADLINE:
            break
        def obj(x):
            return _smooth_max(x, beta)[0]
        def jac(x):
            return _smooth_max(x, beta)[1]
        try:
            res = minimize(obj, h, jac=jac, method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * N,
                           options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False})
            h = _proj(res.x)
        except Exception:
            pass
    return h

# ---------------------------------------------------------------------------
def construct():
    best_h = None
    best_val = np.inf

    for h in _inits():
        v = np.max(_overlaps(h))
        if v < best_val:
            best_val = v
            best_h = h.copy()
        if time.time() > DEADLINE - 40:
            break

    # Rank, then apply the structural active-set LP + subgradient + smooth polish
    scored = sorted(((np.max(_overlaps(h)), h) for h in _inits()), key=lambda t: t[0])
    seen = set()
    top = []
    for v, h in scored:
        key = tuple(np.round(h, 3))
        if key in seen:
            continue
        seen.add(key)
        top.append(h)
        if len(top) >= 8:
            break

    for h0 in top:
        if time.time() > DEADLINE - 12:
            break
        cur = h0.copy()
        # 1) exact convex active-set LP (the structural core)
        cur = _solve_lp(cur, tol=0.05, max_iter=8)
        # 2) subgradient over near-max shifts
        cur = _subgrad_polish(cur, steps=200, tol=0.05)
        # 3) smooth-max local polish
        cur = _smooth_polish(cur)
        v = np.max(_overlaps(cur))
        if v < best_val:
            best_val = v
            best_h = cur.copy()

    # Final tightening on the global best with tighter tolerance
    if time.time() < DEADLINE - 6 and best_h is not None:
        c = _solve_lp(best_h, tol=0.01, max_iter=10)
        c = _subgrad_polish(c, steps=300, tol=0.01)
        c = _smooth_polish(c)
        if np.max(_overlaps(c)) < best_val:
            best_h = c

    return best_h
