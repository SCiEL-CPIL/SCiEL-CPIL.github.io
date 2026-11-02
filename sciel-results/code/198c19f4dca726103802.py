import time
import numpy as np
from scipy.optimize import minimize

N = 50
d = 0.01
_BUDGET = 110.0

# precompute geometry
_a = -0.25 + np.arange(N) * d          # (N,) left edges
_a1 = _a + d                            # (N,) right edges
_m = np.arange(-N, N + 1)               # (2N+1,)
_t = _m * d                              # (2N+1,)
# precompute overlap geometry for all (m,i,j) as functions? ov depends on h only through H, geometry fixed.
# left = max(a_i, t - a1_j); right = min(a1_i, t - a_j); ov = clip(right-left,0,None)
_ai = _a[None, :, None]                 # (1,N,1)
_a1i = _a1[None, :, None]               # (1,N,1)
_tv = _t[:, None, None]                 # (M,1,1)
_a1j = _a1[None, None, :]               # (1,1,N)
_aj = _a[None, None, :]                 # (1,1,N)
_left = np.maximum(_ai, _tv - _a1j)     # (M,N,N)
_right = np.minimum(_a1i, _tv - _aj)    # (M,N,N)
_OV = np.clip(_right - _left, 0.0, None)  # (M,N,N) fixed geometry

def _autoconv(h):
    # h (N,) -> g (M,)
    H = h[:, None] * h[None, :]         # (N,N)
    g = np.einsum('mij,ij->m', _OV, H)
    return g

def _true_score(h):
    h = np.asarray(h, dtype=float)
    h = np.clip(h, 0.0, None)
    if h.sum() <= 0:
        return 0.0
    h = h / h.sum() * N   # normalize to mean 1 (sum N)
    g = _autoconv(h)
    sup = g.max()
    if sup <= 0:
        return 0.0
    l2 = float(np.sum(h * h))
    l1 = float(np.sum(h))
    return l2 / (l1 * sup)

def _neg_logR(x, tau):
    h = np.softplus(x)
    s = h.sum()
    if s <= 0 or not np.isfinite(s):
        return 1e9
    h = h / s * N   # mean 1
    g = _autoconv(h)
    gm = float(g.max())
    if gm <= 0 or not np.isfinite(gm):
        return 1e9
    tau_eff = max(tau * gm, 1e-9)
    # stable soft-max
    sup = gm + tau_eff * float(np.log(np.sum(np.exp((g - gm) / tau_eff))))
    l2 = float(np.sum(h * h))
    l1 = float(np.sum(h))
    R = l2 / (l1 * sup)
    if R <= 0 or not np.isfinite(R):
        return 1e9
    return -float(np.log(R))

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_score = _true_score(best_h)
    def time_ok():
        return (time.time() - t0) < _BUDGET
    # initial guesses (in x-space, softplus inverse approx)
    def to_x(h):
        h = np.clip(np.asarray(h, float), 1e-6, None)
        return np.log(np.expm1(h))   # inverse softplus
    inits = [np.ones(N) * 0.0]  # box (x=0 -> h=softplus(0)=ln2, normalized)
    # central spike
    sp = np.zeros(N); sp[N//2] = 1.0; inits.append(to_x(sp))
    # two spikes
    sp2 = np.zeros(N); sp2[N//4]=1; sp2[3*N//4]=1; inits.append(to_x(sp2))
    # gaussian
    xx = np.linspace(-1,1,N); gg = np.exp(-2*xx*xx); inits.append(to_x(gg))
    # triangle
    tt = np.abs(np.linspace(-1,1,N)); inits.append(to_x(1-tt))
    # random
    for _ in range(4):
        rr = np.abs(np.random.randn(N)) + 0.1; inits.append(to_x(rr))
    tau = 0.2
    xcur = None
    for itau in range(6):
        if not time_ok(): break
        for x0 in inits:
            if not time_ok(): break
            def obj(x):
                return _neg_logR(x, tau)
            try:
                res = minimize(obj, x0, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
                hx = np.softplus(res.x)
                sc = _true_score(hx)
                if sc > best_score:
                    best_score = sc
                    best_h = np.clip(hx, 0.0, None)
                if xcur is None or _neg_logR(res.x, tau) < _neg_logR(xcur, tau):
                    xcur = res.x
            except Exception:
                pass
        tau *= 0.5
    # final refine at very low tau from best x
    if time_ok() and xcur is not None:
        try:
            res = minimize(lambda x: _neg_logR(x, 1e-4), xcur, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9})
            hx = np.softplus(res.x)
            sc = _true_score(hx)
            if sc > best_score:
                best_score = sc
                best_h = np.clip(hx, 0.0, None)
        except Exception:
            pass
    # normalize output to mean 1 (sum N) to match seed scale
    best_h = np.clip(best_h, 0.0, None)
    s = best_h.sum()
    if s > 0:
        best_h = best_h / s * N
    return [float(v) for v in best_h]
