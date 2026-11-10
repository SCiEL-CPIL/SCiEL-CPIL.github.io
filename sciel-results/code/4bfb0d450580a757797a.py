"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: hard wall-clock limit with safety margin.
_DEADLINE = time.time() + 80.0  # 80s, 10s margin under 90s kill

def _E_of_a(a, w0, w1):
    """E(a)=H(a)/H0 for CPL in spatially flat universe. a in (0,1]."""
    a = np.asarray(a, dtype=float)
    # E(a) = a^{-3 w1} (1-a)^{3(1+w0+w1)}
    # Use log for stability
    out = np.empty_like(a)
    mask = a > 0
    out[mask] = np.exp(-3.0*w1*np.log(a[mask]) + 3.0*(1.0+w0+w1)*np.log1p(-a[mask]))
    # at a=0 handle limit
    if np.any(~mask):
        # a->0: if w1>0 -> inf, w1<0 -> 0, w1=0 -> (1)^{...}=1
        if w1 > 0:
            out[~mask] = np.inf
        elif w1 < 0:
            out[~mask] = 0.0
        else:
            out[~mask] = 1.0
    return out

def _reference_integral(z, w0, w1, npts=4000):
    """High-accuracy reference of int_0^z dz'/E(z') using adaptive-like dense Simpson."""
    z = float(z)
    if z <= 0:
        return 0.0
    # integrate in a-space: int_0^1 da/(a^2 E(a)) but with upper a=1/(1+z)
    # Actually int_0^z dz'/E(z') ; change var a=1/(1+z'): z'=1/a-1, dz'=-da/a^2
    # = int_{a=1}^{1/(1+z)} (-da/a^2)/E(a) = int_{1/(1+z)}^{1} da/(a^2 E(a))
    a_lo = 1.0/(1.0+z)
    a_hi = 1.0
    # dense composite Simpson in a
    n = npts
    if n % 2 == 1:
        n += 1
    a = np.linspace(a_lo, a_hi, n)
    E = _E_of_a(a, w0, w1)
    f = 1.0/(a*a*E)
    # Simpson
    h = (a_hi-a_lo)/(n-1)
    s = f[0] + f[-1] + 4.0*np.sum(f[1:-1:2]) + 2.0*np.sum(f[2:-1:2])
    return (h/3.0)*s

def _quad_value(z, nodes, weights, w0, w1):
    """Quadrature: (1+z)*z*sum_i w_i/E(a_i), a_i=1/(1+z*nodes[i])."""
    z = float(z)
    if z <= 0:
        return 0.0
    a = 1.0/(1.0 + z*nodes)
    E = _E_of_a(a, w0, w1)
    return (1.0+z)*z*np.sum(weights/E)

def _relative_errors(nodes, weights, zgrid, w0s, w1s):
    errs = []
    for w0, w1 in zip(w0s, w1s):
        for z in zgrid:
            ref = _reference_integral(z, w0, w1, npts=2000)
            q = _quad_value(z, nodes, weights, w0, w1)
            if ref > 1e-12:
                errs.append(abs(q-ref)/ref)
    return np.array(errs)

def _objective(params):
    # params: [log_w (8), log-spaced node offsets ensuring monotone in (0,1)]
    # nodes via: x_i = sigmoid(cumsum) style; simpler: nodes = 1/(1+exp(-y)) with y sorted
    lw = params[:8]
    y = params[8:16]
    w = np.exp(lw)
    w = w/np.sum(w)  # normalize so sum=1
    # nodes strictly increasing in (0,1): use sigmoid of sorted y
    ys = np.sort(y)
    nodes = 1.0/(1.0+np.exp(-ys))
    nodes = np.clip(nodes, 1e-6, 1.0-1e-6)
    # ensure strict increase
    for i in range(1,8):
        if nodes[i] <= nodes[i-1] + 1e-12:
            nodes[i] = nodes[i-1] + 1e-12
    nodes = np.clip(nodes, 1e-6, 1.0-1e-6)
    errs = _relative_errors(nodes, w, ZGRID, W0S, W1S)
    # soft-max of squared errors (surrogate for worst-case), scaled
    m = np.max(errs)
    # log-sum-exp of (errs^2) for smooth max
    e2 = (errs/1e-2)**2
    em = np.max(e2)
    lse = em + np.log(np.mean(np.exp(e2-em)))
    return lse + 1e-3*np.sum(errs)

# Training cosmologies (CPL w0,w1) and redshift grid
W0S = [-1.0, -0.9, -0.8, -0.7, -0.5, -1.5, -1.2, -0.3, 0.0, -1.0]
W1S = [0.0, 0.3, 0.5, -0.5, 0.0, -0.3, 0.4, 0.0, 0.5, 0.0]
ZGRID = np.array([0.05,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0,3.0,5.0])

def construct():
    # Start from midpoint rule
    n = 8
    nodes0 = (np.arange(n)+0.5)/n
    w0 = np.full(n, 1.0/n)
    params0 = np.concatenate([np.log(w0), np.log(nodes0/(1-nodes0))])
    
    best_params = params0
    best_val = _objective(params0)
    
    # Local refinement with L-BFGS-B
    try:
        res = minimize(_objective, params0, method='L-BFGS-B',
                       options={'maxiter': 200, 'maxfun': 4000,
                                'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        if res.fun < best_val and time.time() < _DEADLINE - 10:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Polish: a few more iterations if time permits
    if time.time() < _DEADLINE - 15:
        try:
            res2 = minimize(_objective, best_params, method='L-BFGS-B',
                            options={'maxiter': 100, 'maxfun': 2000,
                                     'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            if res2.fun < best_val:
                best_params = res2.x
                best_val = res2.fun
        except Exception:
            pass
    
    lw = best_params[:8]
    y = best_params[8:16]
    w = np.exp(lw); w = w/np.sum(w)
    ys = np.sort(y)
    nodes = 1.0/(1.0+np.exp(-ys))
    nodes = np.clip(nodes, 1e-6, 1.0-1e-6)
    for i in range(1,8):
        if nodes[i] <= nodes[i-1]+1e-12:
            nodes[i] = nodes[i-1]+1e-12
    nodes = np.clip(nodes, 1e-6, 1.0-1e-6)
    w = np.clip(w, 1e-15, None); w = w/np.sum(w)
    
    if time.time() > _DEADLINE - 5:
        # safety: return midpoint
        return {"nodes": (np.arange(n)+0.5)/n, "weights": np.full(n,1.0/n)}
    
    return {"nodes": nodes, "weights": w}
# EVOLVE-BLOCK-END
