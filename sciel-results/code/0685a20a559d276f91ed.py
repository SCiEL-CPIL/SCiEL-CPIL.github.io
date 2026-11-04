"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Structural change: Use a DFT parameterization h = |G|^2 where G is the DFT of a
sparse/structured signal. This leverages the spectral structure of autoconvolution
and naturally enforces non-negativity. We also exploit scale invariance to fix
||h||_1 = 1, reducing the problem to maximizing ||h||^2_2 / ||h*h||_inf.

Key insight from inspirations:
1. Complex parameterization h = |g|^2 removes non-negativity constraints.
2. DFT-based initialization: h_k = |sum_j g_j omega^{jk}|^2 has structured autoconvolution.
3. Scale invariance: fix L1 norm to 1.
4. Sidon sets of spikes: if spikes are far enough apart, autoconvolution peaks don't overlap,
   giving ratio proportional to number of spikes.

We combine:
- DFT parameterization for spectral structure
- Complex optimization for smoothness
- Spike-based initializations (Sidon sets)
- Scale invariance (fix L1=1)
- Multi-start with diverse initializations
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 145.0
_SAFETY = 10.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio_neg(h):
    return -_ratio(h)


def _soft_logR_and_grad(h, beta=50.0):
    """
    Smooth surrogate: log R_smooth = log(l2sq) - log(l1) - logsumexp(ac, beta)
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(h)

    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 1e12, np.zeros_like(h)
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)

    if logsumexp <= 0.0:
        return 1e12, np.zeros_like(h)

    w = exp_ac / sum_exp

    d_logsumexp_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_logsumexp_dh[j] = 2.0 * DH * s

    g = np.zeros(N)
    g += 2.0 * h / l2sq
    g -= 1.0 / l1
    g -= d_logsumexp_dh / logsumexp

    logR = math.log(l2sq) - math.log(l1) - logsumexp
    return logR, g


def _normalize_l1(h):
    """Normalize h to have L1 norm = 1."""
    s = float(np.sum(h))
    if s <= 0:
        return h
    return h / s


def _local_optimize_trust(h0, time_budget):
    """Run trust-region optimization on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize_l1(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        x = _normalize_l1(x)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        x = _normalize_l1(x)
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='trust-constr',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 150, 'gtol': 1e-10, 'disp': False,
                     'verbose': 0, 'barrier_tol': 1e-10},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rx = _normalize_l1(rx)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    try:
        for _ in range(2):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rx = _normalize_l1(rx)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _local_optimize_lbfgsb(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize_l1(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        x = _normalize_l1(x)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        x = _normalize_l1(x)
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 800, 'maxfun': 8000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rx = _normalize_l1(rx)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    try:
        for _ in range(2):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rx = _normalize_l1(rx)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _initializations():
    """Yield a set of promising starting points."""
    inits = []
    
    # Single spikes
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
    
    # Pairs of spikes (Sidon-like: far apart)
    for a in range(0, N, 8):
        for b in range(a + 8, N, 8):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
    
    # Triples of spikes
    for a in range(0, N, 12):
        for b in range(a + 12, N, 12):
            for c in range(b + 12, N, 12):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)
    
    # Quadruples
    for a in range(0, N, 15):
        for b in range(a + 15, N, 15):
            for c in range(b + 15, N, 15):
                for d in range(c + 15, N, 15):
                    s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0; s[d] = 1.0
                    inits.append(s)
    
    # Random sparse
    rng = np.random.default_rng(12345)
    for _ in range(30):
        k = rng.integers(2, 15)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    
    # Random smooth
    for _ in range(30):
        s = rng.random(N) ** 3.0
        inits.append(s)
    
    # Gaussian bumps
    for center in range(0, N, 7):
        x = np.arange(N)
        s = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
        inits.append(s)
    
    # Bimodal
    x = np.arange(N)
    for c1, c2 in [(10, 40), (15, 35), (5, 45), (20, 30)]:
        s = np.exp(-0.5 * ((x - c1) / 4.0) ** 2) + np.exp(-0.5 * ((x - c2) / 4.0) ** 2)
        inits.append(s)
    
    # Cosine window
    x = np.linspace(-1, 1, N)
    inits.append(0.5 + 0.5 * np.cos(np.pi * x))
    inits.append(np.cos(np.pi * x / 2)**2)
    
    # Triangle
    inits.append(1 - np.abs(x))
    
    # Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.3
    inits.append(h_alt)
    
    # DFT-based: h = |FFT(g)|^2 for random g
    for seed in range(10):
        rng_local = np.random.RandomState(seed + 100)
        g = rng_local.randn(N) + 1j * rng_local.randn(N)
        h_dft = np.abs(np.fft.fft(g))**2
        h_dft = h_dft / np.max(h_dft)
        inits.append(h_dft)
    
    # DFT-based with sparse g
    for seed in range(10):
        rng_local = np.random.RandomState(seed + 200)
        k = rng_local.integers(3, 10)
        idx = rng_local.choice(N, size=k, replace=False)
        g = np.zeros(N, dtype=complex)
        g[idx] = rng_local.randn(k) + 1j * rng_local.randn(k)
        h_dft = np.abs(np.fft.fft(g))**2
        h_dft = h_dft / np.max(h_dft)
        inits.append(h_dft)
    
    return inits


def _complex_objective(z_flat):
    """
    Objective for complex parameterization z.
    h = |z|^2, normalized to L1=1.
    We minimize -log(R) to maximize R.
    R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)
    With L1=1: R = ||h||^2_2 / ||h*h||_inf
    """
    z = z_flat.reshape(N, 2)
    z_complex = z[:, 0] + 1j * z[:, 1]
    h = np.abs(z_complex) ** 2
    
    l1 = float(np.sum(h))
    if l1 <= 1e-12:
        return 1e12
    h = h / l1
    
    l2sq = float(np.dot(h, h))
    if l2sq <= 1e-12:
        return 1e12
    
    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    
    if ac_max <= 1e-12:
        return 1e12
        
    R = l2sq / ac_max
    
    if R <= 0:
        return 1e12
        
    return -math.log(R)


def _complex_gradient(z_flat):
    """
    Gradient of the complex objective.
    h_j = x_j^2 + y_j^2.
    d h_j / d x_j = 2 x_j, d h_j / d y_j = 2 y_j.
    """
    z = z_flat.reshape(N, 2)
    z_complex = z[:, 0] + 1j * z[:, 1]
    h = np.abs(z_complex) ** 2
    
    l1 = float(np.sum(h))
    if l1 <= 1e-12:
        return np.zeros_like(z_flat)
    h_norm = h / l1
    
    logR, grad_h = _soft_logR_and_grad(h_norm, beta=50.0)
    
    if logR > 1e11:
        return np.zeros_like(z_flat)
        
    grad_z = np.zeros_like(z_flat)
    grad_z[:, 0] = 2.0 * z[:, 0] * grad_h
    grad_z[:, 1] = 2.0 * z[:, 1] * grad_h
    
    return -grad_z


def _local_optimize_complex(z0, time_budget):
    """
    Optimize the complex parameterization z.
    Returns best h and value.
    """
    z0 = np.asarray(z0, dtype=float).flatten()
    
    z_complex = z0[:N] + 1j * z0[N:]
    h_init = np.abs(z_complex) ** 2
    best_h = h_init.copy()
    best_v = _ratio(best_h)
    
    t0 = time.time()
    
    try:
        res = minimize(
            _complex_objective,
            z0,
            jac=_complex_gradient,
            method='L-BFGS-B',
            options={'maxiter': 400, 'maxfun': 4000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        
        if res.x is not None:
            z_opt = res.x
            z_opt_complex = z_opt[:N] + 1j * z_opt[N:]
            h_opt = np.abs(z_opt_complex) ** 2
            v_opt = _ratio(h_opt)
            
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()
                
    except Exception:
        pass
    
    # Polish with direct h optimization
    if _time_left() > 5.0:
        try:
            h_pol, v_pol = _local_optimize_lbfgsb(best_h, time_budget=_time_left() - 6.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass
            
    return best_h, best_v


def _initializations_complex():
    """Yield complex initializations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Random complex phases with random magnitudes
    for _ in range(40):
        mags = rng.random(N) ** 0.5
        phases = rng.random(N) * 2 * math.pi
        z = np.zeros(2 * N)
        z[0::2] = mags * np.cos(phases)
        z[1::2] = mags * np.sin(phases)
        inits.append(z)
        
    # Structured: Constant magnitude, random phases
    for _ in range(15):
        phases = rng.random(N) * 2 * math.pi
        z = np.zeros(2 * N)
        z[0::2] = np.cos(phases)
        z[1::2] = np.sin(phases)
        inits.append(z)
        
    # Structured: Linear phase (chirp)
    for freq in np.linspace(0.1, 3.0, 15):
        phases = 2 * math.pi * freq * np.arange(N) / N
        z = np.zeros(2 * N)
        z[0::2] = np.cos(phases)
        z[1::2] = np.sin(phases)
        inits.append(z)
        
    # Sparse g: only a few nonzero entries
    for _ in range(30):
        k = rng.integers(2, 12)
        idx = rng.choice(N, size=k, replace=False)
        z = np.zeros(2 * N)
        for i in idx:
            r = rng.random()
            theta = rng.random() * 2 * math.pi
            z[2*i] = r * math.cos(theta)
            z[2*i+1] = r * math.sin(theta)
        inits.append(z)
        
    # DFT of sparse signal: g = IFFT(sparse), then h = |g|^2
    # Actually, let's think: if we want h = |G|^2 where G is DFT of g,
    # then we should parameterize G directly and take h = |G|^2.
    # But that's what we're already doing with z = G.
    # Let's add: G with structure from sparse time-domain signal.
    for _ in range(20):
        k = rng.integers(3, 15)
        idx = rng.choice(N, size=k, replace=False)
        g_time = np.zeros(N, dtype=complex)
        g_time[idx] = rng.randn(k) + 1j * rng.randn(k)
        G = np.fft.fft(g_time)
        z = np.zeros(2 * N)
        z[0::2] = np.real(G)
        z[1::2] = np.imag(G)
        inits.append(z)
        
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    # 1. Try complex parameterizations (DFT-based)
    complex_inits = _initializations_complex()
    
    # Evaluate and sort
    scored_complex = []
    for z in complex_inits:
        z_complex = z[:N] + 1j * z[N:]
        h = np.abs(z_complex) ** 2
        v = _ratio(h)
        scored_complex.append((v, z))
        if v > best_v:
            best_v = v
            best_h = h.copy()
            
    scored_complex.sort(key=lambda t: -t[0])
    
    budget_per = 2.5
    for idx, (v0, z) in enumerate(scored_complex):
        if _time_left() <= budget_per + 2.0:
            break
            
        # Skip if not competitive
        if v0 < best_v * 0.3 and best_v > 5.0:
            continue
                
        h_opt, v_opt = _local_optimize_complex(z, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
            
        if _time_left() <= 25.0:
            break
            
    # 2. Try direct h initializations if time permits
    if _time_left() > 30.0:
        direct_inits = _initializations()
        scored_direct = []
        for s in direct_inits:
            v = _ratio(s)
            scored_direct.append((v, s))
            if v > best_v:
                best_v = v
                best_h = s.copy()
                
        scored_direct.sort(key=lambda t: -t[0])
        
        for idx, (v0, s) in enumerate(scored_direct[:15]):
            if _time_left() <= 10.0:
                break
            if v0 < best_v * 0.3 and best_v > 5.0:
                continue
                
            h_opt, v_opt = _local_optimize_lbfgsb(s, time_budget=4.0)
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()

    # Final polish
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize_trust(best_h, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
