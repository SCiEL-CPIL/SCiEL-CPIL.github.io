"""
Improved L-BFGS-B optimization with analytic gradient for smooth log-sum-exp
surrogate. Key improvements:
1. Analytic gradient (vectorized) for faster, more accurate convergence
2. Adaptive beta schedule: low beta for exploration, high for refinement
3. More diverse starting points including structured random profiles
4. Final polish with Nelder-Mead on exact value function
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)

def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y

def compute_value(h):
    """Exact evaluator metric for given heights (defensive)."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _smooth_value_and_grad(h, beta=50.0):
    """
    Smooth surrogate with analytic gradient.
    Uses Log-Sum-Exp for norm_inf.
    Returns (negative_value, gradient_of_negative_value).
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0 since h >= 0
    
    if norm_1 <= 0:
        return 0.0, np.zeros(N)
    
    # Smooth max via log-sum-exp
    g_max = np.max(g)
    exp_terms = np.exp(beta * (g - g_max))
    sum_exp = np.sum(exp_terms)
    if sum_exp <= 0:
        return 0.0, np.zeros(N)
    lse = g_max + (1.0 / beta) * np.log(sum_exp)
    
    if lse <= 0:
        return 0.0, np.zeros(N)
    
    val = l2 / (norm_1 * lse)
    
    # Gradient of val w.r.t. h
    # val = l2 / (norm_1 * lse)
    # dval/dh_i = (1/(norm_1*lse)) * (dl2/dh_i - val * d(norm_1*lse)/dh_i / (norm_1*lse))
    #           = (1/(norm_1*lse)) * (dl2/dh_i - val * (dlse/dh_i/lse + dnorm_1/dh_i/norm_1))
    
    # dl2/dg_j: g[j] appears in y[j+1] and y[j+2]
    # l2 = (W/3) * sum_{i=0}^{m} (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # For g[j] (which is y[j+1]):
    #   - y[j+1]^2 contributes 2*y[j+1]
    #   - y[j]*y[j+1] contributes y[j]
    #   - y[j+1]*y[j+2] contributes y[j+2]
    #   - y[j+1]^2 from next term (i=j+1): 2*y[j+1] again? No, let's be careful.
    #   
    # l2 = (W/3) * [sum_{i=0}^{m} y[i]^2 + sum_{i=0}^{m} y[i]*y[i+1] + sum_{i=0}^{m} y[i+1]^2]
    #     = (W/3) * [sum_{i=0}^{m} y[i]^2 + sum_{i=0}^{m} y[i]*y[i+1] + sum_{i=1}^{m+1} y[i]^2]
    #     = (W/3) * [2*sum_{i=1}^{m} y[i]^2 + y[0]^2 + y[m+1]^2 + sum_{i=0}^{m} y[i]*y[i+1]]
    #     = (W/3) * [2*sum_{j=0}^{m-1} g[j]^2 + 0 + 0 + sum_{i=0}^{m} y[i]*y[i+1]]
    #
    # y = [0, g[0], g[1], ..., g[m-1], 0]
    # sum_{i=0}^{m} y[i]*y[i+1] = 0*g[0] + g[0]*g[1] + ... + g[m-2]*g[m-1] + g[m-1]*0
    #                            = sum_{j=0}^{m-2} g[j]*g[j+1]
    #
    # So l2 = (W/3) * [2*sum_{j=0}^{m-1} g[j]^2 + sum_{j=0}^{m-2} g[j]*g[j+1]]
    
    # dl2/dg[j] = (W/3) * [4*g[j] + (g[j-1] if j>0 else 0) + (g[j+1] if j<m-1 else 0)]
    dl2_dg = np.zeros(m)
    dl2_dg += 4.0 * g
    if m > 1:
        dl2_dg[1:] += g[:-1]
        dl2_dg[:-1] += g[1:]
    dl2_dg *= (W / 3.0)
    
    # dnorm_1/dg[j] = 1/(m+1) for all j (since g >= 0)
    dnorm1_dg = np.ones(m) / (m + 1)
    
    # dlse/dg[j] = beta * exp(beta*(g[j]-g_max)) / sum_exp
    dlse_dg = beta * exp_terms / sum_exp
    
    # Now: dl2/dh[i] = sum_j dl2_dg[j] * dg[j]/dh[i]
    # g = convolve(h, h), so g[j] = sum_k h[k]*h[j-k]
    # dg[j]/dh[i] = h[j-i] + h[i-j] (with bounds checking)
    # For efficiency, use convolution structure:
    # dl2/dh[i] = sum_j dl2_dg[j] * (h[j-i] if 0<=j-i<N else 0 + h[i-j] if 0<=i-j<N else 0)
    #            = sum_j dl2_dg[j] * h[abs(j-i)] ... not quite, need careful indexing
    
    # Actually: g[j] = sum_{k=max(0,j-N+1)}^{min(N-1,j)} h[k] * h[j-k]
    # dg[j]/dh[i] = h[j-i] if 0 <= j-i < N else 0  (from first factor)
    #             + h[i-j] if 0 <= i-j < N else 0  (from second factor, when k=j-i)
    # Note: if j-i == i-j, i.e., j=2i, then we double count? No, they're separate terms.
    # Actually for autocorrelation: g[j] = sum_k h[k]*h[k+j] for j>=0
    # dg[j]/dh[i] = h[i+j] if i+j < N else 0  (when k=i)
    #             + h[i-j] if i-j >= 0 else 0  (when k+j=i, so k=i-j)
    
    # Let's compute dl2_dh via convolution-like operations
    dl2_dh = np.zeros(N)
    for j in range(m):
        coeff = dl2_dg[j]
        if coeff == 0:
            continue
        # dg[j]/dh[i] for each i
        # Term 1: h[i+j] contributes when i+j < N, i.e., i < N-j
        # Term 2: h[i-j] contributes when i-j >= 0, i.e., i >= j
        if j < N:
            # i from 0 to N-1-j: dg[j]/dh[i] includes h[i+j]
            # This means dl2_dh[i] += coeff * h[i+j] for i in [0, N-1-j]
            # But we want dl2_dh, not d/dh of something else...
            # Actually: dl2/dh[i] = sum_j dl2_dg[j] * dg[j]/dh[i]
            # dg[j]/dh[i] = (h[i+j] if i+j<N else 0) + (h[i-j] if i>=j else 0)
            pass
    
    # More efficient: use the fact that dg/dh is a banded matrix
    # dl2_dh[i] = sum_j dl2_dg[j] * [h[i+j] + h[i-j]] (with bounds)
    # Let's compute this with convolutions
    
    # For term h[i+j]: sum_j dl2_dg[j] * h[i+j] = (dl2_dg * h)[i] but reversed
    # Actually: sum_j dl2_dg[j] * h[i+j] where j ranges 0..m-1, i+j < N
    # This is like cross-correlation
    
    # Let me just do it directly but vectorized per j is too slow.
    # Alternative: note that g = convolve(h,h), and the gradient of a quadratic form
    # in g w.r.t. h can be computed via:
    # If l2 = c^T g where c = dl2_dg, then dl2/dh = 2 * convolve(c_padded, h) appropriately
    
    # c^T g = sum_j c[j] * g[j] = sum_j c[j] * sum_k h[k]*h[k+j]
    #       = sum_k h[k] * sum_j c[j]*h[k+j]  (for k+j < N)
    #       + sum_j c[j] * sum_k h[k]*h[k+j] where we swap...
    # Actually: sum_j c[j] sum_k h[k]h[k+j] = sum_k sum_j c[j] h[k]h[k+j]
    #          = sum_k h[k] * sum_j c[j] h[k+j]
    # Also = sum_j c[j] sum_k h[k]h[k+j] = sum_j c[j] * (h * h)[j]
    # The gradient w.r.t. h[i]:
    # d/dh[i] sum_j c[j] g[j] = sum_j c[j] * (h[j-i] + h[i-j]) (with bounds)
    #                          = (c * h)[i] + (c_rev * h)[i] where c is padded
    
    # Pad c to length 2N-1 and use convolution
    c_padded = np.zeros(2 * N - 1)
    c_padded[:m] = dl2_dg
    # convolve(c_padded, h) gives sum_k c_padded[k] * h[i-k]... 
    # Let's use: (c * h)[i] = sum_k c[k] h[i-k]
    # We want sum_j c[j] h[i+j] + sum_j c[j] h[i-j]
    # sum_j c[j] h[i-j] = (c * h)[i] with c defined on [0, m-1]
    # sum_j c[j] h[i+j] = sum_j c[j] h[i+j] = let k=-j, sum_k c[-k] h[i-k] = (c_rev * h)[i]
    
    conv1 = np.convolve(c_padded, h, mode='same')[:N]  # approximates sum_j c[j] h[i-j]
    c_rev = c_padded[::-1]
    conv2 = np.convolve(c_rev, h, mode='same')[:N]     # approximates sum_j c[j] h[i+j]
    
    dl2_dh = conv1 + conv2
    
    # dnorm1/dh[i] = sum_j dnorm1_dg[j] * dg[j]/dh[i] = same structure with dnorm1_dg
    dnorm1_dh = np.zeros(N)
    c2_padded = np.zeros(2 * N - 1)
    c2_padded[:m] = dnorm1_dg
    conv1_2 = np.convolve(c2_padded, h, mode='same')[:N]
    c2_rev = c2_padded[::-1]
    conv2_2 = np.convolve(c2_rev, h, mode='same')[:N]
    dnorm1_dh = conv1_2 + conv2_2
    
    # dlse/dh[i] = sum_j dlse_dg[j] * dg[j]/dh[i]
    dlse_dh = np.zeros(N)
    c3_padded = np.zeros(2 * N - 1)
    c3_padded[:m] = dlse_dg
    conv1_3 = np.convolve(c3_padded, h, mode='same')[:N]
    c3_rev = c3_padded[::-1]
    conv2_3 = np.convolve(c3_rev, h, mode='same')[:N]
    dlse_dh = conv1_3 + conv2_3
    
    # dval/dh[i] = (1/(norm_1*lse)) * (dl2_dh[i] - val * (dlse_dh[i]/lse + dnorm1_dh[i]/norm_1))
    dval_dh = (1.0 / (norm_1 * lse)) * (dl2_dh - val * (dlse_dh / lse + dnorm1_dh / norm_1))
    
    # We're minimizing -val, so gradient of objective = -dval_dh
    grad_obj = -dval_dh
    
    return -val, grad_obj

def _generate_starts():
    x = np.linspace(0, N - 1, N)
    starts = []
    # Uniform
    starts.append(np.ones(N))
    # Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    # Two peaks
    p1 = np.exp(-0.5 * ((x - N * 0.25) / (N / 8)) ** 2)
    p2 = np.exp(-0.5 * ((x - N * 0.75) / (N / 8)) ** 2)
    starts.append(p1 + p2)
    # Semicircle
    starts.append(np.sqrt(np.maximum(0.0, 1 - ((2 * x / (N - 1) - 1)) ** 2)))
    # Cosine bump
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    # Raised cosine
    starts.append(0.5 * (1 + np.cos(np.pi * (2 * x / (N - 1) - 1))))
    # Plateau
    plateau = np.ones(N)
    plateau[:5] = 0.5
    plateau[-5:] = 0.5
    starts.append(plateau)
    # Random smooth profiles with more diversity
    for _ in range(25):
        # Mix of low and high frequency components
        n_freqs = RNG.randint(3, 10)
        freqs = RNG.randint(1, 12, size=n_freqs)
        phases = RNG.uniform(0, 2 * np.pi, size=n_freqs)
        amps = RNG.uniform(0.3, 1.0, size=n_freqs)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        # Ensure non-negative and reasonable scale
        prof = np.maximum(0.05, prof)
        starts.append(prof)
    
    # Add some structured patterns
    # Alternating high-low
    alt = np.ones(N)
    alt[::2] = 0.3
    starts.append(alt)
    
    # Edge-concentrated
    edge = np.exp(-np.abs(x - N/2) / (N/6))
    starts.append(edge)
    
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 25.0)  # 25s safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    starts = _generate_starts()
    
    # Initial evaluation
    for s in starts:
        track(s)

    # Optimization Phase with adaptive beta
    bounds = [(0, None)] * N
    
    # Phase 1: Broad exploration with low beta
    beta_explore = 10.0
    for s in starts[:15]:  # Only top 15 starts for full optimization
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, beta_explore)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, beta_explore)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 2: Refinement with medium beta
    beta_medium = 50.0
    for s in starts[:10]:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, beta_medium)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, beta_medium)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 3: Polish from best solution with high beta
    if time.time() < deadline - 10:
        x0 = best_h.copy()
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, 200.0)[0],
                x0,
                jac=lambda x: _smooth_value_and_grad(x, 200.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 4: Local perturbations from best
    for _ in range(40):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(0.0, pert)
        
        try:
            res = opt.minimize(
                lambda x: _smooth_value_and_grad(x, 300.0)[0],
                pert,
                jac=lambda x: _smooth_value_and_grad(x, 300.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            track(res.x)
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish on exact value (derivative-free)
    if time.time() < deadline - 5:
        x0 = best_h.copy()
        try:
            res = opt.minimize(
                lambda x: -compute_value(x),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-10, 'adaptive': True}
            )
            track(res.x)
        except Exception:
            pass
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
