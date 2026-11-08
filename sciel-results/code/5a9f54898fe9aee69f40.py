"""
Optimization of the second autocorrelation inequality constant C2.
Uses L-BFGS-B with analytic gradient for the log-sum-exp smoothed objective.
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

def _smooth_value(h, beta=100.0):
    """
    Smooth surrogate for the value function.
    Uses Log-Sum-Exp for norm_inf to provide differentiability.
    Returns negative value for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1) # g >= 0
    
    # Smooth max: (1/beta) * log(sum(exp(beta * g)))
    # Shift g to prevent overflow
    g_max = np.max(g)
    lse = g_max + (1.0 / beta) * np.log(np.sum(np.exp(beta * (g - g_max))))
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0
        
    val = l2 / (norm_1 * lse)
    return -val

def _smooth_value_grad(h, beta=100.0):
    """
    Compute value and gradient of the smooth surrogate.
    Returns (value, gradient) where value is negative for minimization.
    """
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(g) / (m + 1)
    
    # Smooth max computation
    g_max = np.max(g)
    exp_terms = np.exp(beta * (g - g_max))
    sum_exp = np.sum(exp_terms)
    lse = g_max + (1.0 / beta) * np.log(sum_exp)
    
    if norm_1 <= 0 or lse <= 0:
        return 0.0, np.zeros(N)
        
    val = l2 / (norm_1 * lse)
    
    # Gradient computation
    # dl2/dh: derivative of l2 with respect to h
    # l2 = (W/3) * sum(a^2 + a*b + b^2) where a=y[:m+1], b=y[1:m+2]
    # y[1:-1] = g = convolve(h, h)
    # dy/dh_j = convolve(e_j, h) + convolve(h, e_j) where e_j is unit vector
    
    # Compute dl2/dg first
    # l2 = (W/3) * sum_{i=0}^{m} (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    # dl2/dy_i = (W/3) * (2*y_i + y_{i-1} + y_{i+1}) with boundary handling
    
    dl2_dy = np.zeros(M + 2)
    # For i in 0..m: terms involving y_i
    # y_i appears in: y_i^2 (coefficient 2), y_{i-1}*y_i (coefficient 1), y_i*y_{i+1} (coefficient 1)
    dl2_dy[1:-1] = (W / 3.0) * (2 * y[1:-1] + y[:-2] + y[2:])
    dl2_dy[0] = (W / 3.0) * (2 * y[0] + y[1])
    dl2_dy[-1] = (W / 3.0) * (2 * y[-1] + y[-2])
    
    # d(norm_1)/dh_j = sum of d(norm_1)/dg_k * dg_k/dh_j
    # norm_1 = sum(g)/(m+1), so d(norm_1)/dg_k = 1/(m+1)
    # dg_k/dh_j = h[k-j] + h[j-k] (with appropriate bounds)
    
    # d(lse)/dh_j = sum_k (d(lse)/dg_k) * (dg_k/dh_j)
    # d(lse)/dg_k = exp(beta*(g_k - g_max)) / sum_exp
    
    d_lse_dg = exp_terms / sum_exp
    
    # Now compute full gradient
    # val = l2 / (norm_1 * lse)
    # dval/dh_j = (dl2/dh_j * norm_1 * lse - l2 * (d(norm_1)/dh_j * lse + norm_1 * d(lse)/dh_j)) / (norm_1 * lse)^2
    
    # Compute dl2/dh_j = sum_k (dl2_dy[k] * dy_k/dh_j)
    # dy_k/dh_j = dg_k/dh_j for k in 1..m (since y[1:-1]=g)
    # dg_k/dh_j = h[k-j] if 0<=k-j<N else 0, plus h[j-k] if 0<=j-k<N else 0
    
    # More efficiently: dl2/dh_j = sum_{k} dl2_dy[k] * (h[k-j] + h[j-k])
    # This is a convolution-like operation
    
    # Let's compute using the structure
    # dg_k/dh_j: g is autocorrelation, g[k] = sum_i h[i]*h[i+k]
    # dg_k/dh_j = h[k-j] + h[j-k] (where indices are valid)
    
    # dl2/dh_j = sum_{k=0}^{m} dl2_dy[k] * dg_k/dh_j
    # Note: dl2_dy[0] and dl2_dy[m+1] correspond to y[0]=0 and y[m+1]=0, so they don't contribute
    # Actually y[0]=0 and y[m+1]=0, so dl2_dy[0] and dl2_dy[m+1] only affect through y[1] and y[m]
    
    # Let me recompute more carefully
    # y = [0, g[0], g[1], ..., g[m-1], 0]
    # y has length m+2 = 101
    # a = y[0:m+1] = [0, g[0], ..., g[m-1]] (length m+1=100)
    # b = y[1:m+2] = [g[0], ..., g[m-1], 0] (length m+1=100)
    
    # l2 = (W/3) * sum_{i=0}^{m} (a_i^2 + a_i*b_i + b_i^2)
    # = (W/3) * sum_{i=0}^{m} (y_i^2 + y_i*y_{i+1} + y_{i+1}^2)
    
    # dl2/dy_i for i=0..m+1:
    # y_0: appears in i=0 term as y_0^2 and y_0*y_1: d/dy_0 = 2*y_0 + y_1
    # y_{m+1}: appears in i=m term as y_{m+1}^2 and y_m*y_{m+1}: d/dy_{m+1} = 2*y_{m+1} + y_m
    # y_i for 1<=i<=m: appears in i-1 term (y_i^2, y_{i-1}*y_i) and i term (y_i^2, y_i*y_{i+1})
    #   d/dy_i = 2*y_i + y_{i-1} + y_{i+1}
    
    # This matches what I had above.
    
    # Now dy_i/dh_j:
    # y_0 = 0, y_{m+1} = 0: no contribution
    # y_i = g_{i-1} for 1<=i<=m
    # g_k = sum_{t} h[t] * h[t+k]
    # dg_k/dh_j = h[k-j] (if 0<=k-j<N) + h[j-k] (if 0<=j-k<N)
    
    # dl2/dh_j = sum_{i=1}^{m} (dl2_dy[i] * dg_{i-1}/dh_j)
    # = sum_{i=1}^{m} dl2_dy[i] * (h[i-1-j] + h[j-(i-1)])
    
    # Let c_k = dl2_dy[k+1] for k=0..m-1 (so c_k corresponds to g_k)
    # dl2/dh_j = sum_{k=0}^{m-1} c_k * (h[k-j] + h[j-k])
    
    c = dl2_dy[1:m+1]  # length m = 99
    
    # This is a correlation: dl2/dh_j = sum_k c_k * h[k-j] + sum_k c_k * h[j-k]
    # = corr(c, h)[j] + corr(h, c)[j] but with sign conventions...
    
    # Actually: sum_k c_k * h[k-j] = (c * h)_j where * is cross-correlation
    # And sum_k c_k * h[j-k] = (c * h)_{-j} or similar
    
    # Let's use np.correlate
    # np.correlate(c, h, mode='full') gives sum_k c[k] * h[k+j] for various j
    # We want sum_k c[k] * h[k-j] = sum_k c[k] * h[(k-j)]
    
    # Let me just compute it directly for now (N=50, m=99, this is fast enough)
    dl2_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(m):
            # h[k-j]
            if 0 <= k - j < N:
                s += c[k] * h[k - j]
            # h[j-k]
            if 0 <= j - k < N:
                s += c[k] * h[j - k]
        dl2_dh[j] = s
    
    # d(norm_1)/dh_j = sum_k (1/(m+1)) * dg_k/dh_j = (1/(m+1)) * sum_k (h[k-j] + h[j-k])
    # = (1/(m+1)) * (sum_k h[k-j] + sum_k h[j-k])
    # For fixed j: sum_{k=0}^{m-1} h[k-j] where 0<=k-j<N, plus sum_{k=0}^{m-1} h[j-k] where 0<=j-k<N
    
    dnorm1_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(m):
            if 0 <= k - j < N:
                s += h[k - j]
            if 0 <= j - k < N:
                s += h[j - k]
        dnorm1_dh[j] = s / (m + 1)
    
    # d(lse)/dh_j = sum_k d_lse_dg[k] * dg_k/dh_j
    d_lse_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for k in range(m):
            if 0 <= k - j < N:
                s += d_lse_dg[k] * h[k - j]
            if 0 <= j - k < N:
                s += d_lse_dg[k] * h[j - k]
        d_lse_dh[j] = s
    
    # dval/dh_j = [dl2_dh[j] * norm_1 * lse - l2 * (dnorm1_dh[j] * lse + norm_1 * d_lse_dh[j])] / (norm_1 * lse)^2
    denom = (norm_1 * lse) ** 2
    grad_val = (dl2_dh * norm_1 * lse - l2 * (dnorm1_dh * lse + norm_1 * d_lse_dh)) / denom
    
    # We're minimizing -val, so gradient of objective is -grad_val
    return -val, -grad_val

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
    # Random smooth profiles
    for _ in range(20):
        freqs = RNG.randint(1, 8, size=6)
        phases = RNG.uniform(0, 2 * np.pi, size=6)
        amps = RNG.uniform(0.5, 1.0, size=6)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.1, prof))
        
    return starts

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 20.0) # 20s safety margin

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

    # Optimization Phase
    # We use L-BFGS-B with bounds [0, inf) and analytic gradient
    bounds = [(0, None)] * N
    
    def _obj_grad(x, beta):
        val, grad = _smooth_value_grad(x, beta)
        return val, grad
    
    # We try multiple betas (smoothness parameters) and starts
    betas = [50.0, 100.0, 200.0]
    
    # Sort starts by initial value to prioritize better ones
    start_vals = [(compute_value(np.maximum(0.0, np.asarray(s, dtype=float))), s) for s in starts]
    start_vals.sort(key=lambda x: -x[0])
    
    for _, s in start_vals:
        if time.time() > deadline:
            break
        
        x0 = np.maximum(0.0, np.asarray(s, dtype=float))
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            try:
                res = opt.minimize(
                    lambda x: _obj_grad(x, beta)[0],
                    x0,
                    jac=lambda x: _obj_grad(x, beta)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                track(res.x)
            except Exception:
                pass

    # Polish phase: Perturb best and re-optimize with very high beta
    for _ in range(50):
        if time.time() > deadline:
            break
        pert = best_h + RNG.normal(0.0, 0.05, N)
        pert = np.maximum(0.0, pert)
        
        # Use very high beta for sharpness
        try:
            res = opt.minimize(
                lambda x: _obj_grad(x, 500.0)[0],
                pert,
                jac=lambda x: _obj_grad(x, 500.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            track(res.x)
        except Exception:
            pass
    
    # Final polish on the absolute best
    if time.time() < deadline - 10:
        try:
            res = opt.minimize(
                lambda x: _obj_grad(x, 1000.0)[0],
                best_h,
                jac=lambda x: _obj_grad(x, 1000.0)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 20}
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
