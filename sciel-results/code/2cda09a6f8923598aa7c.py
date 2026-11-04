import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = 115.0  # safety margin

def _autoconv_and_grad(h):
    """Return (auto_lags, grad_auto) where grad_auto[i] = d(auto_j)/d(h_i) for all j.
    auto has 2N-1 lags. We compute auto and its Jacobian w.r.t. h.
    """
    nfft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    auto_full = np.fft.irfft(H * H, n=nfft)
    # Extract lags 0, 1, ..., N-1, -(N-1), ..., -1
    auto = np.concatenate([auto_full[:N], auto_full[nfft - N + 1:]])
    
    # Gradient: auto_j = sum_{i} h[i] * h[i+j] (linear convolution, not circular)
    # d(auto_j)/d(h_k) = h[k+j] + h[k-j] (with h out of range = 0)
    # We need d(sup)/d(h_k) = sum_j w_j * d(auto_j)/d(h_k)
    # where w_j are softmax weights
    return auto

def _compute_ratio_and_grad(h, a):
    """Compute ratio = ||h||_2^2 / ||h*h||_inf (smooth) and its gradient."""
    h = np.maximum(h, 0.0)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0, np.zeros_like(h)
    hu = h / l1
    l2_sq = np.dot(hu, hu)
    auto = _autoconv_and_grad(hu)
    m = auto.max()
    if m < 1e-15:
        return 0.0, np.zeros_like(h)
    w = np.exp(a * (auto - m))
    ws = w.sum()
    w = w / ws
    sup = m + np.log(ws) / a
    
    # d(ratio)/d(hu_k) = (2*hu_k * sup - l2_sq * d(sup)/d(hu_k)) / sup^2
    # d(sup)/d(auto_j) = w_j
    # d(auto_j)/d(hu_k) = hu[k+j] + hu[k-j] (linear conv, out-of-bounds = 0)
    # d(auto_j)/d(h_k) = d(auto_j)/d(hu_k) * d(hu_k)/d(h_k)
    # d(hu_k)/d(h_k) = (l1 - h_k)/l1^2 = (1 - hu_k)/l1
    # So d(ratio)/d(h_k) = d(ratio)/d(hu_k) * (1 - hu_k)/l1
    
    # Compute d(sup)/d(hu_k) = sum_j w_j * (hu[k+j] + hu[k-j])
    d_sup_d_hu = np.zeros(N)
    for j_idx, lag in enumerate(range(-N + 1, N)):
        # auto index: j=0 at index N-1? Let me reindex
        # auto array: index 0 = lag 0, index 1 = lag 1, ..., index N-1 = lag N-1,
        # index N = lag -(N-1), ..., index 2N-2 = lag -1
        pass
    
    # Simpler: use the structure directly
    # auto[0] = lag 0 = sum_k hu[k]^2
    # auto[1] = lag 1 = sum_k hu[k]*hu[k+1]
    # ...
    # auto[N-1] = lag N-1 = hu[0]*hu[N-1]
    # auto[N] = lag -(N-1) = hu[N-1]*hu[0]
    # ...
    # auto[2N-2] = lag -1 = sum_k hu[k+1]*hu[k] = auto[1]
    
    # d(auto_lag_l)/d(hu_k) = hu[k+l] + hu[k-l] (if in range)
    # For positive lag l (index l in auto): d/d(hu_k) = hu[k+l] (if k+l<N) + hu[k-l] (if k-l>=0)
    # Wait, auto_lag_l = sum_k hu[k] * hu[k+l] for lag l >= 0
    # d/d(hu_k) = hu[k+l] (from term k) + hu[k-l] (from term k-l, where (k-l)+l = k)
    # So d(auto_l)/d(hu_k) = hu[k+l] * [k+l<N] + hu[k-l] * [k-l>=0]
    
    d_sup_d_hu = np.zeros(N)
    for l in range(0, N):  # positive lags 0..N-1
        wl = w[l]
        if wl < 1e-15:
            continue
        for k in range(N):
            if k + l < N:
                d_sup_d_hu[k] += wl * hu[k + l]
            if k - l >= 0:
                d_sup_d_hu[k] += wl * hu[k - l]
    for l in range(1, N):  # negative lags, auto index N..2N-2 corresponds to lag -l
        wl = w[N - 1 + l]  # lag -l is at index N-1+l? Let me check: index N = lag -(N-1), so lag -l is at index N-1+l
        if wl < 1e-15:
            continue
        for k in range(N):
            # auto_{-l} = sum_k hu[k] * hu[k-l] (for k-l >= 0) = sum_k hu[k+l] * hu[k]
            # d/d(hu_k) = hu[k-l] (from term where first index is k) + hu[k+l] (from term where second index is k)
            if k - l >= 0:
                d_sup_d_hu[k] += wl * hu[k - l]
            if k + l < N:
                d_sup_d_hu[k] += wl * hu[k + l]
    
    # d(ratio)/d(hu_k) = (2*hu_k * sup - l2_sq * d_sup_d_hu[k]) / sup^2
    d_ratio_d_hu = (2 * hu * sup - l2_sq * d_sup_d_hu) / (sup * sup)
    
    # d(ratio)/d(h_k) = d_ratio_d_hu[k] * (1 - hu[k]) / l1
    d_ratio_d_h = d_ratio_d_hu * (1 - hu) / l1
    
    ratio = l2_sq / sup
    return ratio, d_ratio_d_h

def _neg_ratio(x, a):
    r, _ = _compute_ratio_and_grad(x, a)
    return -r

def _neg_ratio_grad(x, a):
    _, g = _compute_ratio_and_grad(x, a)
    return -g

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    if elapsed > _DEADLINE - 5:
        return [1.0] * N
    
    best_h = np.ones(N) / N
    best_val = 0.0
    
    # Compute seed value
    r_seed, _ = _compute_ratio_and_grad(best_h, 10000)
    best_val = r_seed
    
    bounds = [(0.0, None)] * N
    rng = np.random.default_rng(123)
    
    inits = [
        np.ones(N) / N,
        np.array([np.exp(-((i - N / 2 + 0.5) ** 2) / (2 * 5.0 ** 2)) for i in range(N)]),
        np.array([np.cos(np.pi * i / N) ** 2 for i in range(N)]),
    ]
    for init in inits:
        init = np.maximum(init, 0.0)
        s = init.sum()
        if s > 1e-15:
            inits.append(init / s)
    for _ in range(8):
        h = rng.random(N)
        h = np.maximum(h, 0.01)
        h = h / h.sum()
        inits.append(h)
    
    smooth_params = [100, 300, 1000, 3000]
    
    for sp in smooth_params:
        for init in inits:
            elapsed = time.time() - _START
            if elapsed > _DEADLINE - 10:
                break
            try:
                res = minimize(
                    _neg_ratio,
                    init.copy(),
                    jac=_neg_ratio_grad,
                    args=(sp,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                h_opt = np.maximum(res.x, 0.0)
                s = h_opt.sum()
                if s < 1e-15:
                    continue
                h_opt = h_opt / s
                r, _ = _compute_ratio_and_grad(h_opt, sp)
                if r > best_val:
                    best_val = r
                    best_h = h_opt.copy()
            except Exception:
                continue
        if time.time() - _START > _DEADLINE - 10:
            break
    
    # Polish
    elapsed = time.time() - _START
    if elapsed < _DEADLINE - 15:
        try:
            res = minimize(
                _neg_ratio,
                best_h.copy(),
                jac=_neg_ratio_grad,
                args=(5000,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = np.maximum(res.x, 0.0)
            s = h_opt.sum()
            if s > 1e-15:
                h_opt = h_opt / s
                r, _ = _compute_ratio_and_grad(h_opt, 5000)
                if r > best_val:
                    best_val = r
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final true value
    h_final = np.maximum(best_h, 0.0)
    s = h_final.sum()
    if s < 1e-15:
        return [1.0] * N
    h_final = h_final / s
    auto = _autoconv_and_grad(h_final)
    l2_sq = np.dot(h_final, h_final)
    sup_true = auto.max()
    if sup_true < 1e-15:
        return [1.0] * N
    true_val = l2_sq / sup_true
    
    if true_val <= r_seed + 1e-12:
        return [1.0] * N
    
    return list(h_final)
