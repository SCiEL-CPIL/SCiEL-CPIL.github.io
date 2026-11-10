import numpy as np
import time
from scipy.optimize import minimize

_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# Precompute bin indices
_d_vals = (_x_flat - _y_flat + _N - 1).astype(np.int64)
_h0_vals = _x_flat.astype(np.int64)
_h1_vals = (_x_flat + _y_flat).astype(np.int64)
_h2_vals = (_x_flat + 2 * _y_flat).astype(np.int64)

_N_D = _d_vals.max() + 1
_N_H0 = _h0_vals.max() + 1
_N_H1 = _h1_vals.max() + 1
_N_H2 = _h2_vals.max() + 1

# Precompute sparse pooling matrices (100 cells -> bins)
_M_D = np.zeros((_N_D, _N * _N), dtype=np.float64)
_M_H0 = np.zeros((_N_H0, _N * _N), dtype=np.float64)
_M_H1 = np.zeros((_N_H1, _N * _N), dtype=np.float64)
_M_H2 = np.zeros((_N_H2, _N * _N), dtype=np.float64)
for i in range(_N * _N):
    _M_D[_d_vals[i], i] = 1.0
    _M_H0[_h0_vals[i], i] = 1.0
    _M_H1[_h1_vals[i], i] = 1.0
    _M_H2[_h2_vals[i], i] = 1.0


def _entropy_from_probs(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    nz = probs[probs > 1e-15]
    if len(nz) == 0:
        return 0.0
    return -np.sum(nz * np.log2(nz))


def _entropy_grad(probs):
    """Gradient of Shannon entropy w.r.t. probs: dH/dp_i = -log2(p_i) - 1/ln(2) for p_i > 0."""
    grad = np.zeros_like(probs)
    mask = probs > 1e-15
    if np.any(mask):
        grad[mask] = -np.log2(probs[mask]) - 1.0 / np.log(2)
    return grad


def _score(w):
    """Compute D/M ratio from flat weight array w (assumed non-negative, sum 1)."""
    d_probs = _M_D @ w
    h0_probs = _M_H0 @ w
    h1_probs = _M_H1 @ w
    h2_probs = _M_H2 @ w
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    return D / M


def _grad_ratio(params):
    """
    Compute gradient of D/M with respect to log-softmax parameters.
    Returns (value, gradient).
    """
    probs = _log_softmax_to_prob(params)
    n = len(probs)
    
    d_probs = _M_D @ probs
    h0_probs = _M_H0 @ probs
    h1_probs = _M_H1 @ probs
    h2_probs = _M_H2 @ probs
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    hs = np.array([H0, H1, H2])
    M = np.max(hs)
    
    if M <= 1e-9:
        return 0.0, np.zeros(n)
    
    value = D / M
    
    # Gradients of entropies w.r.t. probs
    dD = _M_D.T @ _entropy_grad(d_probs)
    dH0 = _M_H0.T @ _entropy_grad(h0_probs)
    dH1 = _M_H1.T @ _entropy_grad(h1_probs)
    dH2 = _M_H2.T @ _entropy_grad(h2_probs)
    
    # d(D/M)/dprobs = (dD*M - D*dM) / M^2
    # dM/dprobs: M = max(H0,H1,H2), so dM = dH_k where H_k is the max
    argmax_h = np.argmax(hs)
    if argmax_h == 0:
        dM = dH0
    elif argmax_h == 1:
        dM = dH1
    else:
        dM = dH2
    
    d_val_dprobs = (dD * M - D * dM) / (M * M)
    
    # Chain rule: dprobs/dlog_params = probs * (I - probs^T)
    # So d_val/dlog_params = probs * (d_val_dprobs - sum(probs * d_val_dprobs))
    correction = np.dot(probs, d_val_dprobs)
    grad = probs * (d_val_dprobs - correction)
    
    return value, grad


def _normalize(P):
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return None
    return P / s


def _log_softmax_to_prob(params):
    log_p = params
    max_log_p = np.max(log_p)
    log_probs = log_p - max_log_p
    exp_probs = np.exp(log_probs)
    return exp_probs / np.sum(exp_probs)


def _neg_ratio(params):
    probs = _log_softmax_to_prob(params)
    return -_score(probs)


def _neg_ratio_grad(params):
    """Negative ratio with analytic gradient."""
    val, grad = _grad_ratio(params)
    return -val, -grad


def _neg_ratio_smooth(params, temperature=0.1):
    """Smooth objective with soft-max for M."""
    probs = _log_softmax_to_prob(params)
    d_probs = _M_D @ probs
    h0_probs = _M_H0 @ probs
    h1_probs = _M_H1 @ probs
    h2_probs = _M_H2 @ probs
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    # Soft max via log-sum-exp
    hs = np.array([H0, H1, H2])
    scaled = hs / temperature
    max_scaled = np.max(scaled)
    soft_max = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) * temperature
    
    if soft_max <= 1e-9:
        return 0.0
    return -D / soft_max


def _neg_ratio_smooth_grad(params, temperature=0.1):
    """Smooth negative ratio with analytic gradient using soft-max."""
    probs = _log_softmax_to_prob(params)
    n = len(probs)
    
    d_probs = _M_D @ probs
    h0_probs = _M_H0 @ probs
    h1_probs = _M_H1 @ probs
    h2_probs = _M_H2 @ probs
    
    D = _entropy_from_probs(d_probs)
    H0 = _entropy_from_probs(h0_probs)
    H1 = _entropy_from_probs(h1_probs)
    H2 = _entropy_from_probs(h2_probs)
    
    hs = np.array([H0, H1, H2])
    scaled = hs / temperature
    max_scaled = np.max(scaled)
    shifted = np.exp(scaled - max_scaled)
    sum_exp = np.sum(shifted)
    soft_max = (max_scaled + np.log(sum_exp)) * temperature
    
    if soft_max <= 1e-9:
        return 0.0, np.zeros(n)
    
    value = D / soft_max
    
    # Gradients of entropies
    dD = _M_D.T @ _entropy_grad(d_probs)
    dH0 = _M_H0.T @ _entropy_grad(h0_probs)
    dH1 = _M_H1.T @ _entropy_grad(h1_probs)
    dH2 = _M_H2.T @ _entropy_grad(h2_probs)
    
    # d(soft_max)/dprobs: soft_max = T * (max_scaled + log(sum_exp))
    # d log(sum_exp)/dH_k = shifted_k / sum_exp = softmax_k
    # dH_k/dprobs = dH_k (already computed)
    softmax_weights = shifted / sum_exp
    d_soft_max = (softmax_weights[0] * dH0 + softmax_weights[1] * dH1 + softmax_weights[2] * dH2) * temperature
    
    # d(value)/dprobs = (dD * soft_max - D * d_soft_max) / soft_max^2
    d_val_dprobs = (dD * soft_max - D * d_soft_max) / (soft_max * soft_max)
    
    # Chain rule for log-softmax
    correction = np.dot(probs, d_val_dprobs)
    grad = probs * (d_val_dprobs - correction)
    
    return -value, -grad


def _build_family_A(c, alpha, beta):
    """Family A: uniform over x=y+c (diagonals), weighted by exp(-alpha*x^2) * exp(-beta*y^2)."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            P[y, x] = np.exp(-alpha * x * x - beta * y * y)
    return _normalize(P)


def _build_family_B(c, gamma, delta):
    """Family B: weight cells on x=y+c line, Gaussian in position along diagonal."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            mid = (x + y) / 2.0
            P[y, x] = np.exp(-gamma * (mid - 4.5) ** 2 - delta * (x - y) ** 2)
    return _normalize(P)


def _build_family_C(c, alpha, beta, center_x, center_y, sigma):
    """Family C: 2D Gaussian centered at (center_x, center_y) restricted to x-y=c."""
    P = np.zeros((_N, _N))
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            dx = x - center_x
            dy = y - center_y
            P[y, x] = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    return _normalize(P)


def _build_family_D(c, weights):
    """Family D: arbitrary weights on x=y=c line."""
    P = np.zeros((_N, _N))
    valid = []
    for y in range(_N):
        x = y + c
        if 0 <= x < _N:
            valid.append((y, x))
    if len(valid) == 0:
        return None
    w = np.array(weights[:len(valid)])
    w = np.clip(w, 0, None)
    s = w.sum()
    if s <= 1e-15:
        return None
    w = w / s
    for (y, x), val in zip(valid, w):
        P[y, x] = val
    return P


def _refine_scipy(P_init, timeout_remaining):
    """Refine using scipy minimize with analytic gradients and annealing."""
    if P_init is None:
        return None
    P_clipped = np.clip(P_init, 1e-15, None)
    P_norm = P_clipped / P_clipped.sum()
    log_params = np.log(P_norm.flatten())
    
    best_P = P_init.copy()
    best_score = _score(P_init.flatten())
    
    # Annealing schedule with analytic gradient-based optimization
    temps = [1.0, 0.6, 0.4, 0.25, 0.15, 0.1, 0.06, 0.04, 0.025, 0.015, 0.01, 0.005, 0.002]
    start = time.time()
    
    for temp in temps:
        if time.time() - start > timeout_remaining - 2:
            break
        try:
            result = minimize(
                _neg_ratio_smooth_grad,
                log_params,
                args=(temp,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
            )
            opt_params = result.x
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            opt_score = _score(opt_P.flatten())
            
            if opt_score > best_score:
                best_score = opt_score
                best_P = opt_P.copy()
                log_params = np.log(np.clip(opt_P, 1e-15, None).flatten() / np.clip(opt_P, 1e-15, None).sum())
        except Exception:
            pass
    
    # Final exact (non-smooth) polish with tight tolerance
    if time.time() - start < timeout_remaining - 2:
        try:
            result = minimize(
                _neg_ratio_grad,
                log_params,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            opt_params = result.x
            opt_P = _log_softmax_to_prob(opt_params).reshape(_N, _N)
            opt_score = _score(opt_P.flatten())
            
            if opt_score > best_score:
                best_score = opt_score
                best_P = opt_P.copy()
        except Exception:
            pass
    
    return best_P


def _coordinate_descent_polish(P, time_limit=5.0):
    """Systematic coordinate descent: explore each cell with multiple step sizes."""
    if P is None:
        return None
    P = P.copy()
    score = _score(P.flatten())
    start = time.time()
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    while time.time() - start < time_limit:
        improved = False
        for step in step_sizes:
            if time.time() - start > time_limit - 0.1:
                break
            for i in range(_N):
                for j in range(_N):
                    if time.time() - start > time_limit - 0.05:
                        break
                    # Try increasing
                    new_P = P.copy()
                    new_P[i, j] += step
                    new_P = np.clip(new_P, 0, None)
                    s = new_P.sum()
                    if s > 1e-9:
                        new_P = new_P / s
                        new_score = _score(new_P.flatten())
                        if new_score > score:
                            P = new_P
                            score = new_score
                            improved = True
                    # Try decreasing
                    new_P = P.copy()
                    new_P[i, j] -= step
                    new_P = np.clip(new_P, 0, None)
                    s = new_P.sum()
                    if s > 1e-9:
                        new_P = new_P / s
                        new_score = _score(new_P.flatten())
                        if new_score > score:
                            P = new_P
                            score = new_score
                            improved = True
        if not improved:
            break
    
    return P


def _hill_climb(P, time_limit=5.0):
    """Local hill climbing refinement with coordinate descent polish."""
    if P is None:
        return None
    P = P.copy()
    score = _score(P.flatten())
    step_size = 0.2
    start = time.time()
    no_improve = 0
    
    # Phase 1: Random hill climbing
    while time.time() - start < time_limit * 0.6:
        i, j = np.random.randint(0, _N, 2)
        delta = np.random.uniform(-step_size, step_size)
        new_P = P.copy()
        new_P[i, j] += delta
        new_P = np.clip(new_P, 0, None)
        s = new_P.sum()
        if s <= 1e-9:
            continue
        new_P = new_P / s
        new_score = _score(new_P.flatten())
        
        if new_score > score:
            P = new_P
            score = new_score
            step_size *= 0.98
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.95
            if no_improve > 200 or step_size < 0.0005:
                break
    
    # Phase 2: Coordinate descent polish with remaining time
    remaining = time_limit - (time.time() - start)
    if remaining > 1.0:
        P = _coordinate_descent_polish(P, time_limit=remaining)
    
    return P


def arithmetic_kakeya_distribution() -> np.ndarray:
    np.random.seed(12345)
    start_time = time.time()
    timeout = 620  # safety margin
    
    best_P = None
    best_score = 0.0
    
    def _update_best(P):
        nonlocal best_P, best_score
        if P is None:
            return
        score = _score(P.flatten())
        if score > best_score:
            best_score = score
            best_P = P.copy()
    
    # Phase 1: Explore structured families
    # Family A: Diagonals x = y + c
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for alpha in [0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            for beta in [0, 0.05, 0.1, 0.2, 0.5, 1.0]:
                P = _build_family_A(c, alpha, beta)
                if P is not None:
                    _update_best(P)
    
    # Family B: Diagonals with different weighting
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for gamma in [0, 0.1, 0.3, 0.5, 1.0]:
            for delta in [0, 0.1, 0.3, 0.5, 1.0]:
                P = _build_family_B(c, gamma, delta)
                if P is not None:
                    _update_best(P)
    
    # Family C: 2D Gaussians on diagonals
    for c in range(-9, 10):
        if time.time() - start_time > timeout:
            break
        for cx in [2, 4, 5, 6, 8]:
            for cy in [2, 4, 5, 6, 8]:
                for sigma in [1.0, 2.0, 3.0, 5.0]:
                    P = _build_family_C(c, 0, 0, cx, cy, sigma)
                    if P is not None:
                        _update_best(P)
    
    # Phase 2: Refine best candidates with scipy
    if time.time() - start_time < timeout - 60:
        # Refine the overall best
        if best_P is not None:
            refined = _refine_scipy(best_P, timeout - (time.time() - start_time) - 30)
            if refined is not None:
                _update_best(refined)
    
    # Phase 3: Multi-start random initialization + scipy
    if time.time() - start_time < timeout - 30:
        num_starts = 20
        remaining = timeout - (time.time() - start_time) - 10
        per_start = remaining / num_starts
        
        for s in range(num_starts):
            if time.time() - start_time > timeout - 10:
                break
            if per_start < 3:
                break
            
            # Random init: pick a random diagonal and put random weights
            c = np.random.randint(-9, 10)
            valid_cells = []
            for y in range(_N):
                x = y + c
                if 0 <= x < _N:
                    valid_cells.append((y, x))
            
            if len(valid_cells) == 0:
                continue
            
            # Random weights on these cells
            w = np.random.dirichlet(np.ones(len(valid_cells)))
            P = np.zeros((_N, _N))
            for (y, x), val in zip(valid_cells, w):
                P[y, x] = val
            
            # Quick scipy refinement with analytic gradient
            P_clipped = np.clip(P, 1e-15, None)
            P_norm = P_clipped / P_clipped.sum()
            log_params = np.log(P_norm.flatten())
            
            try:
                result = minimize(
                    _neg_ratio_smooth_grad,
                    log_params,
                    args=(0.3,),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                opt_P = _log_softmax_to_prob(result.x).reshape(_N, _N)
                _update_best(opt_P)
            except Exception:
                pass
            
            # Hill climb with coordinate descent
            if time.time() - start_time < timeout - 15:
                hc = _hill_climb(best_P if best_P is not None else P, time_limit=min(4, per_start * 0.4))
                if hc is not None:
                    _update_best(hc)
    
    # Phase 4: Final fine-grained hill climbing with coordinate descent polish
    if time.time() - start_time < timeout - 10 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 8
        if remaining > 5:
            final = _hill_climb(best_P, time_limit=remaining)
            if final is not None:
                _update_best(final)
    
    # Phase 5: Final scipy polish with aggressive annealing
    if time.time() - start_time < timeout - 5 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 2
        if remaining > 3:
            try:
                refined = _refine_scipy(best_P, remaining)
                if refined is not None:
                    _update_best(refined)
            except Exception:
                pass
    
    # Phase 6: Final coordinate descent polish
    if time.time() - start_time < timeout - 2 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 1
        if remaining > 1:
            final = _coordinate_descent_polish(best_P, time_limit=remaining)
            if final is not None:
                _update_best(final)
    
    # Phase 7: Ultra-fine scipy polish with exact gradient (very tight tolerance)
    if time.time() - start_time < timeout - 1 and best_P is not None:
        remaining = timeout - (time.time() - start_time) - 0.5
        if remaining > 1.0:
            try:
                P_clipped = np.clip(best_P, 1e-15, None)
                P_norm = P_clipped / P_clipped.sum()
                log_params = np.log(P_norm.flatten())
                result = minimize(
                    _neg_ratio_grad,
                    log_params,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50}
                )
                opt_P = _log_softmax_to_prob(result.x).reshape(_N, _N)
                _update_best(opt_P)
            except Exception:
                pass
    
    # Ensure valid output
    if best_P is None:
        best_P = np.ones((_N, _N)) / (_N * _N)
    
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N)) / (_N * _N)
    
    return best_P.astype(np.float64)
