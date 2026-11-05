"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _objective_hard(h):
    """Compute the hard objective value (the value the evaluator uses)."""
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj(h, beta):
    """
    Soft-max surrogate for 2*N*max(conv)/(sum(h))^2.
    Uses log-sum-exp for numerical stability.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum_exp = m + np.log(np.sum(np.exp(shifted))) / beta
    val = 2.0 * N * log_sum_exp / (s * s)
    return val

def _soft_max_grad(h, beta):
    """
    Analytical gradient of the soft-max surrogate objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros(N)
    
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    M = m + np.log(sum_exp) / beta
    
    dM_dh = np.zeros(N)
    for j in range(2 * N - 1):
        if weights[j] < 1e-15:
            continue
        k = j - (N - 1)
        # term1: h[i+k]
        i_start = max(0, -k)
        i_end = min(N, N - k)
        if i_start < i_end:
            dM_dh[i_start:i_end] += weights[j] * h[i_start+k:i_end+k]
        # term2: h[i-k]
        i_start2 = max(0, k)
        i_end2 = min(N, k + N)
        if i_start2 < i_end2:
            dM_dh[i_start2:i_end2] += weights[j] * h[i_start2-k:i_end2-k]
    
    grad = 2.0 * N * (dM_dh * s - 2.0 * M) / (s * s * s)
    return grad

def _chirp_h(theta, scale=1.0, offset=0.0):
    """
    Generate a chirp-like signal: h[k] = scale * cos(theta * k) + offset.
    This creates a signal with bounded autoconvolution peak (Dirichlet kernel-like).
    """
    k = np.arange(N)
    return scale * np.cos(theta * k) + offset

def _optimize_chirp(theta0):
    """
    Optimize theta, scale, offset for a chirp signal to minimize the objective.
    """
    def _chirp_obj(params):
        theta, scale, offset = params
        h = _chirp_h(theta, scale, offset)
        return _objective_hard(h)
    
    def _chirp_grad(params):
        theta, scale, offset = params
        h = _chirp_h(theta, scale, offset)
        # Numerical gradient (small step)
        eps = 1e-6
        grad = np.zeros(3)
        for i in range(3):
            p_plus = params.copy()
            p_plus[i] += eps
            p_minus = params.copy()
            p_minus[i] -= eps
            grad[i] = (_chirp_obj(p_plus) - _chirp_obj(p_minus)) / (2 * eps)
        return grad
    
    try:
        res = minimize(_chirp_obj, [theta0, 1.0, 0.0],
                       jac=_chirp_grad,
                       method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x, _chirp_obj(res.x)
    except Exception:
        return [theta0, 1.0, 0.0], _chirp_obj([theta0, 1.0, 0.0])

def _coordinate_descent(h, max_iter=5000, tol=1e-12, time_limit=None):
    """
    Coordinate descent: optimize one coordinate at a time.
    For a fixed h with h[i] varying, the objective is:
    f(h_i) = 2*N*max(conv(h)) / (S + delta_i)^2
    where S = sum(h) - h[i], delta_i = h[i].
    
    The max of conv is a piecewise-linear function of h_i (since conv is bilinear).
    We can find the optimal h_i by checking the breakpoints.
    """
    start_time = time.time()
    h = h.copy()
    s = np.sum(h)
    
    for iteration in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        if abs(s) < 1e-10:
            break
        
        improved = False
        for i in range(N):
            # Save old value
            old_val = h[i]
            s_old = s
            
            # Compute conv with h[i] = 0 to get the "base" conv
            # conv_full = conv_base + h[i] * (h[i] at lag 0 + contributions from other lags)
            # Actually, let's just do a 1D optimization over h[i].
            
            # The conv entries that depend on h[i]:
            # conv[j] for j where the window includes index i.
            # conv[j] = sum_k h[k]*h[k+j-N+1]
            # h[i] appears when i = k or i = k+j-N+1
            # So conv[j] = base_j + h[i]*(h[i+j-N+1] if valid) + h[i]*(h[i-j+N-1] if valid) + h[i]^2 if j=N-1
            
            # Let's compute the derivative of the max-conv w.r.t. h[i]
            # and do a line search.
            
            # Simple approach: try a few candidate values for h[i]
            # and pick the best one.
            
            # Compute current conv
            conv = np.convolve(h, h, mode='full')
            max_conv = np.max(conv)
            current_obj = abs(2.0 * N * max_conv / (s_old * s_old))
            
            # Try perturbing h[i]
            # The optimal h[i] for a fixed max_conv entry j* would satisfy:
            # d/dh_i [max_conv / (S + h[i])^2] = 0
            # => d(max_conv)/dh_i * (S + h[i])^2 - max_conv * 2*(S + h[i]) = 0
            # => d(max_conv)/dh_i * (S + h[i]) = 2 * max_conv
            # => h[i] = 2*max_conv / (d(max_conv)/dh_i) - S
            
            # But d(max_conv)/dh_i depends on which entry is max.
            # Let's just do a coarse grid search around the current value.
            
            # Estimate the range of h[i] that could be useful
            h_range = abs(old_val) * 0.5 + 1.0
            candidates = np.linspace(old_val - h_range, old_val + h_range, 20)
            
            best_candidate = old_val
            best_candidate_obj = current_obj
            
            for cand in candidates:
                h[i] = cand
                s_new = s_old - old_val + cand
                if abs(s_new) < 1e-10:
                    h[i] = old_val
                    continue
                conv_new = np.convolve(h, h, mode='full')
                obj_new = abs(2.0 * N * np.max(conv_new) / (s_new * s_new))
                if obj_new < best_candidate_obj:
                    best_candidate_obj = obj_new
                    best_candidate = cand
            
            if best_candidate != old_val:
                h[i] = best_candidate
                s = s_old - old_val + best_candidate
                improved = True
            else:
                h[i] = old_val
        
        if not improved:
            break
    
    return h, s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _objective_hard(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Chirp optimization - try various initial thetas
    # The chirp h[k] = cos(theta*k) has autoconvolution peak ~ N/(1 - cos(theta)) for small theta
    # We want to minimize 2*N*max(conv)/S^2
    # For h = cos(theta*k), S = sum(cos(theta*k)) = sin(N*theta/2)*cos((N-1)*theta/2)/sin(theta/2)
    # max(conv) is at lag 0: sum(cos^2(theta*k)) = N/2 + 1/2*sum(cos(2*theta*k))
    
    # Let's try optimizing theta directly
    theta_candidates = np.linspace(0.01, 6.0, 50)
    for theta0 in theta_candidates:
        if time.time() > deadline - 30.0:
            break
        params, val = _optimize_chirp(theta0)
        if val < best_val:
            best_val = val
            best_h = _chirp_h(params[0], params[1], params[2])
    
    # Phase 2: Generate diverse initializations for full optimization
    initializations = []
    
    # 1. Chirp-based initializations
    for theta in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]:
        h0 = _chirp_h(theta, 1.0, 0.0)
        if abs(np.sum(h0)) < 1e-3:
            h0 += 0.1
        initializations.append(h0)
    
    # 2. Random normal with positive mean
    for _ in range(8):
        h0 = rng.standard_normal(N)
        h0 += 0.5
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 3. Sign sequences with DC offset
    for _ in range(8):
        h0 = rng.choice([-1, 1], size=N).astype(float)
        h0 += 0.3 * rng.standard_normal()
        if abs(np.sum(h0)) < 1e-3:
            h0[0] += 1.0
        initializations.append(h0)
        
    # 4. Sine waves
    for freq in [1, 2, 3, 5, 7, 10]:
        t = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h0 = np.sin(t + rng.random()*2*np.pi)
        h0 += 0.8
        initializations.append(h0)

    # Phase 3: L-BFGS-B optimization with soft-max
    betas = [50.0, 100.0, 200.0]
    
    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        h_curr = h0.copy()
        
        for beta in betas:
            if time.time() > deadline - 20.0:
                break
            
            def obj_func(x):
                return _soft_max_obj(x, beta)
            
            def grad_func(x):
                return _soft_max_grad(x, beta)
            
            try:
                res = minimize(obj_func, h_curr, 
                              jac=grad_func,
                              method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40})
                h_new = res.x
            except Exception:
                h_new = h_curr
            
            val_new = _objective_hard(h_new)
            
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
                h_curr = h_new.copy()
            else:
                val_curr = _objective_hard(h_curr)
                if val_new < val_curr:
                    h_curr = h_new.copy()

    # Phase 4: Coordinate descent polish
    if time.time() < deadline - 10.0:
        h_cd, _ = _coordinate_descent(best_h, max_iter=2000, time_limit=deadline - time.time() - 5.0)
        val_cd = _objective_hard(h_cd)
        if val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()

    # Ensure sum is not zero
    if abs(np.sum(best_h)) < 1e-10:
        best_h = np.ones(N)
        best_val = _objective_hard(best_h)
        
    return best_h
