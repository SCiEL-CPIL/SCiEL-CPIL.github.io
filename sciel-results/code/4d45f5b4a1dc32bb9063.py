"""Epigraph QP formulation with Chebyshev parameterization for autocorrelation optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _eval_value(h):
    """Compute the exact evaluation metric."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _autoconvolve_fft(h):
    """Compute autoconvolution using FFT for speed."""
    n_conv = 2 * N - 1
    n_fft = 1
    while n_fft < n_conv:
        n_fft *= 2
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:n_conv]
    return conv

def _chebyshev_basis(n_terms, x):
    """
    Compute Chebyshev polynomials T_0(x), T_1(x), ..., T_{n_terms-1}(x) evaluated at x.
    x should be in [-1, 1]. Returns shape (n_terms, len(x)).
    """
    T = np.zeros((n_terms, len(x)))
    T[0, :] = 1.0
    if n_terms > 1:
        T[1, :] = x
        for k in range(2, n_terms):
            T[k, :] = 2.0 * x * T[k-1, :] - T[k-2, :]
    return T

def _cheb_to_h(coeffs):
    """Convert Chebyshev coefficients to step heights h of length N."""
    # Map indices 0..N-1 to x in [-1, 1]
    x = np.linspace(-1.0, 1.0, N)
    T = _chebyshev_basis(len(coeffs), x)
    h = T.T @ coeffs
    return h

def _h_to_cheb(h, n_terms):
    """Convert step heights to Chebyshev coefficients (least squares)."""
    x = np.linspace(-1.0, 1.0, N)
    T = _chebyshev_basis(n_terms, x)
    # Solve T @ c = h in least squares sense
    c, _, _, _ = np.linalg.lstsq(T.T, h, rcond=None)
    return c

def _epigraph_obj(params, n_terms):
    """
    Epigraph formulation: minimize t subject to autoconv(h)[k] <= t for all k, sum(h) = 1.
    params = [c_0, ..., c_{n_terms-1}, t] where h = cheb_to_h(c)
    """
    c = params[:n_terms]
    t = params[n_terms]
    h = _cheb_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    # Normalize so sum(h) = 1
    h_norm = h / s
    conv = _autoconvolve_fft(h_norm)
    max_conv = np.max(conv)
    # Violation: max(0, max_conv - t) + penalty for not having max_conv <= t
    # Use smooth approximation: soft_max of (conv - t)
    # Or just: t + penalty * max(0, max_conv - t)^2
    # Actually for SLSQP, we use constraints directly
    # Objective: just t (we want to minimize t)
    return t

def _epigraph_constraints(params, n_terms):
    """
    Constraints for SLSQP:
    1. sum(h) = 1
    2. autoconv(h)[k] - t <= 0 for all k (799 constraints)
    """
    c = params[:n_terms]
    t = params[n_terms]
    h = _cheb_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return [0.0]  # invalid
    h_norm = h / s
    conv = _autoconvolve_fft(h_norm)
    # Constraint: conv[k] - t <= 0 for all k
    # This is 799 constraints. To keep it manageable, we can use the max
    # But SLSQP needs individual constraints. Let's use a reduced set.
    # Actually, let's just return the max violation as a single constraint
    max_viol = np.max(conv) - t
    return [s - 1.0, max_viol]

def _epigraph_grad(params, n_terms):
    """Gradient of the epigraph objective (just t, so gradient is zeros except last element)."""
    grad = np.zeros(len(params))
    grad[n_terms] = 1.0  # d(t)/dt = 1
    return grad

def _epigraph_jac(params, n_terms):
    """Jacobian of constraints."""
    c = params[:n_terms]
    t = params[n_terms]
    h = _cheb_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.zeros((2, len(params)))
    
    h_norm = h / s
    conv = _autoconvolve_fft(h_norm)
    
    # Constraint 1: sum(h) - 1 = 0
    # d(sum(h))/d(c_j) = sum_i T_j(x_i)
    x = np.linspace(-1.0, 1.0, N)
    T = _chebyshev_basis(n_terms, x)
    ds_dc = np.sum(T, axis=1)  # shape (n_terms,)
    
    # Constraint 2: max(conv) - t <= 0
    # This is nonsmooth. Use subgradient or approximate.
    # For SLSQP, we can use the index of the max
    k_max = np.argmax(conv)
    
    # d(conv[k_max])/d(h_j) = 2 * h_norm[k_max - j] if 0 <= k_max - j < N else 0
    # More carefully: conv[k] = sum_i h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j] * (1 if k-j == j else 0)
    # Wait: conv[k] = sum_i h[i]*h[k-i]. 
    # d/dh[j]: term where i=j gives h[k-j], term where k-i=j gives i=k-j, gives h[k-j]
    # So d(conv[k])/d(h[j]) = h[k-j] + h[k-j] = 2*h[k-j] if 0<=k-j<N, else just one term
    # Actually if 0<=k-j<N and 0<=j<N (always true), then both terms exist if k-j != j
    # If k-j == j, i.e., k=2j, then it's h[j]^2, derivative is 2*h[j]
    # So in all cases: d(conv[k])/d(h[j]) = 2*h[k-j] if 0<=k-j<N, else 0
    # But wait, h is indexed 0..N-1, and k ranges 0..2N-2
    # For a given j in 0..N-1, k-j is in range [k-N+1, k]
    # We need 0 <= k-j < N, i.e., j <= k and k-j < N, i.e., j > k-N
    
    d_conv_dh = np.zeros(N)
    for j in range(N):
        idx = k_max - j
        if 0 <= idx < N:
            d_conv_dh[j] = 2.0 * h_norm[idx]
    
    # Now d(conv[k_max])/d(c_j) = sum_i d(conv[k_max])/d(h_i) * d(h_i)/d(c_j)
    # d(h_i)/d(c_j) = T_j(x_i)
    d_conv_dc = T @ d_conv_dh  # shape (n_terms,)
    
    # d(sum(h))/d(c_j) = sum_i T_j(x_i) = ds_dc[j]
    # But we have h_norm = h/s, so we need to account for normalization
    # Let's just use the unnormalized version and add the constraint sum(h)=1
    
    jac = np.zeros((2, len(params)))
    # Constraint 1: sum(h) - 1
    jac[0, :n_terms] = ds_dc
    # Constraint 2: max(conv) - t
    jac[1, :n_terms] = d_conv_dc
    jac[1, n_terms] = -1.0  # d(max_conv - t)/dt = -1
    
    return jac

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses epigraph formulation with Chebyshev parameterization and SLSQP.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0

    # Try different numbers of Chebyshev terms
    n_terms_list = [10, 15, 20, 25, 30, 40]
    
    for n_terms in n_terms_list:
        if time.time() > deadline:
            break
        
        # Initial guess: start with a sign-changing pattern
        # Use a sine-like initial condition in Chebyshev space
        x = np.linspace(-1.0, 1.0, N)
        h_init = np.sin(np.pi * x)  # sign-changing, zero mean approximately
        h_init += 0.1 * np.random.RandomState(42 + n_terms).normal(0, 1, N)
        
        # Ensure sum is not zero
        s_init = np.sum(h_init)
        if abs(s_init) < 1e-10:
            h_init += 0.1
            s_init = np.sum(h_init)
        
        c_init = _h_to_cheb(h_init, n_terms)
        
        # Initial t: estimate from current max conv
        h_norm_init = h_init / s_init
        conv_init = _autoconvolve_fft(h_norm_init)
        t_init = np.max(conv_init) + 0.1
        
        params_init = np.concatenate([c_init, [t_init]])
        
        # Bounds: t >= 0, coefficients bounded
        bounds = [(-10.0, 10.0)] * n_terms + [(0.0, 100.0)]
        
        try:
            result = minimize(
                _epigraph_obj,
                params_init,
                method='SLSQP',
                jac=_epigraph_grad,
                constraints=[
                    {'type': 'eq', 'fun': lambda p: _epigraph_constraints(p, n_terms)[0]},
                    {'type': 'ineq', 'fun': lambda p: -_epigraph_constraints(p, n_terms)[1]},
                ],
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success and result.fun < 1e10:
                c_opt = result.x[:n_terms]
                t_opt = result.x[n_terms]
                h_opt = _cheb_to_h(c_opt)
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                    h_opt = h_opt / s_opt  # normalize to sum=1
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Direct L-BFGS-B on h with soft-max surrogate for polishing
    if time.time() < deadline - 10:
        beta = 50.0
        h_work = best_h.copy()
        
        def soft_max_obj(h):
            s = np.sum(h)
            if abs(s) < 1e-15:
                return 1e10
            h_norm = h / s
            conv = _autoconvolve_fft(h_norm)
            m = np.max(conv)
            if m > 1e15:
                return 1e10
            logsum = m + np.log(np.sum(np.exp(beta * (conv - m))))
            soft_max_val = logsum / beta
            val = 2.0 * N * soft_max_val
            return val * val
        
        def soft_max_grad(h):
            s = np.sum(h)
            if abs(s) < 1e-15:
                return np.ones_like(h)
            h_norm = h / s
            conv = _autoconvolve_fft(h_norm)
            m = np.max(conv)
            if m > 1e15:
                return np.ones_like(h)
            
            exp_conv = np.exp(beta * (conv - m))
            sum_exp = np.sum(exp_conv)
            if sum_exp < 1e-300:
                return np.ones_like(h)
            
            weights = exp_conv / sum_exp
            
            # Gradient of soft_max w.r.t. h_norm
            grad_hnorm = np.zeros(N)
            for j in range(N):
                max_l = min(N, 2 * N - 1 - j)
                if max_l > 0:
                    grad_hnorm[j] = (2.0 / beta) * np.dot(h_norm[:max_l], weights[j:j + max_l])
            
            # Chain rule: h_norm = h/s
            # d(soft_max)/d(h_j) = grad_hnorm[j]/s - (sum_i grad_hnorm[i]*h[i]/s^2)
            # = grad_hnorm[j]/s - dot(grad_hnorm, h)/s^2
            val = 2.0 * N * (m + np.log(sum_exp)) / beta
            d_val_d_h = 2.0 * N * (grad_hnorm / s - np.dot(grad_hnorm, h) / (s * s))
            grad = 2.0 * val * d_val_d_h
            
            if not np.all(np.isfinite(grad)):
                return np.ones_like(h)
            return grad
        
        try:
            result = minimize(
                soft_max_obj,
                h_work.copy(),
                jac=soft_max_grad,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < 1e10:
                h_opt = result.x
                s_opt = np.sum(h_opt)
                if abs(s_opt) > 1e-10 and np.all(np.isfinite(h_opt)):
                    h_opt = h_opt / s_opt
                    v_opt = _eval_value(h_opt)
                    if v_opt < best_val:
                        best_val = v_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
