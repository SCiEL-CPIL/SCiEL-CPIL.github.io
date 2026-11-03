"""Optimized program for Erdos minimum overlap task using Markov chain parameterization."""
N = 100
import numpy as np
import time

def _autocorr(h):
    """Compute all cyclic autocorrelations using FFT."""
    n = len(h)
    F_h = np.fft.rfft(h, n=2*n)
    ac = np.fft.irfft(F_h * np.conj(F_h), n=2*n)[:n]
    return ac

def _overlap_values(h):
    """Compute all overlap values."""
    n = len(h)
    ac = _autocorr(h)
    return n/2.0 - ac

def _score(h):
    """Compute the maximum overlap (lower is better)."""
    return np.max(_overlap_values(h))

def _project(h, target_sum):
    """Project h onto {x in [0,1]^n : sum(x) = target_sum}."""
    h = np.clip(h, 0.0, 1.0)
    for _ in range(200):
        diff = target_sum - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-15
        else:
            mask = h > 1e-15
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
    return h

def _markov_to_h(theta, n):
    """Convert Markov chain parameters to step heights.
    
    theta: (n, 2) array where theta[i, 0] = P(0->0), theta[i, 1] = P(0->1)
           with theta[i, 0] + theta[i, 1] = 1
    We parameterize via logits: logit(theta[i, 0]) = a[i], logit(theta[i, 1]) = b[i]
    But simpler: use a single parameter per node.
    
    Actually, for a 2-state Markov chain on a cycle, the stationary distribution
    is determined by the transition matrix. Let's use a different approach.
    
    We'll parameterize h directly but enforce structure via a low-rank model:
    h[i] = 0.5 + sum_{k=1}^{K} c_k * cos(2*pi*k*i/n) + d_k * sin(2*pi*k*i/n)
    This ensures h is periodic and we can optimize the Fourier coefficients.
    """
    # Use Fourier parameterization: h[i] = 0.5 + sum_{k=1}^{K} (a_k * cos(2*pi*k*i/n) + b_k * sin(2*pi*k*i/n))
    # This automatically gives sum(h) = n/2 if we include only k>=1
    K = min(10, n//2 - 1)
    i = np.arange(n)
    h = np.full(n, 0.5)
    for k in range(1, K+1):
        a_k = theta[2*(k-1)]
        b_k = theta[2*(k-1)+1]
        h += a_k * np.cos(2*np.pi*k*i/n) + b_k * np.sin(2*np.pi*k*i/n)
    return h

def _smooth_max_and_grad(overlaps, beta):
    """Compute smooth max and its gradient."""
    m = np.max(overlaps)
    if m > 50.0/beta:
        idx = np.argmax(overlaps)
        grad = np.zeros(len(overlaps))
        grad[idx] = 1.0
        return m, grad
    exp_vals = np.exp(beta * (overlaps - m))
    weights = exp_vals / np.sum(exp_vals)
    val = m + np.log(np.sum(exp_vals)) / beta
    return val, weights

def _fourier_grad_h(theta, n):
    """Compute h and its Jacobian w.r.t. theta for Fourier parameterization."""
    K = min(10, n//2 - 1)
    i = np.arange(n)
    h = np.full(n, 0.5)
    jac = np.zeros((n, 2*K))
    for k in range(1, K+1):
        cos_k = np.cos(2*np.pi*k*i/n)
        sin_k = np.sin(2*np.pi*k*i/n)
        h += theta[2*(k-1)] * cos_k + theta[2*(k-1)+1] * sin_k
        jac[:, 2*(k-1)] = cos_k
        jac[:, 2*(k-1)+1] = sin_k
    return h, jac

def _optimize_fourier(theta0, n, deadline, target_sum):
    """Optimize Fourier coefficients to minimize max overlap."""
    K = min(10, n//2 - 1)
    theta = theta0.copy()
    best_theta = theta.copy()
    best_val = _score(_project(_markov_to_h(theta, n), target_sum))
    
    betas = [10, 50, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        lr = 0.05 / (1.0 + beta * 0.005)
        max_iters = 300
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            h, jac = _fourier_grad_h(theta, n)
            h_proj = _project(h, target_sum)
            overlaps = _overlap_values(h_proj)
            
            val, w = _smooth_max_and_grad(overlaps, beta)
            
            # Gradient w.r.t. theta: d/dtheta = jac^T * d/dh
            # d/dh overlap(k) = -(h[(i+k)%n] + h[(i-k)%n]) for k!=0, -2*h[i] for k=0
            g_h = np.zeros(n)
            for k in range(n):
                if w[k] < 1e-12:
                    continue
                if k == 0:
                    g_h += w[k] * (-2.0 * h_proj)
                else:
                    g_h += w[k] * (-(h_proj[np.roll(np.arange(n), k)] + h_proj[np.roll(np.arange(n), -k)]))
            
            g_theta = jac.T @ g_h
            
            # Nesterov momentum
            if not hasattr(_optimize_fourier, '_m'):
                _optimize_fourier._m = np.zeros_like(theta)
            
            m_old = _optimize_fourier._m.copy()
            _optimize_fourier._m = 0.9 * _optimize_fourier._m - lr * g_theta
            
            theta_new = theta + _optimize_fourier._m - m_old
            
            h_new = _project(_markov_to_h(theta_new, n), target_sum)
            raw_val = _score(h_new)
            
            if raw_val < best_val - 1e-12:
                best_theta = theta_new.copy()
                best_val = raw_val
            
            theta = theta_new
        
        _optimize_fourier._m = np.zeros_like(theta)
    
    return best_theta, best_val

def _random_start(n, K):
    """Generate random initial Fourier coefficients."""
    return np.random.randn(2*K) * 0.1

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n = N
    target_sum = n / 2.0
    K = min(10, n//2 - 1)
    
    # Multiple restarts with different random initializations
    best_h = np.full(n, 0.5)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    # Try a few random starts
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        theta0 = _random_start(n, K)
        theta_opt, val = _optimize_fourier(theta0, n, min(deadline, start_time + time_per_restart * (r+1)), target_sum)
        
        h_opt = _project(_markov_to_h(theta_opt, n), target_sum)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Final polish: direct coordinate descent on h
    if time.time() < deadline - 10.0:
        h = best_h.copy()
        lr = 0.005
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
            
            overlaps = _overlap_values(h)
            k_max = np.argmax(overlaps)
            
            if k_max == 0:
                g = -2.0 * h
            else:
                g = -(h[np.roll(np.arange(n), k_max)] + h[np.roll(np.arange(n), -k_max)])
            
            h_new = h - lr * g
            h_new = _project(h_new, target_sum)
            
            val = _score(h_new)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h_new.copy()
            h = h_new
    
    # Ensure exact sum
    diff = target_sum - np.sum(best_h)
    if abs(diff) > 1e-15:
        idx = np.argmax(best_h) if diff > 0 else np.argmin(best_h)
        best_h[idx] += diff
    
    return best_h
