import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110.0  # safety margin before 150s kill

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Return (value, probability_weights) for the smooth max."""
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    val = x_max + (1.0/alpha) * np.log(np.sum(e))
    return val, p

def construct():
    # Precompute grid
    x = np.arange(N)
    
    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    # Try different numbers of Fourier modes
    for K in [3, 5, 8, 12]:
        if time.time() > _DEADLINE - 30:
            break
            
        # Basis functions
        # h = 0.5 + sum(c_k cos + s_k sin)
        # We use L-BFGS-B on the coefficients
        n_params = 2 * K
        
        def get_h(params):
            h = np.full(N, 0.5)
            for k in range(1, K + 1):
                c_k = params[2*(k-1)]
                s_k = params[2*(k-1)+1]
                h += c_k * np.cos(2 * np.pi * k * x / N)
                h += s_k * np.sin(2 * np.pi * k * x / N)
            return h

        # Precompute basis vectors for this K
        basis = np.zeros((N, n_params))
        for k in range(1, K + 1):
            basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * x / N)
            basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * x / N)

        def objective(params):
            h = 0.5 + basis @ params
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            penalty = 100.0 * (np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2))
            
            sm_val, _ = _smooth_max(overlap, alpha=200.0)
            return sm_val + penalty

        def objective_grad(params):
            h = 0.5 + basis @ params
            h_clipped = np.clip(h, 0.0, 1.0)
            
            shifted = np.zeros(N)
            shifted[:-1] = h_clipped[1:]
            overlap = h_clipped * (1.0 - shifted)
            
            _, p = _smooth_max_grad(overlap, alpha=200.0)
            
            # Gradient of overlap w.r.t h_clipped
            mask = (h >= 0.0) & (h <= 1.0)
            
            h_next = np.roll(h_clipped, -1)
            h_next[-1] = 0.0
            h_prev = np.roll(h_clipped, 1)
            h_prev[0] = 0.0
            
            grad_h = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
            grad_h *= mask
            
            # Gradient of penalty
            grad_penalty = np.zeros(N)
            grad_penalty[h > 1.0] = 200.0 * (h[h > 1.0] - 1.0)
            grad_penalty[h < 0.0] = -200.0 * h[h < 0.0]
            
            total_grad_h = grad_h + grad_penalty
            
            # Chain rule via precomputed basis
            grad_params = basis.T @ total_grad_h
            
            return grad_params

        # Initial guess: small random coefficients
        np.random.seed(42)
        h0 = np.random.randn(n_params) * 0.1
        
        try:
            res = minimize(
                objective,
                h0,
                jac=objective_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            
            h_candidate = get_h(res.x)
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            # Re-project to sum N/2 iteratively
            for _ in range(20):
                diff = N/2.0 - np.sum(h_candidate)
                h_candidate += diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
            sc = _score(h_candidate)
            if sc < score_best:
                score_best = sc
                h_best = h_candidate.copy()
                
        except Exception:
            pass

    # Final high-precision polish: SLSQP epigraph on raw h-vector
    if time.time() < _DEADLINE - 15:
        try:
            h0 = h_best.copy()
            
            def epigraph_obj(z):
                # z = [h_0, ..., h_{N-1}, t]
                h = z[:N]
                t = z[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                # Smooth max of (overlap - t)
                diff = overlap - t
                d_max = np.max(diff)
                if d_max > 0:
                    sm = d_max + (1.0/500.0) * np.log(np.sum(np.exp(500.0 * (diff - d_max))))
                else:
                    sm = 0.0
                return t + 0.01 * sm

            def epigraph_grad(z):
                h = z[:N]
                t = z[N]
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                diff = overlap - t
                d_max = np.max(diff)
                e = np.exp(500.0 * (diff - d_max))
                p = e / np.sum(e)
                
                # d overlap[i] / d h[i] = 1 - h[i+1]  (h_next)
                # d overlap[i-1] / d h[i] = -h[i-1]    (h_prev)
                h_next = np.roll(h, -1)
                h_next[-1] = 0.0
                h_prev = np.roll(h, 1)
                h_prev[0] = 0.0
                
                grad_h = p * (1.0 - h_next) - np.roll(p, 1) * h_prev
                
                # d/dt = 1 + 0.01 * d(sm)/dt = 1 + 0.01 * (-1) * sum(p) = 1 - 0.01
                grad_t = 1.0 - 0.01 * np.sum(p)
                
                # Scale grad_h by 0.01
                grad_h *= 0.01
                
                return np.concatenate([grad_h, [grad_t]])

            # Constraints: 0 <= h_i <= 1, sum(h) = N/2
            bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]  # t bounded in [0,1]
            
            cons = [
                {'type': 'eq', 'fun': lambda z: np.sum(z[:N]) - N/2.0,
                 'jac': lambda z: np.concatenate([np.ones(N), [0.0]])}
            ]
            
            z0 = np.concatenate([h0, [score_best]])
            
            res = minimize(
                epigraph_obj,
                z0,
                jac=epigraph_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                # Ensure exact sum
                for _ in range(50):
                    diff = N/2.0 - np.sum(h_candidate)
                    if abs(diff) < 1e-12:
                        break
                    h_candidate += diff / N
                    h_candidate = np.clip(h_candidate, 0.0, 1.0)
                
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()
                    
        except Exception:
            pass

    return h_best
