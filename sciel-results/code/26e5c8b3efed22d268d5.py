import numpy as np
import time
from scipy.optimize import minimize

N = 100

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
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    return p

def _project_sum(h, target_sum=N/2.0):
    """Project h onto {0<=h<=1, sum(h)=target_sum} via bisection on clip offset."""
    h = h.copy()
    lo, hi = -1.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_clipped = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_clipped)
        if s > target_sum:
            hi = mid
        else:
            lo = mid
    return np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)

def _objective_epigraph(h, t, alpha=200.0):
    """Epigraph: minimize t + penalty if t < max_overlap."""
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    # Soft constraint: penalize when overlap > t
    diff = overlap - t
    penalty = (1.0/alpha) * np.log(np.sum(np.exp(alpha * diff)))
    return t + penalty

def _objective_epigraph_grad(h, t, alpha=200.0):
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    diff = overlap - t
    e = np.exp(alpha * diff)
    p = e / np.sum(e)
    
    # grad w.r.t. t
    grad_t = 1.0
    
    # grad w.r.t. h
    h_next = np.zeros(N)
    h_next[:-1] = h[1:]
    h_prev = np.zeros(N)
    h_prev[1:] = h[:-1]
    p_prev = np.zeros(N)
    p_prev[1:] = p[:-1]
    
    grad_h = p * (1.0 - h_next) - p_prev * h
    
    return grad_h, grad_t

def _polish_direct(h_init, max_iter=50000, tol=1e-15):
    """Direct coordinate descent polish on the 100-dim space."""
    h = h_init.copy()
    h = _project_sum(h)
    best_score = _score(h)
    best_h = h.copy()
    
    # Compute current overlap
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    max_ov = np.max(overlap)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try to adjust h[i] to reduce max overlap
            # The overlap at position i involves h[i] and h[i-1]
            # overlap[i] = h[i] * (1 - h[i+1]) if i < N-1
            # overlap[i-1] = h[i-1] * (1 - h[i]) if i > 0
            
            h_old = h[i]
            
            # Binary search for best h[i] in [0,1] keeping sum constraint
            # We'll do a local line search
            best_local = h_old
            best_local_score = best_score
            
            # Sample a few values around current
            for delta in [-0.05, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.05]:
                h_test = h.copy()
                h_test[i] = np.clip(h_old + delta, 0.0, 1.0)
                # Adjust sum: distribute correction
                diff = N/2.0 - np.sum(h_test)
                if abs(diff) > 1e-10:
                    # Distribute to other elements
                    mask = np.ones(N, dtype=bool)
                    mask[i] = False
                    h_test[mask] += diff / (N-1)
                    h_test = np.clip(h_test, 0.0, 1.0)
                    # Re-project
                    h_test = _project_sum(h_test)
                
                sc = _score(h_test)
                if sc < best_local_score:
                    best_local_score = sc
                    best_local = h_test.copy()
            
            if best_local_score < best_score - 1e-16:
                h = best_local
                best_score = best_local_score
                best_h = h.copy()
                improved = True
        
        if not improved:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    x = np.arange(N)

    for K in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 15:
            break

        n_params = 2 * K

        basis = np.zeros((N, 2*K))
        for k in range(1, K+1):
            basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * x / N)
            basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * x / N)

        def get_h(params):
            return 0.5 + basis @ params

        def project(h):
            return _project_sum(h)

        def objective(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)
            return _smooth_max(overlap, alpha=300.0)

        def objective_grad(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)

            p = _smooth_max_grad(overlap, alpha=300.0)

            grad_hc = np.zeros(N)
            h_next = np.zeros(N)
            h_next[:-1] = h_c[1:]
            grad_hc += p * (1.0 - h_next)
            h_prev = np.zeros(N)
            h_prev[1:] = h_c[:-1]
            p_prev = np.zeros(N)
            p_prev[1:] = p[:-1]
            grad_hc -= p_prev * h_c

            mask = (h > 0.01) & (h < 0.99)
            grad_h = grad_hc * mask

            return basis.T @ grad_h

        for init_seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
            if time.time() > deadline - 10:
                break

            np.random.seed(init_seed)
            h0 = np.random.randn(2*K) * 0.02

            try:
                res = minimize(
                    objective,
                    h0,
                    jac=objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
                )

                h_candidate = project(get_h(res.x))
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()

            except Exception:
                pass

    # Final polish phase
    if time.time() < deadline - 5:
        h_polished, sc_polished = _polish_direct(h_best, max_iter=20000, tol=1e-15)
        if sc_polished < score_best:
            score_best = sc_polished
            h_best = h_polished

    return h_best
