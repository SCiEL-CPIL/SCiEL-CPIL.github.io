"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _overlap_vector(h):
    """Compute the overlap vector h_i * (1 - h_{i+1}) for all i."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)

def _tv_gradient(h, lam):
    """
    Compute the gradient of the total variation term ||h_i - h_{i+1}||_1.
    Using subgradient: sign(h_i - h_{i+1}) for interior, with boundary handling.
    We use a smooth approximation: tanh(h_i - h_{i+1}) / scale.
    """
    n = len(h)
    grad = np.zeros(n)
    # Interior differences
    diff = h[:-1] - h[1:]
    # Smooth sign: tanh(diff / eps)
    eps = 0.01
    sgn = np.tanh(diff / eps)
    # grad[i] gets +sgn[i] from term |h_i - h_{i+1}|
    # grad[i+1] gets -sgn[i] from term |h_i - h_{i+1}|
    grad[:-1] += sgn
    grad[1:] -= sgn
    # Boundary terms: |h_0| and |h_{n-1}|? No, TV is usually sum |h_i - h_{i+1}| for i=0..n-2
    # Sometimes includes boundaries. Let's stick to interior for simplicity.
    return lam * grad

def _data_gradient(h, t, alpha):
    """
    Gradient of the data term: t + penalty for max overlap > t.
    Objective: t + mu * (smooth_max(overlap) - t)^2 if smooth_max > t else t.
    Actually, let's use a simpler epigraph formulation gradient.
    Let M = smooth_max(overlap).
    If t >= M, obj = t, grad_t = 1, grad_h = 0.
    If t < M, obj = t + mu*(M-t)^2.
    grad_t = 1 - 2*mu*(M-t)
    grad_h = -2*mu*(M-t) * grad_M
    """
    n = len(h)
    overlap = _overlap_vector(h)
    M = _smooth_max(overlap, alpha=alpha)
    
    if t >= M:
        return np.zeros(n), 1.0
    
    mu = 1000.0
    factor = -2.0 * mu * (M - t)
    
    # Gradient of smooth_max(overlap) w.r.t h
    # M = max_j + (1/alpha) * log(sum(exp(alpha*(overlap_j - max_j))))
    # dM/dh_i = sum_j (dM/doverlap_j) * (doverlap_j/dh_i)
    
    x_max = np.max(overlap)
    exp_terms = np.exp(alpha * (overlap - x_max))
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # dM/doverlap_j = (1/alpha) * weights[j]
    dM_dov = (1.0/alpha) * weights
    
    # overlap_j = h_j * (1 - h_{j+1})  (with h_n = 0)
    # doverlap_j/dh_j = 1 - h_{j+1}
    # doverlap_j/dh_{j+1} = -h_j
    # doverlap_{j-1}/d_h_j = -h_{j-1} (from term j-1: h_{j-1}(1-h_j))
    
    grad_h = np.zeros(n)
    
    for j in range(n):
        w = dM_dov[j]
        # Contribution to h_j
        grad_h[j] += w * (1.0 - (h[j+1] if j < n-1 else 0.0))
        # Contribution to h_{j+1}
        if j < n - 1:
            grad_h[j+1] += w * (-h[j])
            
    return factor * grad_h, 1.0 - 2.0 * mu * (M - t)

def _project_to_feasible(h, t):
    """Project h to sum N/2 and [0,1], and t to [0,1]."""
    h = _normalize(h)
    t = np.clip(t, 0.0, 1.0)
    return h, t

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Initial guess: 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()
    t_final = current_score + 0.05
    
    # Also try a smooth 3-block pattern as alternative initial guess
    h3 = np.zeros(N)
    third = N // 3
    vals3 = [0.3, 0.7, 0.5]
    for i, v in enumerate(vals3):
        start = i * third
        end = (i + 1) * third if i < 2 else N
        h3[start:end] = v
    h3 = _normalize(h3)
    s3 = _score(h3)
    if s3 < current_score:
        h_final = h3.copy()
        t_final = s3 + 0.05
        current_score = s3

    # Also try cosine wave from inspiration 1
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _normalize(h_cos)
    s_cos = _score(h_cos)
    if s_cos < current_score:
        h_final = h_cos.copy()
        t_final = s_cos + 0.05
        current_score = s_cos

    # Phase 1: Split-Bregman style optimization with TV regularization
    # We minimize: t + TV(h) * lam + penalty(t < smooth_max(overlap))
    # Using projected gradient descent on (h, t)
    
    def _run_sb_optimization(h_init, t_init, alpha, lam, max_iter, step_size):
        h = h_init.copy()
        t = t_init
        b = np.zeros(N)  # Bregman variable for TV
        
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            
            # 1. Update h: minimize data term + ||h - (b + grad_tv)/tau||^2
            # Gradient of data term
            grad_data_h, grad_data_t = _data_gradient(h, t, alpha)
            
            # Gradient of TV term (using subgradient from previous bregman)
            # In split Bregman, we usually solve:
            # h^{k+1} = argmin (1/2)||h - (b^k + grad_tv^k)/tau||^2 + data(h)
            # For simplicity, we do gradient step:
            # h -= step * (grad_data_h + lam * grad_tv)
            
            # Compute TV subgradient
            diff = h[:-1] - h[1:]
            sgn = np.sign(diff)
            # Smooth it
            sgn = np.tanh(diff / 0.05)
            grad_tv = np.zeros(N)
            grad_tv[:-1] += sgn
            grad_tv[1:] -= sgn
            
            # Update h
            h_new = h - step_size * (grad_data_h + lam * grad_tv)
            h_new, t_new = _project_to_feasible(h_new, t)
            
            # Update t
            t_new = t - step_size * grad_data_t
            t_new = np.clip(t_new, 0.0, 1.0)
            
            # Update Bregman variable b
            b = b + (h_new - h)  # Simplified Bregman update
            
            h = h_new
            t = t_new
            
            # Check convergence
            if it % 100 == 0:
                s = _score(h)
                if s < current_score:
                    current_score = s
                    h_final = h.copy()
                    t_final = t
                    
        return h, t

    # Run SB optimization
    try:
        if time.time() < deadline - 30.0:
            h_opt, t_opt = _run_sb_optimization(h_final, t_final, alpha=100.0, lam=0.1, max_iter=500, step_size=0.01)
            h_opt = _normalize(h_opt)
            new_score = _score(h_opt)
            if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                h_final = h_opt
                current_score = new_score
    except Exception:
        pass

    # Phase 2: SLSQP refinement (from current program)
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

    def _run_smooth_optimization(h_init, t_init, alpha, maxiter, ftol):
        """Run SLSQP on smooth epigraph formulation."""
        def obj(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=alpha)
            penalty = 1000.0 * np.maximum(0.0, smax - t)**2
            return t + penalty
        
        def con_sum(params):
            return np.sum(params[:N]) - N/2.0
        
        def con_epi(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return t - np.max(overlap)

        constraints = [
            {'type': 'eq', 'fun': con_sum},
            {'type': 'ineq', 'fun': con_epi}
        ]

        x0 = np.concatenate([h_init, [t_init]])
        result = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds_ext,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        return result

    try:
        if time.time() < deadline - 20.0:
            result = _run_smooth_optimization(h_final, current_score + 0.01, alpha=300.0, maxiter=500, ftol=1e-13)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Phase 3: High-precision polish with very high alpha
    try:
        if time.time() < deadline - 10.0:
            result = _run_smooth_optimization(h_final, current_score + 0.001, alpha=1000.0, maxiter=1000, ftol=1e-15)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Phase 4: Structured periodic restarts with smooth optimization
    try:
        np.random.seed(42)
        n_restarts = 12
        for trial in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Create diverse structured initial points
            if trial % 4 == 0:
                # Periodic pattern with random period
                period = np.random.randint(3, N//3 + 1)
                base = np.random.uniform(0.2, 0.8, period)
                h_rand = np.tile(base, N // period + 1)[:N]
            elif trial % 4 == 1:
                # Two-block with random split
                split = np.random.randint(N//4, 3*N//4)
                a_val = np.random.uniform(0.2, 0.8)
                b_val = np.random.uniform(0.2, 0.8)
                h_rand = np.zeros(N)
                h_rand[:split] = a_val
                h_rand[split:] = b_val
            elif trial % 4 == 2:
                # Smooth random with low-frequency structure
                freqs = np.random.uniform(0.5, 3.0, 3)
                phases = np.random.uniform(0, 2*np.pi, 3)
                amplitudes = np.random.uniform(0.1, 0.3, 3)
                x = np.arange(N) / N
                h_rand = 0.5
                for f, p, a in zip(freqs, phases, amplitudes):
                    h_rand += a * np.sin(2*np.pi*f*x + p)
                h_rand = np.clip(h_rand, 0.0, 1.0)
            else:
                # Multi-block with smooth transitions
                n_blocks = np.random.randint(3, 7)
                block_sizes = np.random.dirichlet(np.ones(n_blocks)) * N
                block_sizes = np.round(block_sizes).astype(int)
                block_sizes[-1] = N - sum(block_sizes[:-1])
                h_rand = np.zeros(N)
                pos = 0
                for i, bs in enumerate(block_sizes):
                    if bs <= 0:
                        continue
                    val = 0.3 + 0.4 * np.sin(np.pi * i / max(1, n_blocks-1))
                    h_rand[pos:pos+bs] = val
                    pos += bs
                h_rand = np.clip(h_rand, 0.0, 1.0)
            
            h_rand = _normalize(h_rand)
            rand_score = _score(h_rand)
            
            # Only optimize if this starting point is promising
            if rand_score < current_score + 0.05:
                t_init = max(rand_score, 0.01) + 0.02
                result = _run_smooth_optimization(h_rand, t_init, alpha=200.0, maxiter=300, ftol=1e-12)
                if result.fun < 100.0:
                    h_opt = _normalize(result.x[:N])
                    new_score = _score(h_opt)
                    if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                        h_final = h_opt
                        current_score = new_score
    except Exception:
        pass

    # Final: one more polish on the best found
    try:
        if time.time() < deadline - 2.0:
            result = _run_smooth_optimization(h_final, current_score + 0.0001, alpha=500.0, maxiter=500, ftol=1e-14)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Final local refinement: adjust only high-overlap positions
    try:
        if time.time() < deadline - 1.0:
            h_refine = h_final.copy()
            shifted = np.zeros(N)
            shifted[:-1] = h_refine[1:]
            overlap = h_refine * (1.0 - shifted)
            # Find positions with overlap above 80% of max
            threshold = 0.8 * np.max(overlap)
            hot_indices = np.where(overlap > threshold)[0]
            
            # Try small perturbations on hot spots
            for _iter in range(50):
                if time.time() > deadline - 0.5:
                    break
                improved = False
                for idx in hot_indices:
                    if idx < N - 1:
                        # Try adjusting h[idx] and h[idx+1]
                        for delta in [-0.02, -0.01, 0.01, 0.02]:
                            h_test = h_refine.copy()
                            h_test[idx] = np.clip(h_test[idx] + delta, 0, 1)
                            h_test[idx+1] = np.clip(h_test[idx+1] - delta, 0, 1)
                            # Renormalize
                            h_test = _normalize(h_test)
                            s_test = _score(h_test)
                            if s_test < current_score - 1e-10:
                                h_refine = h_test
                                current_score = s_test
                                improved = True
                                break
                        if improved:
                            break
                if not improved:
                    break
            
            if current_score < _score(h_final):
                h_final = h_refine
    except Exception:
        pass

    return h_final
