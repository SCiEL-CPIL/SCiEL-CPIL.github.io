"""Combined program: analytic soft-max gradient + simulated annealing + random-walk init."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _compute_value(h):
    """Compute the objective value for a given h."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = np.sum(h)
    if abs(s) < 1e-15:
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))

def _soft_max(x, beta=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    shifted = beta * (x - x_max)
    shifted = np.clip(shifted, -500, 500)
    return x_max + np.log(np.sum(np.exp(shifted))) / beta

def _objective_soft(x, beta=100.0):
    """Differentiable objective using soft-max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    sm = _soft_max(conv, beta)
    return 2.0 * N * sm / (s ** 2)

def _grad_soft(x, beta=100.0):
    """Gradient of the soft-max objective (analytic, O(n^2))."""
    h = x
    n = len(h)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    conv = np.convolve(h, h, mode='full')
    x_max = np.max(conv)
    shifted = beta * (conv - x_max)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp

    # Gradient of soft_max w.r.t. h_j:
    # d(conv_k)/d(h_j) = h[k-j] (if 0<=k-j<n) + h[j-k] (if 0<=j-k<n)
    # So grad_h[j] = sum_k weights[k]*(h[k-j] + h[j-k]) with valid indices
    # Term 1: sum_k w[k]*h[k-j], k-j in [0,n-1] => m=k-j, k=m+j
    #   = sum_{m=0}^{n-1} w[m+j]*h[m]
    # Term 2: sum_k w[k]*h[j-k], j-k in [0,n-1] => m=j-k, k=j-m
    #   = sum_{m=0}^{n-1} w[j-m]*h[m]
    grad_h = np.zeros(n)
    significant = np.where(weights > 1e-15)[0]
    for j in range(n):
        # Term 1: m in [max(0,-j), min(n-1, 2n-2-j)]
        m_start = max(0, -j)
        m_end = min(n - 1, 2 * n - 2 - j)
        if m_start <= m_end:
            k_indices = m_start + j
            grad_h[j] += np.dot(weights[k_indices], h[m_start:m_end + 1])
        # Term 2: m in [max(0, j-(2n-2)), min(n-1, j)]
        m_start2 = max(0, j - (2 * n - 2))
        m_end2 = min(n - 1, j)
        if m_start2 <= m_end2:
            k_indices2 = j - np.arange(m_start2, m_end2 + 1)
            grad_h[j] += np.dot(weights[k_indices2], h[m_start2:m_end2 + 1])

    sm = _soft_max(conv, beta)
    factor = 2.0 * N / (s ** 3)
    grad_obj = factor * (s * grad_h - 2.0 * sm * np.ones(n))
    return grad_obj

def _objective_hard(x):
    """Exact objective with hard max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s ** 2))

def _anneal_temp(it, total, t_start=1.0, t_end=0.01):
    """Exponential annealing schedule."""
    return t_start * (t_end / t_start) ** (it / max(total, 1))

def construct():
    """Construct step heights to minimize the autocorrelation inequality constant."""
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    rng = np.random.default_rng(42)

    def try_candidate(h):
        nonlocal best_h, best_val
        h = np.asarray(h, dtype=np.float64)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # ---- Generate diverse initializations ----
    initializations = []

    # 1. Random walk with negative drift (measured improvement: 1.279 -> 1.315)
    for _ in range(6):
        steps = rng.standard_normal(N)
        steps -= np.mean(steps) + 0.05 * rng.standard_normal()  # negative drift
        h0 = np.cumsum(steps)
        h0 -= np.mean(h0)  # center
        h0 += 0.1 * rng.standard_normal(N)
        if abs(np.sum(h0)) < 1e-6:
            h0[0] += 1.0
        initializations.append(h0)

    # 2. Comb patterns (alternating signs, varying period)
    for period in [2, 3, 4, 5, 7]:
        h0 = np.zeros(N)
        for i in range(N):
            h0[i] = 1.0 if (i % period) < period // 2 else -1.0
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)

    # 3. Higher frequency waves
    for freq in [2, 5, 10, 20]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h0 = np.sin(t) + 0.5 * np.cos(2 * t)
        h0 += 0.1 * rng.standard_normal(N)
        initializations.append(h0)

    # 4. Random smooth signals (few harmonics)
    for _ in range(6):
        h = np.ones(N) * 0.5
        for _ in range(3):
            f = rng.uniform(0.5, 4.0)
            a = rng.uniform(0.1, 1.0)
            p = rng.uniform(0, 2 * np.pi)
            h += a * np.sin(2 * np.pi * f * np.arange(N) / N + p)
        initializations.append(h)

    # 5. Block patterns
    for block_size in [10, 20, 50]:
        h0 = np.zeros(N)
        n_blocks = N // block_size
        for b in range(n_blocks):
            h0[b * block_size:(b + 1) * block_size] = rng.standard_normal()
        initializations.append(h0)

    # ---- Optimize each initialization with multi-beta annealing ----
    # For each init: run L-BFGS-B at increasing beta with simulated annealing acceptance
    def run_anneal_opt(h0, max_total_iter=80):
        nonlocal best_h, best_val
        h_curr = np.asarray(h0, dtype=np.float64).copy()
        s = np.sum(h_curr)
        if abs(s) < 1e-10:
            h_curr[0] += 1.0

        # Multi-beta: start low (smooth), end high (sharp)
        betas = [20.0, 50.0, 100.0, 200.0]
        iters_per_beta = max_total_iter // len(betas)

        for beta in betas:
            if time.time() > deadline - 15.0:
                break
            for it in range(iters_per_beta):
                if time.time() > deadline - 15.0:
                    break
                temp = _anneal_temp(it, iters_per_beta, t_start=1.0, t_end=0.01)

                # L-BFGS-B step with analytic gradient
                try:
                    res = minimize(_objective_soft, h_curr,
                                  jac=lambda x: _grad_soft(x, beta),
                                  args=(),
                                  method='L-BFGS-B',
                                  options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 30})
                    h_new = res.x
                except Exception:
                    h_new = h_curr

                # Simulated annealing acceptance
                val_curr = _objective_hard(h_curr)
                val_new = _objective_hard(h_new)

                if val_new < val_curr:
                    h_curr = h_new
                else:
                    delta = val_new - val_curr
                    if delta < 100 * temp and rng.random() < np.exp(-delta / temp):
                        h_curr = h_new

                val_curr = _objective_hard(h_curr)
                if val_curr < best_val:
                    best_val = val_curr
                    best_h = h_curr.copy()

    # Run anneal optimization on each initialization
    for h0 in initializations:
        if time.time() > deadline - 25.0:
            break
        run_anneal_opt(h0, max_total_iter=60)

    # ---- Local refinement with higher beta on best so far ----
    if time.time() < deadline - 20.0:
        for beta in [200.0, 500.0]:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(_objective_soft, best_h,
                              jac=lambda x: _grad_soft(x, beta),
                              method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                hard_val = _objective_hard(res.x)
                if hard_val < best_val:
                    best_val = hard_val
                    best_h = res.x.copy()
            except Exception:
                pass

    # ---- Nelder-Mead polish on best ----
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_objective_hard, best_h, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            hard_val = _objective_hard(res.x)
            if hard_val < best_val:
                best_val = hard_val
                best_h = res.x.copy()
        except Exception:
            pass

    # ---- Quick structured pattern scan (two-block) ----
    if time.time() < deadline - 3.0:
        for k in range(1, N, 20):
            if time.time() > deadline - 2.0:
                break
            for ratio in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                h = np.zeros(N)
                h[:k] = ratio
                h[k:] = -1.0
                try_candidate(h)

    return best_h
