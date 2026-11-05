"""QAOA parameter design with exact state-vector simulation and L-BFGS-B optimization with analytical gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


def _precompute_cut_values(n, edges):
    """Precompute cut value for every state mask. Returns array of shape (2^n,)."""
    dim = 1 << n
    cut_vals = np.zeros(dim, dtype=np.float64)
    for mask in range(dim):
        val = 0
        for u, v in edges:
            if ((mask >> u) ^ (mask >> v)) & 1:
                val += 1
        cut_vals[mask] = val
    return cut_vals


def _precompute_mixer_indices(n):
    """Precompute index pairs for X-gate application per qubit."""
    dim = 1 << n
    pairs_per_qubit = []
    for k in range(n):
        bit = 1 << k
        pairs = []
        for i in range(dim):
            if (i & bit) == 0:
                j = i | bit
                pairs.append((i, j))
        pairs_per_qubit.append(np.array(pairs, dtype=np.int64))
    return pairs_per_qubit


def _apply_mixer_fast(psi, n, beta, pairs_per_qubit):
    """Apply exp(-i*beta*sum(X_i)) using precomputed index pairs."""
    cb = np.cos(beta)
    sb = -1j * np.sin(beta)
    for k in range(n):
        pairs = pairs_per_qubit[k]
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        psi_i = psi[idx_i]
        psi_j = psi[idx_j]
        psi[idx_i] = cb * psi_i + sb * psi_j
        psi[idx_j] = sb * psi_i + cb * psi_j
    return psi


def _qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit):
    """Compute expected cut size via state vector simulation with fast mixer."""
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for k in range(len(gammas)):
        # Cost layer
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        # Mixer layer
        psi = _apply_mixer_fast(psi, n, betas[k], pairs_per_qubit)
    
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, cut_vals))


def _qaoa_expectation_and_grad(n, gammas, betas, cut_vals, pairs_per_qubit):
    """Compute expected cut size and analytical gradient using parameter-shift rule.
    
    For QAOA with p=2, we use the parameter-shift rule for exact gradients:
    d/dγ_k <C> = 0.5 * (<C>(γ_k + π/2) - <C>(γ_k - π/2))
    d/dβ_k <C> = 0.5 * (<C>(β_k + π/2) - <C>(β_k - π/2))
    """
    dim = 1 << n
    p = len(gammas)
    exp_val = _qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit)
    
    grad = np.zeros(2 * p)
    
    # Gradient with respect to gammas using parameter-shift rule
    for k in range(p):
        gp = gammas.copy()
        gm = gammas.copy()
        gp[k] += np.pi / 2
        gm[k] -= np.pi / 2
        fp = _qaoa_expectation(n, gp, betas, cut_vals, pairs_per_qubit)
        fm = _qaoa_expectation(n, gm, betas, cut_vals, pairs_per_qubit)
        grad[k] = 0.5 * (fp - fm)
    
    # Gradient with respect to betas using parameter-shift rule
    for k in range(p):
        bp = betas.copy()
        bm = betas.copy()
        bp[k] += np.pi / 2
        bm[k] -= np.pi / 2
        fp = _qaoa_expectation(n, gammas, bp, cut_vals, pairs_per_qubit)
        fm = _qaoa_expectation(n, gammas, bm, cut_vals, pairs_per_qubit)
        grad[p + k] = 0.5 * (fp - fm)
    
    return exp_val, grad


def _qaoa_gradient(n, params, cut_vals, pairs_per_qubit):
    """Compute gradient using parameter-shift rule for exact gradients.
    
    params = [gamma_0, beta_0, gamma_1, beta_1, ...]
    """
    p = len(params) // 2
    gammas = np.array([params[2*k] for k in range(p)])
    betas = np.array([params[2*k+1] for k in range(p)])
    
    _, grad_gb = _qaoa_expectation_and_grad(n, gammas, betas, cut_vals, pairs_per_qubit)
    
    # Reorder gradient to match params layout [g0, b0, g1, b1, ...]
    grad = np.zeros(2 * p)
    for k in range(p):
        grad[2*k] = grad_gb[k]       # d/d gamma_k
        grad[2*k+1] = grad_gb[p+k]   # d/d beta_k
    
    return grad


def _optimize_single(n, init_params, cut_vals, pairs_per_qubit, max_iter=500):
    """Run one optimization from a given initial point using L-BFGS-B with exact gradients."""
    
    def objective(params):
        try:
            p = len(params) // 2
            gammas = np.array([params[2*k] for k in range(p)])
            betas = np.array([params[2*k+1] for k in range(p)])
            return -_qaoa_expectation(n, gammas, betas, cut_vals, pairs_per_qubit)
        except Exception:
            return 1e9
    
    def objective_grad(params):
        try:
            p = len(params) // 2
            gammas = np.array([params[2*k] for k in range(p)])
            betas = np.array([params[2*k+1] for k in range(p)])
            _, grad_gb = _qaoa_expectation_and_grad(n, gammas, betas, cut_vals, pairs_per_qubit)
            grad = np.zeros(2 * p)
            for k in range(p):
                grad[2*k] = -grad_gb[k]       # negative for minimization
                grad[2*k+1] = -grad_gb[p+k]
            return objective(params), grad
        except Exception:
            return 1e9, np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective_grad, init_params, method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, init_params


def _polish(n, params, cut_vals, pairs_per_qubit, max_iter=1000):
    """High-precision polish using L-BFGS-B with very tight tolerances."""
    def objective_grad(params):
        try:
            p = len(params) // 2
            gammas = np.array([params[2*k] for k in range(p)])
            betas = np.array([params[2*k+1] for k in range(p)])
            val, grad_gb = _qaoa_expectation_and_grad(n, gammas, betas, cut_vals, pairs_per_qubit)
            grad = np.zeros(2 * p)
            for k in range(p):
                grad[2*k] = -grad_gb[k]
                grad[2*k+1] = -grad_gb[p+k]
            return -val, grad
        except Exception:
            return 1e9, np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective_grad, params, method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 10000, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, params


def _optimize_angles(n, edges, p, start_time, timeout):
    """Optimize angles with multiple restarts, L-BFGS-B with exact gradients, and high-precision polish."""
    cut_vals = _precompute_cut_values(n, edges)
    pairs_per_qubit = _precompute_mixer_indices(n)
    
    best_score = -np.inf
    best_params = None
    
    # Measure evaluation speed
    t0 = time.time()
    _qaoa_expectation(n, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals, pairs_per_qubit)
    eval_time = max(time.time() - t0, 0.0001)
    
    # L-BFGS-B with parameter-shift: each gradient eval costs 2*p + 1 = 5 function evals
    # Each L-BFGS-B iteration does ~1-3 gradient evals, so ~5-15 function evals per iteration
    time_per_iteration = 10 * eval_time
    
    remaining = timeout - (time.time() - start_time) - 5
    max_total_iterations = max(10, int(remaining / max(time_per_iteration, 0.0001)))
    
    np.random.seed(42)
    
    # Diverse initial points spread across the parameter space
    # For p=2: params = [gamma_0, beta_0, gamma_1, beta_1]
    base_inits = [
        [0.5, 0.5, 0.3, 0.3],
        [0.7, 0.7, 0.4, 0.4],
        [0.3, 0.4, 0.2, 0.3],
        [0.5, 0.3, 0.5, 0.3],
        [0.4, 0.6, 0.4, 0.6],
        [0.6, 0.2, 0.6, 0.2],
        [0.8, 0.5, 0.5, 0.4],
        [0.2, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.45, 0.35, 0.45, 0.35],
        [0.35, 0.25, 0.35, 0.25],
        [0.6, 0.4, 0.5, 0.4],
        [0.4, 0.5, 0.3, 0.4],
        [0.55, 0.45, 0.45, 0.55],
        [0.45, 0.55, 0.55, 0.45],
        [0.9, 0.3, 0.7, 0.3],
        [0.1, 0.8, 0.2, 0.7],
        [0.65, 0.65, 0.55, 0.55],
        [0.5, 0.2, 0.5, 0.2],
        [0.3, 0.6, 0.3, 0.6],
        [0.7, 0.3, 0.7, 0.3],
        [0.4, 0.4, 0.4, 0.4],
        [0.6, 0.6, 0.6, 0.6],
    ]
    
    # Determine how many restarts we can afford
    # L-BFGS-B is more efficient per iteration than Powell
    num_restarts = min(len(base_inits), max(8, max_total_iterations // 15))
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 15:
            break
        
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform([0.05, 0.05, 0.05, 0.05], [1.5, 1.5, 1.5, 1.5])
        
        # Limit iterations per restart based on remaining time
        time_remaining = timeout - (time.time() - start_time) - 12
        max_iter_this = max(30, min(500, int(time_remaining / max(time_per_iteration, 0.0001) / max(num_restarts - r, 1))))
        
        score, params = _optimize_single(n, init, cut_vals, pairs_per_qubit, max_iter=max_iter_this)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # High-precision polish pass
    if best_params is not None and time.time() - start_time < timeout - 10:
        polish_score, polish_params = _polish(n, best_params, cut_vals, pairs_per_qubit, max_iter=1000)
        if polish_score > best_score:
            best_score = polish_score
            best_params = polish_params.copy()
    
    # Second polish from slightly perturbed best if time allows
    if best_params is not None and time.time() - start_time < timeout - 8:
        np.random.seed(999)
        perturbed = best_params + np.random.normal(0, 0.01, 4)
        perturbed = np.clip(perturbed, 0.0, np.pi)
        polish_score2, polish_params2 = _polish(n, perturbed, cut_vals, pairs_per_qubit, max_iter=500)
        if polish_score2 > best_score:
            best_score = polish_score2
            best_params = polish_params2.copy()
    
    # Third polish from another perturbation if time allows
    if best_params is not None and time.time() - start_time < timeout - 6:
        np.random.seed(1234)
        perturbed2 = best_params + np.random.normal(0, 0.005, 4)
        perturbed2 = np.clip(perturbed2, 0.0, np.pi)
        polish_score3, polish_params3 = _polish(n, perturbed2, cut_vals, pairs_per_qubit, max_iter=300)
        if polish_score3 > best_score:
            best_score = polish_score3
            best_params = polish_params3.copy()
    
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


def angles(n, edges, p):
    start_time = time.time()
    timeout = 140  # Safety margin from 150s limit
    
    try:
        optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    except Exception:
        return np.array([[0.35, 0.25]] * p)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
    
    return optimal_angles
# EVOLVE-BLOCK-END
