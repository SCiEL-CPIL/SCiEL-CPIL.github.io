"""QAOA parameter design with exact combinatorial probability aggregation and gradient-based optimization."""
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


def _qaoa_probabilities(n, edges, gammas, betas):
    """
    Compute exact probabilities for all 2^n masks using combinatorial aggregation.
    
    For p=2 QAOA with unweighted MaxCut:
    |psi> = U_B1 * U_C1 * U_B0 * U_C0 * |+>^n
    
    where U_C = exp(-i*gamma*C) and U_B = exp(-i*beta*sum(X_i))
    
    The key insight: since the cost operator C = sum_{(u,v) in E} Z_u Z_v / 2
    (or equivalently, C counts edges cut), and the mixer is a product of single-qubit
    rotations, we can compute the probability of each mask by tracking the amplitude
    through the circuit layer by layer.
    
    However, for p=2, we can be more clever. Let's use the fact that:
    - After U_C0: each basis state |x> gets phase exp(-i*gamma0*cut(x))
    - After U_B0: each qubit is rotated by beta0
    - After U_C1: each basis state |y> gets phase exp(-i*gamma1*cut(y))
    - After U_B1: each qubit is rotated by beta1
    
    The final amplitude for mask z is:
    <z|psi> = sum_x sum_y <z|U_B1|y> * exp(-i*gamma1*cut(y)) * <y|U_B0|x> * exp(-i*gamma0*cut(x)) * <x|+>^n
    
    Since <x|+>^n = 1/sqrt(2^n) for all x, and U_B = tensor product of single-qubit
    rotations R_x(beta) = cos(beta)*I - i*sin(beta)*X, we have:
    <y|U_B|x> = prod_i <y_i|x_i>_{R_x(beta)}
    
    where <0|0> = cos(beta), <0|1> = -i*sin(beta), <1|0> = -i*sin(beta), <1|1> = cos(beta)
    
    So <y|U_B|x> = prod_i [cos(beta) if x_i==y_i else -i*sin(beta)]
    
    This means the probability |<z|psi>|^2 involves a double sum over x,y, which is O(4^n).
    For n=10, that's 1M operations per evaluation - still feasible but slow.
    
    Alternative: just do the state vector simulation but vectorize it better.
    For n<=10, 2^n <= 1024, and the mixer application is O(n*2^n) = O(10*1024) ~ 10k ops.
    That's fast enough. Let's stick with state vector but optimize the mixer.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    cut_vals = _precompute_cut_values(n, edges)
    
    for k in range(len(gammas)):
        # Apply cost layer: psi -> psi * exp(-i*gamma_k*cut_vals)
        psi = psi * np.exp(-1j * gammas[k] * cut_vals)
        
        # Apply mixer: exp(-i*beta_k*sum(X_i))
        beta = betas[k]
        cb = np.cos(beta)
        sb = -1j * np.sin(beta)
        
        # Apply X rotation on each qubit
        for i in range(n):
            # Split into even and odd indices for qubit i
            # Qubit i is the i-th bit (0-indexed from LSB)
            # Masks where bit i is 0: mask & (1<<i) == 0
            # Masks where bit i is 1: mask & (1<<i) != 0
            
            mask_i = 1 << i
            # For each pair (mask with bit i=0, mask with bit i=1)
            # They differ only in bit i
            # Index of the partner: mask ^ mask_i
            
            # Vectorized: for each mask, partner = mask ^ mask_i
            # We want: new_psi[mask] = cb*psi[mask] + sb*psi[mask^mask_i]
            # This is symmetric, so we can do it in-place carefully
            
            # Create index array
            idx = np.arange(dim)
            partner = idx ^ mask_i
            
            # We need to be careful: if we update in place, we corrupt values
            # So use a copy
            psi_new = np.empty_like(psi)
            psi_new[idx] = cb * psi[idx] + sb * psi[partner]
            psi = psi_new
    
    return np.abs(psi) ** 2


def _qaoa_expectation(n, edges, gammas, betas, cut_vals=None):
    """Compute expected cut size."""
    if cut_vals is None:
        cut_vals = _precompute_cut_values(n, edges)
    probs = _qaoa_probabilities(n, edges, gammas, betas)
    return float(np.dot(probs, cut_vals))


def _qaoa_expectation_fast(n, edges, gammas, betas, cut_vals):
    """Faster version that reuses precomputed cut_vals."""
    probs = _qaoa_probabilities(n, edges, gammas, betas)
    return float(np.dot(probs, cut_vals))


def _qaoa_gradient(n, edges, params, cut_vals):
    """Compute gradient using central finite differences."""
    eps = 1e-6
    grad = np.zeros(4)
    for i in range(4):
        pp = params.copy()
        pm = params.copy()
        pp[i] += eps
        pm[i] -= eps
        
        gp = np.array([pp[0], pp[2]])
        bp = np.array([pp[1], pp[3]])
        gm = np.array([pm[0], pm[2]])
        bm = np.array([pm[1], pm[3]])
        
        fp = _qaoa_expectation_fast(n, edges, gp, bp, cut_vals)
        fm = _qaoa_expectation_fast(n, edges, gm, bm, cut_vals)
        grad[i] = (fp - fm) / (2 * eps)
    
    return grad


def _optimize_single(n, edges, p, init_params, cut_vals, max_iter=300):
    """Run one optimization from a given initial point."""
    
    def objective(params):
        try:
            gammas = np.array([params[0], params[2]])
            betas = np.array([params[1], params[3]])
            return -_qaoa_expectation_fast(n, edges, gammas, betas, cut_vals)
        except Exception:
            return 1e9
    
    def gradient(params):
        try:
            return -_qaoa_gradient(n, edges, params, cut_vals)
        except Exception:
            return np.zeros(4)
    
    try:
        res = scipy.optimize.minimize(
            objective, init_params, jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, np.pi)] * 4,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        return -res.fun, res.x
    except Exception:
        return -1e9, init_params


def _optimize_angles(n, edges, p, start_time, timeout):
    """Optimize angles with multiple restarts and polish."""
    cut_vals = _precompute_cut_values(n, edges)
    
    best_score = -np.inf
    best_params = None
    
    # Measure evaluation speed
    t0 = time.time()
    _qaoa_expectation_fast(n, edges, np.array([0.5, 0.5]), np.array([0.5, 0.5]), cut_vals)
    eval_time = max(time.time() - t0, 0.0001)
    
    # Each gradient evaluation costs ~8 function evaluations (4 params * 2 for central diff)
    # L-BFGS-B typically does 1 function eval + 1 gradient eval per iteration
    # So ~9 evals per iteration
    time_per_iteration = 9 * eval_time
    
    remaining = timeout - (time.time() - start_time) - 5
    max_total_iterations = max(10, int(remaining / max(time_per_iteration, 0.0001)))
    
    np.random.seed(42)
    
    # Diverse initial points
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
    ]
    
    # Determine how many restarts we can afford
    # Each restart uses up to max_iter iterations
    # But we want to spread iterations across restarts
    num_restarts = min(len(base_inits), max(3, max_total_iterations // 50))
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 8:
            break
        
        if r < len(base_inits):
            init = np.array(base_inits[r], dtype=np.float64)
        else:
            init = np.random.uniform([0.1, 0.1, 0.1, 0.1], [1.5, 1.5, 1.5, 1.5])
        
        # Limit iterations per restart based on remaining time
        time_remaining = timeout - (time.time() - start_time) - 5
        max_iter_this = max(20, min(300, int(time_remaining / max(time_per_iteration, 0.0001) / (num_restarts - r))))
        
        score, params = _optimize_single(n, edges, p, init, cut_vals, max_iter=max_iter_this)
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    # Polish with Nelder-Mead if we have time
    if best_params is not None and time.time() - start_time < timeout - 5:
        def objective(params):
            try:
                gammas = np.array([params[0], params[2]])
                betas = np.array([params[1], params[3]])
                return -_qaoa_expectation_fast(n, edges, gammas, betas, cut_vals)
            except Exception:
                return 1e9
        
        try:
            res_polish = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            polish_score = -res_polish.fun
            if polish_score > best_score:
                best_score = polish_score
                best_params = res_polish.x
        except Exception:
            pass
    
    if best_params is not None:
        return np.array([[best_params[0], best_params[1]],
                         [best_params[2], best_params[3]]])
    else:
        return np.array([[0.35, 0.25], [0.35, 0.25]])


def angles(n, edges, p):
    start_time = time.time()
    timeout = 145  # Safety margin from 150s limit
    
    try:
        optimal_angles = _optimize_angles(n, edges, p, start_time, timeout)
    except Exception:
        return np.array([[0.35, 0.25]] * p)
    
    if time.time() - start_time > timeout:
        return np.array([[0.35, 0.25]] * p)
    
    return optimal_angles
# EVOLVE-BLOCK-END
