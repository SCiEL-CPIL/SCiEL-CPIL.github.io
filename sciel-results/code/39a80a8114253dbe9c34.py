"""Search-based approach: generate diverse candidates, evaluate against reference, keep best."""
import numpy as np
import time
import scipy.integrate as si
import scipy.optimize as so

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill

    Omega_m = 0.3
    Omega_L = 0.7

    # Reference integration using high-order Gauss-Legendre (64 points is very accurate)
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(64)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    def _reference_integral(z, w0, wa):
        """Compute integral of 1/E(a) da from a=1 to a=1/(1+z) using 64-point GL."""
        a_vals = 1.0 / (1.0 + z * gl_nodes)
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E
        return np.dot(gl_weights, inv_E)

    # Test cases for evaluation
    z_vals = np.logspace(-1, np.log10(5.0), 8)
    w0_vals = [-1.5, -1.0, -0.5, 0.0, 0.5]
    wa_vals = [-1.5, -0.5, 0.0, 0.5, 1.5]

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    # Precompute reference values
    ref_vals = []
    for z, w0, wa in test_cases:
        if time.time() > deadline:
            ref_vals.append(1.0)
            continue
        try:
            val = _reference_integral(z, w0, wa)
            ref_vals.append(val)
        except Exception:
            ref_vals.append(1.0)
    ref_vals = np.array(ref_vals)

    # Precompute test case arrays for vectorized evaluation
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _quadrature_eval_vec(nodes, weights):
        """Evaluate the 8-point quadrature for all test cases at once."""
        # nodes: (8,), weights: (8,)
        # z_arr: (N,), w0_arr: (N,), wa_arr: (N,)
        # a_vals[i,j] = 1/(1 + z_i * nodes_j)
        a_vals = 1.0 / (1.0 + z_arr[:, None] * nodes[None, :])  # (N, 8)
        a_vals = np.clip(a_vals, 1e-12, 1.0)
        
        # log_term[i,j] = 3*(1+w0_i+wa_i)*log(a_ij) - 3*wa_i*(a_ij-1)
        log_term = 3.0 * (1.0 + w0_arr[:, None] + wa_arr[:, None]) * np.log(a_vals) \
                  - 3.0 * wa_arr[:, None] * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        inv_E = 1.0 / E  # (N, 8)
        
        # quadrature values: q_i = sum_j weights_j * inv_E[i,j]
        q_vals = inv_E @ weights  # (N,)
        return q_vals

    def _worst_rel_error(nodes, weights):
        """Compute worst-case relative error in percent."""
        q_vals = _quadrature_eval_vec(nodes, weights)
        rel_errs = np.abs(q_vals - ref_vals) / ref_vals * 100.0
        return float(np.max(rel_errs))

    def _soft_worst_rel_error(nodes, weights, beta=50.0):
        """Smooth approximation of worst-case relative error using log-sum-exp."""
        q_vals = _quadrature_eval_vec(nodes, weights)
        rel_errs = np.abs(q_vals - ref_vals) / ref_vals * 100.0
        # log-sum-exp for smooth max
        max_val = np.max(rel_errs)
        if max_val < 1e-30:
            return 0.0
        scaled = rel_errs / max_val
        log_sum = np.log(np.sum(np.exp(beta * (scaled - 1.0)))) / beta
        return max_val * (1.0 + log_sum / beta)

    # Generate diverse candidate rules
    candidates = []

    # 1. Standard Gauss-Legendre on [0,1]
    nodes, weights = np.polynomial.legendre.leggauss(8)
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    candidates.append((nodes, weights, "gauss_legendre"))

    # 2. Gauss-Laguerre mapped to [0,1]
    try:
        nodes_l, weights_l = np.polynomial.laguerre.laggauss(8)
        # Map from [0, inf) to [0, 1]: u = 1 - exp(-x)
        nodes_m = 1.0 - np.exp(-nodes_l)
        # Adjust weights: du/dx = exp(-x)
        weights_m = weights_l * np.exp(-nodes_l)
        weights_m = weights_m / np.sum(weights_m)
        candidates.append((nodes_m, weights_m, "gauss_laguerre_mapped"))
    except Exception:
        pass

    # 3. Clenshaw-Curtis
    try:
        N = 8
        theta = np.pi * np.arange(N) / (N - 1)
        nodes_cc = 0.5 * (1.0 + np.cos(theta))
        # Reverse to get increasing order
        nodes_cc = nodes_cc[::-1]
        weights_cc = np.zeros(N)
        weights_cc[0] = 1.0 / (N - 1) ** 2
        weights_cc[-1] = 1.0 / (N - 1) ** 2
        for j in range(1, N - 1):
            s = 0.0
            for k in range(2, N - 1, 2):
                s += 2.0 * np.cos(k * theta[j]) / (k ** 2 - 1)
            weights_cc[j] = s / (N - 1) ** 2
        weights_cc = weights_cc[::-1]
        weights_cc = np.abs(weights_cc)
        weights_cc = weights_cc / np.sum(weights_cc)
        candidates.append((nodes_cc, weights_cc, "clenshaw_curtis"))
    except Exception:
        pass

    # 4. Evenly spaced
    nodes_even = np.linspace(0.05, 0.95, 8)
    weights_even = np.ones(8) / 8.0
    candidates.append((nodes_even, weights_even, "even"))

    # 5. Log-spaced nodes
    nodes_log = np.exp(np.linspace(np.log(0.01), np.log(0.99), 8))
    # Trapezoidal weights for log-spaced
    weights_log = np.ones(8)
    weights_log[0] = (nodes_log[1] - nodes_log[0]) / 2
    weights_log[-1] = (nodes_log[-1] - nodes_log[-2]) / 2
    for i in range(1, 7):
        weights_log[i] = (nodes_log[i+1] - nodes_log[i-1]) / 2
    weights_log = weights_log / np.sum(weights_log)
    candidates.append((nodes_log, weights_log, "log_spaced"))

    # 6. Gauss-Jacobi with various parameters
    from scipy.special import roots_jacobi
    for alpha, beta in [(0, 0), (1, 0), (0, 1), (1, 1), (0.5, 0.5), (2, 1), (1, 2), (0.5, 2), (2, 0.5)]:
        try:
            nodes_j, weights_j = roots_jacobi(8, alpha, beta)
            nodes_j = (nodes_j + 1.0) / 2.0
            weights_j = weights_j / 2.0
            if np.all(np.diff(nodes_j) > 0) and np.all(weights_j > 0):
                candidates.append((nodes_j, weights_j, f"jacobi_{alpha}_{beta}"))
        except Exception:
            pass

    # 7. Various beta-weighted Gauss-Legendre (emphasize low or high z)
    for beta in [0.5, 1.0, 2.0, 0.3, 0.7, 1.5, 3.0]:
        try:
            nodes_g, weights_g = np.polynomial.legendre.leggauss(8)
            nodes_g = (nodes_g + 1.0) / 2.0
            # Apply beta weighting: u = v^beta where v is in [0,1]
            nodes_w = nodes_g ** beta
            # Adjust weights: du/dv = beta * v^(beta-1)
            weights_w = weights_g * beta * (nodes_g ** (beta - 1))
            weights_w = np.abs(weights_w)
            weights_w = weights_w / np.sum(weights_w)
            if np.all(np.diff(nodes_w) > 0):
                candidates.append((nodes_w, weights_w, f"beta_{beta}"))
        except Exception:
            pass

    # 8. Random candidates with random smooth node distributions
    rng = np.random.RandomState(42)
    for seed in range(50):
        rng_local = np.random.RandomState(seed)
        # Generate random strictly increasing nodes in (0,1)
        gaps = rng_local.exponential(1.0, size=9)
        gaps = gaps / gaps.sum()
        nodes_r = np.cumsum(gaps)
        nodes_r = np.clip(nodes_r, 1e-6, 1.0 - 1e-6)
        # Ensure strictly increasing
        for i in range(1, 8):
            if nodes_r[i] <= nodes_r[i-1]:
                nodes_r[i] = nodes_r[i-1] + 1e-6
        # Simple trapezoidal weights
        weights_r = np.ones(8)
        weights_r[0] = (nodes_r[1] - nodes_r[0]) / 2
        weights_r[-1] = (nodes_r[-1] - nodes_r[-2]) / 2
        for i in range(1, 7):
            weights_r[i] = (nodes_r[i+1] - nodes_r[i-1]) / 2
        weights_r = np.maximum(weights_r, 1e-12)
        weights_r = weights_r / np.sum(weights_r)
        candidates.append((nodes_r, weights_r, f"random_{seed}"))

    # 9. Gauss-Radau (one endpoint fixed)
    for endpoint in [0.0, 1.0]:
        try:
            from scipy.special import roots_radau
            if endpoint == 0.0:
                nodes_r, weights_r = roots_radau(8, 1.0)  # right endpoint fixed
                nodes_r = (nodes_r + 1.0) / 2.0
                weights_r = weights_r / 2.0
            else:
                nodes_r, weights_r = roots_radau(8, -1.0)  # left endpoint fixed
                nodes_r = (nodes_r + 1.0) / 2.0
                weights_r = weights_r / 2.0
            if np.all(np.diff(nodes_r) > 0) and np.all(weights_r > 0):
                candidates.append((nodes_r, weights_r, f"radau_{endpoint}"))
        except Exception:
            pass

    # 10. Fejer rules
    try:
        N = 8
        nodes_fejer = np.cos(np.pi * np.arange(N) / (N - 1))
        nodes_fejer = 0.5 * (1.0 + nodes_fejer)
        nodes_fejer = nodes_fejer[::-1]  # increasing
        weights_fejer = np.zeros(N)
        weights_fejer[0] = 1.0 / (N - 1) ** 2
        weights_fejer[-1] = 1.0 / (N - 1) ** 2
        for j in range(1, N - 1):
            s = 0.0
            for k in range(2, N - 1, 2):
                s += 2.0 * np.cos(k * np.pi * j / (N - 1)) / (k ** 2 - 1)
            weights_fejer[j] = s / (N - 1) ** 2
        weights_fejer = weights_fejer[::-1]
        weights_fejer = np.abs(weights_fejer)
        weights_fejer = weights_fejer / np.sum(weights_fejer)
        candidates.append((nodes_fejer, weights_fejer, "fejer"))
    except Exception:
        pass

    # Evaluate all candidates
    best_worst_err = float('inf')
    best_nodes = candidates[0][0]
    best_weights = candidates[0][1]
    best_name = candidates[0][2]

    for nodes_c, weights_c, name in candidates:
        if time.time() > deadline - 15.0:
            break
        try:
            worst_err = _worst_rel_error(nodes_c, weights_c)
            if worst_err < best_worst_err:
                best_worst_err = worst_err
                best_nodes = nodes_c.copy()
                best_weights = weights_c.copy()
                best_name = name
        except Exception:
            pass

    # Refine the best candidate using optimization
    def _param_to_nodes_weights(params):
        """Parameterize nodes and weights to ensure constraints."""
        # params: [x1, y1..y7, logw1..logw8]
        x1 = params[0]
        y = params[1:8]
        logw = params[8:16]

        x = np.zeros(8)
        x[0] = x1
        for i in range(1, 8):
            x[i] = x[i-1] + np.exp(y[i-1])

        nodes = 1.0 / (1.0 + np.exp(-x))
        nodes = np.clip(nodes, 1e-12, 1.0 - 1e-12)

        w = np.exp(logw)
        w = w / np.sum(w)
        return nodes, w

    def _nodes_weights_to_param(nodes, weights):
        """Convert nodes/weights to parameter space."""
        x = np.log(nodes / (1.0 - nodes))
        params = np.zeros(16)
        params[0] = x[0]
        for i in range(1, 8):
            gap = x[i] - x[i-1]
            if gap <= 1e-12:
                gap = 1e-12
            params[1+i-1] = np.log(gap)
        params[8:16] = np.log(np.maximum(weights, 1e-12))
        return params

    # Multiple optimization passes from different starting points
    if time.time() < deadline - 10.0:
        start_params = _nodes_weights_to_param(best_nodes, best_weights)

        def _objective(params):
            nodes, weights = _param_to_nodes_weights(params)
            return _worst_rel_error(nodes, weights)

        # Try Nelder-Mead with smooth objective first
        try:
            def _objective_smooth(params):
                nodes, weights = _param_to_nodes_weights(params)
                return _soft_worst_rel_error(nodes, weights, beta=50.0)
            
            res = so.minimize(_objective_smooth, start_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_worst_err:
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
                best_worst_err = _worst_rel_error(best_nodes, best_weights)
        except Exception:
            pass

        # Try Nelder-Mead with exact objective
        try:
            start_params = _nodes_weights_to_param(best_nodes, best_weights)
            res = so.minimize(_objective, start_params, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res.fun < best_worst_err:
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
                best_worst_err = res.fun
        except Exception:
            pass

    # Try COBYLA with smooth objective
    if time.time() < deadline - 5.0:
        try:
            start_params = _nodes_weights_to_param(best_nodes, best_weights)

            def _objective_cobyla(params):
                nodes, weights = _param_to_nodes_weights(params)
                return _soft_worst_rel_error(nodes, weights, beta=50.0)

            res = so.minimize(_objective_cobyla, start_params, method='COBYLA',
                              options={'maxiter': 3000, 'rhobnd': 1e-14})
            if res.fun < best_worst_err:
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
                best_worst_err = _worst_rel_error(best_nodes, best_weights)
        except Exception:
            pass

    # Final polish: use L-BFGS-B with tight tolerance
    if time.time() < deadline - 2.0:
        try:
            start_params = _nodes_weights_to_param(best_nodes, best_weights)
            
            def _objective_lbfgs(params):
                nodes, weights = _param_to_nodes_weights(params)
                return _worst_rel_error(nodes, weights)
            
            # Use L-BFGS-B with numerical gradient
            res = so.minimize(_objective_lbfgs, start_params, method='L-BFGS-B',
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            if res.fun < best_worst_err:
                best_nodes, best_weights = _param_to_nodes_weights(res.x)
                best_worst_err = res.fun
        except Exception:
            pass

    # Final cleanup: ensure nodes strictly increasing, weights positive and sum to 1
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
