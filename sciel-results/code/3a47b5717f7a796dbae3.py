import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Diverse CPL models for training
    models = []
    w0_vals = [-1.5, -1.2, -1.0, -0.8, -0.5, -0.2, 0.0, 0.3, 0.5]
    wa_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for w0 in w0_vals:
        for wa in wa_vals:
            models.append((w0, wa))
    # Add extreme cases
    models.append((-2.0, 2.0))
    models.append((-2.0, -2.0))
    models.append((1.0, -1.0))
    models.append((1.0, 1.0))
    models.append((-1.0, -1.0))
    models.append((-1.0, 1.0))
    models.append((0.0, 0.0))

    Om_m = 0.3
    Om_de = 1.0 - Om_m

    def E_of_a(a, w0, wa):
        z = 1.0 / a - 1.0
        w = w0 + wa * z
        E2 = (1.0 + z)**3 + Om_de * (1.0 + z)**(3.0 * (1.0 + w))
        return np.sqrt(np.maximum(E2, 1e-30))

    def reference_integral(w0, wa, n_points=5000):
        z = np.linspace(0, 1, n_points)
        a = 1.0 / (1.0 + z)
        E_vals = E_of_a(a, w0, wa)
        f = (1.0 + z) * z / E_vals
        # Simpson's rule (n_points must give even intervals)
        if (n_points - 1) % 2 != 0:
            n_points -= 1
            z = np.linspace(0, 1, n_points)
            a = 1.0 / (1.0 + z)
            E_vals = E_of_a(a, w0, wa)
            f = (1.0 + z) * z / E_vals
        h = 1.0 / (n_points - 1)
        result = h/3.0 * (f[0] + f[-1] + 4.0*np.sum(f[1:-1:2]) + 2.0*np.sum(f[2:-1:2]))
        return result

    # Precompute reference integrals
    refs = np.array([reference_integral(w0, wa) for (w0, wa) in models])
    n_models = len(models)

    def compute_errors(nodes, weights):
        """Compute relative errors for all models given nodes and weights."""
        a = 1.0 / (1.0 + nodes)
        errors = np.zeros(n_models)
        for i, (w0, wa) in enumerate(models):
            E_vals = E_of_a(a, w0, wa)
            q_val = np.sum(weights * (1.0 + nodes) * nodes / E_vals)
            ref_val = refs[i]
            if ref_val > 1e-10:
                errors[i] = abs(q_val - ref_val) / abs(ref_val)
        return errors

    def objective(params):
        """Minimize max relative error. params: 8 log-spacings + 7 log-weights."""
        # Decode nodes from log-spacings to guarantee strict ordering in (0,1)
        log_spacings = params[:8]
        # Convert to positive spacings via exp
        spacings = np.exp(log_spacings)
        total_spacing = np.sum(spacings)
        # Nodes are cumulative sums of normalized spacings
        nodes = np.cumsum(spacings) / total_spacing
        # Ensure nodes are in (0,1) - they naturally are since all spacings > 0
        # But clip to be safe
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6

        # Decode weights from log-weights (7 free, 8th determined by sum=1)
        log_weights = params[8:15]
        # Convert to positive weights
        w_raw = np.exp(log_weights)
        # 8th weight: we need sum to be 1
        # Let's use: w_i = exp(log_w_i) for i=0..6, w_7 = 1 - sum(w_0..w_6)
        # But this can be negative. Instead, use softmax-like:
        # All 8 weights from 8 log params, but we only have 7 free
        # Better: use 8 log params for weights, then normalize
        # Actually, let's use 8 log-weights and normalize
        # But we only have 7 in params[8:15]. Let me restructure.
        # Use params[8:16] for 8 log-weights
        # But params has 16 elements: 8 for nodes, 8 for weights
        # Let me fix: use all 8 for weights via softmax
        
        # Actually, let me use a different approach:
        # params[0:8] = log-spacings for nodes
        # params[8:16] = log-weights (all 8), then normalize
        
        # But I only defined 15 params above. Let me use 16.
        # For now, use 7 log-weights and derive 8th
        w_0_6 = np.exp(log_weights)
        # Make 8th weight positive: use a sigmoid-like transform
        # Or simpler: just use 8 log-weights. Let me restructure.
        
        # Actually, let's keep it simple: use 8 log-weights and normalize
        # But I need to pass 16 params. Let me just use 8 for weights.
        
        # For this version, I'll use:
        # params[0:8] = log-spacings
        # params[8:16] = log-weights (all 8)
        # Then normalize weights
        
        # But I only have params[8:15] = 7 values. Let me add one more.
        # I'll restructure to use 16 params total.
        
        # For now, let's just use 7 log-weights and make the 8th = 1 - sum
        # with a floor to keep positive
        w_sum = np.sum(w_0_6)
        w_7 = max(1.0 - w_sum, 0.01)
        weights = np.concatenate([w_0_6, [w_7]])
        weights = weights / np.sum(weights)

        errors = compute_errors(nodes, weights)
        return np.max(errors)

    # Better: use 16 params, 8 for node log-spacings, 8 for weight log-values
    def objective16(params):
        log_spacings = params[:8]
        spacings = np.exp(log_spacings)
        nodes = np.cumsum(spacings) / np.sum(spacings)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6

        log_weights = params[8:16]
        w_raw = np.exp(log_weights - np.max(log_weights))
        weights = w_raw / np.sum(w_raw)

        errors = compute_errors(nodes, weights)
        return np.max(errors)

    # Bounds for DE
    # Log-spacings: reasonable range
    lb_spacings = np.full(8, -5.0)  # exp(-5) ~ 0.0067
    ub_spacings = np.full(8, 5.0)   # exp(5) ~ 148
    # Log-weights: reasonable range
    lb_weights = np.full(8, -5.0)
    ub_weights = np.full(8, 5.0)
    bounds = list(zip(np.concatenate([lb_spacings, lb_weights]),
                      np.concatenate([ub_spacings, ub_weights])))

    # Initial population: midpoint rule
    # Midpoint nodes: (i+0.5)/8 for i=0..7
    mid_nodes = (np.arange(8) + 0.5) / 8.0
    # Convert to log-spacings
    mid_spacings = np.diff(np.concatenate([[0.0], mid_nodes, [1.0]]))
    x0_spacings = np.log(mid_spacings)
    x0_weights = np.full(8, 1.0/8.0)
    x0_log_weights = np.log(x0_weights)
    x0 = np.concatenate([x0_spacings, x0_log_weights])

    # Run differential evolution
    best_err = objective16(x0)
    best_params = x0.copy()

    if time.time() - t0 < deadline - 20:
        try:
            de_result = differential_evolution(
                objective16, bounds, seed=42, maxiter=100, popsize=30,
                tol=1e-8, polish=True, init='sobol', updating='immediate'
            )
            if de_result.fun < best_err:
                best_err = de_result.fun
                best_params = de_result.x
        except Exception:
            pass

    # Local refinement with Nelder-Mead
    if time.time() - t0 < deadline - 5:
        try:
            nm_result = minimize(
                objective16, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if nm_result.fun < best_err:
                best_err = nm_result.fun
                best_params = nm_result.x
        except Exception:
            pass

    # Extract final nodes and weights
    log_spacings = best_params[:8]
    spacings = np.exp(log_spacings)
    nodes = np.cumsum(spacings) / np.sum(spacings)
    nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6

    log_weights = best_params[8:16]
    w_raw = np.exp(log_weights - np.max(log_weights))
    weights = w_raw / np.sum(w_raw)

    # Final normalization
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
