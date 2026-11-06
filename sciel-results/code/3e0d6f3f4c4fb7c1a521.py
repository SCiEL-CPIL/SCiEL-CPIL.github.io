import numpy as np
import time
from scipy.optimize import minimize, nnls
from scipy.special import eval_cheby

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    # High-precision reference quadrature for ground truth
    n_ref = 256
    t_ref, w_ref = np.polynomial.legendre.leggauss(n_ref)
    ref_x = 0.5 * (t_ref + 1.0)
    ref_w = 0.5 * w_ref / np.sum(0.5 * w_ref)

    Omega_m = 0.3
    Omega_de = 1.0 - Omega_m

    def _E_of_a(a, w0, w1):
        a = np.maximum(a, 1e-12)
        w0w1_term = 3.0 * w1 * (1.0 - a) / a
        return np.sqrt(Omega_m * a**(-3.0) + Omega_de * np.exp(w0w1_term) * a**(-3.0*(1.0+w0)))

    def _f(z, w0, w1):
        a = 1.0 / (1.0 + z)
        return (1.0 + z) / _E_of_a(a, w0, w1)

    def _D_true(zmax, w0, w1):
        z_pts = zmax * ref_x
        vals = _f(z_pts, w0, w1)
        return zmax * np.dot(ref_w, vals)

    def _D_quad(zmax, nodes, weights, w0, w1):
        z_pts = zmax * nodes
        vals = _f(z_pts, w0, w1)
        return zmax * np.dot(weights, vals)

    # Test cosmologies and redshifts
    cosmos = [
        (-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5), (-0.9, 0.3),
        (-0.95, -0.3), (-1.2, 0.0), (-0.8, 0.5), (-1.0, 1.0),
        (-1.1, 0.4), (-0.85, -0.4), (-0.7, 0.6), (-1.3, -0.2),
        (-0.95, 0.7), (-1.05, -0.6), (-1.0, 0.8), (-0.9, -0.5),
        (-1.1, 0.6), (-0.85, 0.2), (-1.2, -0.4), (-0.7, -0.3),
        (-1.0, 0.2), (-0.95, -0.2), (-1.15, 0.3), (-0.8, -0.6),
    ]
    z_list = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

    def _worst_rel(nodes, weights):
        wts = np.maximum(weights, 1e-15)
        wts = wts / wts.sum()
        mx = 0.0
        for (w0, w1) in cosmos:
            for z in z_list:
                D_t = _D_true(z, w0, w1)
                D_q = _D_quad(z, nodes, wts, w0, w1)
                r = abs(D_q - D_t) / D_t * 100.0
                if r > mx:
                    mx = r
        return mx

    # --- Structural approach: Chebyshev moment matching + SLSQP ---
    # Express nodes via smooth parameter: x_i = 0.5*(1 + tanh(p_i))
    # This ensures nodes in (0,1) and strictly increasing if p_i increasing.
    # We optimize raw parameters p (8 values, increasing) and weights (8 values, positive).

    def _params_to_nodes_weights(p, w):
        nodes = 0.5 * (1.0 + np.tanh(p))
        nodes = np.maximum(nodes, 1e-10)
        nodes = np.minimum(nodes, 1.0 - 1e-10)
        wts = np.maximum(w, 1e-10)
        wts = wts / wts.sum()
        return nodes, wts

    # Initial guess via Chebyshev-node matching
    # Chebyshev roots on [0,1]: x_i = 0.5*(1+cos((2i+1)pi/(2*8)))
    i_idx = np.arange(8)
    cheb_nodes = 0.5 * (1.0 + np.cos((2*i_idx + 1) * np.pi / 16.0))
    cheb_nodes = np.sort(cheb_nodes)[::-1]  # reverse to increasing
    # Convert to p parameters: p = atanh(2x-1)
    cheb_nodes_clipped = np.clip(cheb_nodes, 1e-6, 1.0-1e-6)
    p_init = np.arctanh(2.0 * cheb_nodes_clipped - 1.0)
    p_init = np.sort(p_init)
    # Ensure strict increase
    for j in range(1, 8):
        if p_init[j] <= p_init[j-1]:
            p_init[j] = p_init[j-1] + 1e-6

    # GL weights as initial weights
    t8, w8 = np.polynomial.legendre.leggauss(8)
    w_init = 0.5 * w8
    w_init = w_init / w_init.sum()

    best_nodes = _params_to_nodes_weights(p_init, w_init)[0]
    best_weights = _params_to_nodes_weights(p_init, w_init)[1]
    best_err = _worst_rel(best_nodes, best_weights)

    def _obj(params):
        if time.time() > deadline:
            return 1e8
        p = params[:8]
        w = params[8:]
        nodes, wts = _params_to_nodes_weights(p, w)
        # Check strict increase
        if np.any(np.diff(nodes) < 1e-10):
            return 1e8
        return _worst_rel(nodes, wts)

    # SLSQP optimization
    x0 = np.concatenate([p_init, w_init])
    bounds = [(-6.0, 6.0)] * 8 + [(1e-8, 10.0)] * 8
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(np.maximum(x[8:], 1e-10)) - 1.0},
        {'type': 'ineq', 'fun': lambda x: np.min(np.diff(np.tanh(x[:8]))) - 1e-8},
    ]

    x_current = x0.copy()
    for iteration in range(30):
        if time.time() > deadline - 3:
            break

        try:
            res = minimize(_obj, x_current, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-10})
            if res.fun < best_err:
                n_opt, w_opt = _params_to_nodes_weights(res.x[:8], res.x[8:])
                best_nodes = n_opt
                best_weights = w_opt
                best_err = res.fun
                x_current = res.x.copy()
            elif time.time() < deadline - 5:
                # If stuck, try a different initialization
                # Use Gauss-Lobatto-like nodes or random perturbations
                if iteration % 3 == 1:
                    # Perturb current best in p-space
                    p_pert = best_nodes  # not directly, recompute from current best
                    p_pert = np.arctanh(2.0 * np.clip(best_nodes, 1e-6, 1-1e-6))
                    p_pert = p_pert + np.random.normal(0, 0.3, 8)
                    p_pert = np.sort(p_pert)
                    for j in range(1, 8):
                        if p_pert[j] <= p_pert[j-1]:
                            p_pert[j] = p_pert[j-1] + 1e-4
                    w_pert = best_weights + np.random.normal(0, 0.02, 8)
                    w_pert = np.maximum(w_pert, 1e-6)
                    w_pert = w_pert / w_pert.sum()
                    x_current = np.concatenate([p_pert, w_pert])
                elif iteration % 3 == 2:
                    # Try Legendre-Gauss initial again with different seed
                    np.random.seed(iteration * 137)
                    p_try = np.sort(np.random.uniform(-3, 3, 8))
                    for j in range(1, 8):
                        if p_try[j] <= p_try[j-1]:
                            p_try[j] = p_try[j-1] + 1e-4
                    w_try = np.random.dirichlet(np.ones(8))
                    x_current = np.concatenate([p_try, w_try])
        except Exception:
            pass

    # Final validation and cleanup
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-7
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / best_weights.sum()

    # Verify sum constraint
    assert abs(best_weights.sum() - 1.0) < 1e-12, f"Weight sum: {best_weights.sum()}"

    return {"nodes": best_nodes, "weights": best_weights}
