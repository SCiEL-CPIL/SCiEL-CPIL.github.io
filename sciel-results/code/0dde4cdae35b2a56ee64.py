"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Exact CPL E(z) for spatially flat universe with Om=0.3
    def E_cpl(z, w0, wa, Omega_m=0.3):
        Omega_DE = 1.0 - Omega_m
        exp_arg = 3.0 * ((w0 + wa) * z - wa * np.log1p(z))
        exp_arg = np.clip(exp_arg, -700, 700)
        return np.sqrt(Omega_m * (1.0 + z)**3 + Omega_DE * np.exp(exp_arg))

    # Diverse set of CPL models (w0, wa)
    test_models = [
        (-1.0, 0.0), (-1.0, 0.3), (-1.0, 0.6), (-1.0, -0.3),
        (-0.5, 0.0), (-0.5, 0.4), (-0.5, -0.2),
        (-1.5, 0.0), (-1.5, 0.3), (-1.5, -0.3),
        (-0.8, 0.3), (-0.8, -0.3),
        (-1.2, 0.4), (-1.2, -0.2),
        (-0.3, 0.2), (-0.3, -0.1),
        (-2.0, 0.2), (-2.0, -0.1),
        (0.0, 0.0), (0.0, 0.3),
        (-1.8, 0.2), (-1.8, -0.2),
        (-0.7, 0.5), (-0.7, -0.4),
        (-1.3, 0.3), (-1.3, -0.3),
        (-0.2, 0.3), (-0.2, -0.2),
        (-1.6, 0.1), (-1.6, -0.1),
        (-1.0, 0.8), (-1.0, -0.5),
    ]

    # Test redshifts
    z_values = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])

    n_models = len(test_models)
    n_z = len(z_values)

    # Precompute true integrals using adaptive quadrature
    true_vals = np.zeros((n_models, n_z))
    for i, (w0, wa) in enumerate(test_models):
        for j, z in enumerate(z_values):
            def integrand(zp, w0=w0, wa=wa):
                return 1.0 / E_cpl(zp, w0, wa)
            result, _ = quad(integrand, 0, z, limit=200)
            true_vals[i, j] = (1.0 + z) * result

    def compute_max_error(nodes, weights):
        """Compute worst-case relative error given nodes and weights."""
        max_err = 0.0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                approx = (1.0 + z) * z * np.dot(weights, 1.0 / E_vals)
                tv = true_vals[i, j]
                if tv > 1e-10:
                    err = abs(approx - tv) / tv
                    if err > max_err:
                        max_err = err
        return max_err

    def solve_weights_lp(nodes):
        """
        Solve for weights that minimize the maximum absolute error
        across all test cases, subject to weights > 0 and sum to 1.
        Uses scipy.optimize.linprog.
        """
        from scipy.optimize import linprog

        # Build the constraint matrix: for each test case (i, j),
        # we have: |sum_k w_k * f_k - t_ij| <= eps
        # This is equivalent to:
        #   sum_k w_k * f_k - t_ij <= eps
        #   -(sum_k w_k * f_k - t_ij) <= eps  =>  -sum_k w_k * f_k + t_ij <= eps
        #
        # Variables: [w_0, ..., w_7, eps]
        # Minimize: eps
        # Subject to: A @ x <= b, w >= 0, sum(w) = 1

        n_cases = n_models * n_z
        n_vars = 9  # 8 weights + eps

        # Build A_ub and b_ub
        A_ub = np.zeros((2 * n_cases, n_vars))
        b_ub = np.zeros(2 * n_cases)

        case_idx = 0
        for i in range(n_models):
            w0, wa = test_models[i]
            for j in range(n_z):
                z = z_values[j]
                z_nodes = z * nodes
                E_vals = E_cpl(z_nodes, w0, wa)
                f_vals = (1.0 + z) * z / E_vals  # coefficients for weights
                t_val = true_vals[i, j]

                # Constraint: sum_k w_k * f_k - t_val <= eps
                A_ub[case_idx, :8] = f_vals
                A_ub[case_idx, 8] = -1.0
                b_ub[case_idx] = 0.0

                # Constraint: -sum_k w_k * f_k + t_val <= eps
                A_ub[case_idx + 1, :8] = -f_vals
                A_ub[case_idx + 1, 8] = -1.0
                b_ub[case_idx + 1] = 0.0

                case_idx += 2

        # Objective: minimize eps
        c = np.zeros(n_vars)
        c[8] = 1.0

        # Bounds: w_k >= 0, eps >= 0
        bounds = [(0, None)] * 8 + [(0, None)]

        # Equality constraint: sum(w) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.success:
                weights = res.x[:8]
                # Ensure positivity and normalize
                weights = np.maximum(weights, 1e-15)
                weights = weights / weights.sum()
                return weights
        except Exception:
            pass

        # Fallback: equal weights
        return np.full(8, 1.0 / 8.0)

    def node_objective(nodes):
        """
        Objective for node optimization: given fixed nodes,
        solve for optimal weights via LP, then compute max relative error.
        """
        if time.time() > deadline - 5:
            return 1e10

        # Check node validity
        if nodes[0] <= 0 or nodes[7] >= 1:
            return 1e10
        for k in range(1, 8):
            if nodes[k] <= nodes[k-1]:
                return 1e10

        weights = solve_weights_lp(nodes)
        return compute_max_error(nodes, weights)

    # Initial guess: Gauss-Legendre nodes
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0

    best_nodes = gl_nodes.copy()
    best_weights = gl_weights.copy()
    best_err = compute_max_error(best_nodes, best_weights)

    # Local refinement of nodes using Nelder-Mead
    # We optimize nodes only; weights are solved via LP for each node configuration
    if time.time() < deadline - 15:
        try:
            result = minimize(node_objective, best_nodes, method='Nelder-Mead',
                             options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8,
                                     'adaptive': True})
            if result.fun < best_err and np.all(np.diff(result.x) > 0):
                best_err = result.fun
                best_nodes = result.x
                best_weights = solve_weights_lp(best_nodes)
        except Exception:
            pass

    # Try differential evolution for global node search if time permits
    if time.time() < deadline - 20:
        try:
            # Define bounds for nodes: strictly increasing in (0,1)
            node_bounds = [(1e-6, 0.125), (0.125, 0.25), (0.25, 0.375), (0.375, 0.5),
                           (0.5, 0.625), (0.625, 0.75), (0.75, 0.875), (0.875, 1.0 - 1e-6)]

            def de_objective(nodes):
                if time.time() > deadline - 5:
                    return 1e10
                if nodes[0] <= 0 or nodes[7] >= 1:
                    return 1e10
                for k in range(1, 8):
                    if nodes[k] <= nodes[k-1]:
                        return 1e10
                weights = solve_weights_lp(nodes)
                return compute_max_error(nodes, weights)

            np.random.seed(42)
            de_result = differential_evolution(
                de_objective,
                node_bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='sobol',
                x0=gl_nodes,
                updating='immediate',
                workers=1,
            )
            if de_result.fun < best_err:
                best_err = de_result.fun
                best_nodes = de_result.x
                best_weights = solve_weights_lp(best_nodes)
        except Exception:
            pass

    # Final local polish on nodes
    if time.time() < deadline - 10:
        try:
            result = minimize(node_objective, best_nodes, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10,
                                     'adaptive': True})
            if result.fun < best_err and np.all(np.diff(result.x) > 0):
                best_err = result.fun
                best_nodes = result.x
                best_weights = solve_weights_lp(best_nodes)
        except Exception:
            pass

    # Final solve for weights with best nodes
    best_weights = solve_weights_lp(best_nodes)

    # Ensure validity
    if not np.all(np.diff(best_nodes) > 0) or best_nodes[0] <= 0 or best_nodes[7] >= 1:
        best_nodes = (np.arange(8) + 0.5) / 8.0
        best_weights = np.full(8, 1.0 / 8.0)
    if not np.all(best_weights > 0):
        best_weights = np.full(8, 1.0 / 8.0)

    best_weights = best_weights / best_weights.sum()

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
