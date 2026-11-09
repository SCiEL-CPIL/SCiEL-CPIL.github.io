import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def _time_left():
        return deadline - time.time()

    # Representative cosmologies: (Omega_m, w0, w1)
    # Using the CPL form: E(a) = sqrt(Om*(1/a)^3 + (1-Om)*(1/a)^(3(1+w0+w1*(1-a))))
    # Note: The task description says E(a)=H(a)/H0.
    # Standard CPL: w(a) = w0 + w1(1-a)
    # E(a)^2 = Om a^-3 + (1-Om) a^-3(1+w0+w1(1-a))
    # Let's stick to the simpler z-form used in the current program which was working.
    # E(z) = sqrt(Om (1+z)^3 + (1-Om) (1+z)^(3(1+w0+w1 z)))
    
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 positive)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 negative)
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -1.5, 0.3),   # Strongly evolving
        (0.3, -0.5, -0.3),  # Mild quintessence with evolution
        (0.35, -1.1, 0.2),  # Slightly matter-dominated
        (0.3, 0.0, 0.0),    # Empty universe (EdS is 0.3,0,0? No, EdS is Om=1, w=0. Here Om=0.3, w=0)
        (0.3, -1.0, 1.0),   # Strong evolution
    ]
    
    # Dense grid of redshifts
    z_test = np.linspace(0.05, 2.0, 20)
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        # Handle potential numerical issues with large exponents
        exp_term = 3.0 * (1.0 + w0 + w1 * z)
        val = Om * (1.0 + z)**3 + OL * (1.0 + z)**exp_term
        return np.sqrt(np.maximum(val, 1e-30))
    
    def true_integral(z, Om, w0, w1):
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res

    # Precompute true integrals
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
            
    if _time_left() < 5.0:
        # Fallback: Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # Structural Change: Optimize nodes using Differential Evolution
    # The objective function for a given set of nodes will be the minimum
    # worst-case error achievable by *optimal* weights for those nodes.
    # However, solving for optimal weights at every DE step is expensive.
    # Instead, we can use a heuristic: for a fixed set of nodes, the optimal
    # weights in a least-squares sense can be found via linear algebra.
    # But we want to minimize MAX error, not L2.
    #
    # Alternative Strategy:
    # 1. Use DE to optimize nodes.
    # 2. Inside the objective, for a given node set, solve for weights that
    #    minimize the max relative error. This is an LP problem.
    #    Minimize epsilon subject to:
    #    -epsilon * T_j <= sum(w_i * F_ij) - T_j <= epsilon * T_j
    #    sum(w_i) = 1
    #    w_i >= 0
    #
    # This is a linear program. We can solve it using scipy.optimize.linprog.
    
    def _solve_optimal_weights(nodes):
        # F matrix: (num_tests, 8) where F[j, i] = integrand(node_i) for test j
        # T vector: (num_tests,) true values
        # We want to find w such that |sum(w_i F_ij) - T_j| / T_j <= eps for all j
        # Let x = [w_0, ..., w_7, eps]
        # Minimize eps (x[8])
        # Subject to:
        # sum(w_i F_ij) - T_j <= eps * T_j  =>  sum(w_i F_ij) - eps*T_j <= T_j
        # -(sum(w_i F_ij) - T_j) <= eps * T_j => -sum(w_i F_ij) + eps*T_j <= T_j
        # sum(w_i) = 1
        # w_i >= 0, eps >= 0
        
        num_tests = num_cosmo * num_z
        F = np.zeros((num_tests, 8))
        T = np.zeros(num_tests)
        
        idx = 0
        for i in range(num_cosmo):
            Om, w0, w1 = cosmo_params[i]
            for j in range(num_z):
                z = z_test[j]
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                F[idx, :] = 1.0 / E_vals
                T[idx] = true_vals[i, j]
                idx += 1
        
        # LP Setup
        # Variables: w_0..w_7, eps (9 variables)
        c = np.zeros(9)
        c[8] = 1.0 # Minimize eps
        
        # Inequalities: A_ub x <= b_ub
        # For each test k:
        # 1) sum(w_i F_ki) - eps T_k <= T_k
        #    [F_k0, ..., F_k7, -T_k] . x <= T_k
        # 2) -sum(w_i F_ki) + eps T_k <= T_k
        #    [-F_k0, ..., -F_k7, T_k] . x <= T_k
        
        A_ub = np.zeros((2 * num_tests, 9))
        b_ub = np.zeros(2 * num_tests)
        
        for k in range(num_tests):
            if T[k] < 1e-10:
                # Skip or handle carefully. True integral should be > 0 for z>0.
                continue
            A_ub[2*k, :8] = F[k, :]
            A_ub[2*k, 8] = -T[k]
            b_ub[2*k] = T[k]
            
            A_ub[2*k+1, :8] = -F[k, :]
            A_ub[2*k+1, 8] = T[k]
            b_ub[2*k+1] = T[k]
            
        # Equality: sum(w_i) = 1
        A_eq = np.zeros((1, 9))
        A_eq[0, :8] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w_i >= 0, eps >= 0
        bounds = [(0, None)] * 8 + [(0, None)]
        
        try:
            res = minimize(
                lambda x: x[8], # Placeholder, linprog does the work
                np.zeros(9),
                method='SLSQP', # Not using SLSQP, using linprog below
            )
            # Actually use linprog
            from scipy.optimize import linprog
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[:8], res.x[8]
            else:
                # Fallback to equal weights if LP fails
                w = np.ones(8) / 8.0
                # Calculate max error for fallback
                max_err = 0.0
                for k in range(num_tests):
                    quad_val = np.dot(w, F[k, :])
                    err = abs(quad_val - T[k]) / T[k]
                    if err > max_err:
                        max_err = err
                return w, max_err
        except Exception:
            w = np.ones(8) / 8.0
            max_err = 0.0
            for k in range(num_tests):
                quad_val = np.dot(w, F[k, :])
                err = abs(quad_val - T[k]) / T[k]
                if err > max_err:
                    max_err = err
            return w, max_err

    def objective_nodes(params):
        if _time_left() < 2.0:
            return 1e10
        # Decode nodes from params (logit transformation to keep in (0,1))
        nodes_raw = 1.0 / (1.0 + np.exp(-params))
        nodes = np.sort(nodes_raw)
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        # Ensure strict increase
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        
        # Solve for optimal weights and get max error
        _, max_err = _solve_optimal_weights(nodes)
        return max_err

    # Initial guess: Gauss-Legendre nodes
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    # Encode GL nodes to params
    gl_params = -np.log(1.0/gl_nodes - 1.0)
    
    best_params = gl_params.copy()
    best_score = objective_nodes(best_params)
    
    # Differential Evolution
    # Bounds for params: logit(1e-6) to logit(1-1e-6)
    # logit(1e-6) ~ -13.8, logit(1-1e-6) ~ 13.8
    bounds = [(-15, 15)] * 8
    
    try:
        # Use DE to optimize nodes
        # Strategy: 'best1bin' or 'best2bin'
        # Maxiter needs to be small to fit in time budget
        # Each objective call involves an LP solve which is fast (~ms) but let's be careful.
        # 8 variables, popsize 10 -> 80 evals per gen.
        # 50 gens -> 4000 evals. If each is 1ms, that's 4s. Very safe.
        de_res = differential_evolution(
            objective_nodes,
            bounds,
            seed=42,
            maxiter=50,
            popsize=10,
            tol=1e-8,
            polish=True, # Try to polish with L-BFGS-B at the end
            init='sobol' # Better initialization
        )
        if de_res.fun < best_score:
            best_score = de_res.fun
            best_params = de_res.x
    except Exception:
        pass
    
    # Final node extraction
    nodes_final_raw = 1.0 / (1.0 + np.exp(-best_params))
    nodes_final = np.sort(nodes_final_raw)
    nodes_final = np.clip(nodes_final, 1e-6, 1.0 - 1e-6)
    for i in range(1, 8):
        if nodes_final[i] <= nodes_final[i-1]:
            nodes_final[i] = nodes_final[i-1] + 1e-6
            
    # Get final optimal weights
    final_weights, _ = _solve_optimal_weights(nodes_final)
    
    # Cleanup weights
    final_weights = np.abs(final_weights)
    if np.sum(final_weights) == 0:
        final_weights = np.ones(8) / 8.0
    else:
        final_weights = final_weights / np.sum(final_weights)
        
    # Ensure strictly positive and sum to 1
    if not np.all(final_weights > 0):
        final_weights = np.ones(8) / 8.0
        
    if abs(np.sum(final_weights) - 1.0) > 1e-12:
        final_weights = final_weights / np.sum(final_weights)
        
    return {"nodes": nodes_final, "weights": final_weights}
