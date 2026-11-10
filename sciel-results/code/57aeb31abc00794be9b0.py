"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import roots_legendre

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Use Sigmoid parameterization for nodes to ensure (0,1) and Softmax for weights 
       to ensure positivity and sum=1.
    2. Pre-compute a dense set of reference integrals for a grid of CPL cosmologies.
    3. Use a two-stage optimization:
       a. Global search (Differential Evolution) on a subset of critical cases.
       b. Local refinement (L-BFGS-B) on the full set of cases.
    4. Minimize the worst-case relative error (max norm) using a smooth approximation 
       or direct max for the final stage.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    # 1. Define representative CPL cosmologies
    # We need a good spread of w0, wa and z to capture the worst-case error.
    # The integrand 1/E(a) varies significantly with w0, wa.
    Om_m = 0.3
    Om_de = 0.7

    # Grid of cosmological parameters
    # w0 in [-1.5, 0.5], wa in [-1.0, 1.0]
    w0_values = np.array([-1.5, -1.0, -0.5, 0.0, 0.5])
    wa_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    # Redshifts: cover low to high z
    z_values = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0, 6.0])

    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model. Vectorized."""
        matter = Om_m / a**3
        # Handle w0+wa == 0 case carefully to avoid division by zero in exponent
        denom = 1.0 + w0 + wa
        # If denom is close to 0, the term is a**(-3*(1+w0))
        # Otherwise: a**(-3*denom) * (1 + wa*(1-a))**(3*wa/denom)
        
        # To be safe and fast, we can use np.where, but logic branching is safer for stability
        # Actually, for numerical stability in optimization, we should avoid exact singularities.
        # The singularities are at specific (w0, wa). We can add a tiny epsilon or handle it.
        # Let's use a stable formulation.
        
        # Standard CPL: rho_de = rho_de0 * (1+z)**(3(1+w0+wa)) * (1+w0+w0*z)**(3*wa/(1+w0+wa))
        # In terms of a: (1+z) = 1/a.
        # rho_de = rho_de0 * a**(-3(1+w0+wa)) * (1 + wa*(1-a))**(3*wa/(1+w0+wa))
        
        # If |1+w0+wa| < 1e-10:
        #   rho_de = rho_de0 * a**(-3(1+w0))
        
        abs_denom = np.abs(denom)
        # We are computing for a single (w0, wa) pair at a time in the loop below, 
        # but we want to vectorize over 'a' for a given (w0, wa).
        
        if abs_denom < 1e-10:
            de_term = Om_de * a**(-3.0 * (1.0 + w0))
        else:
            exponent = 3.0 * wa / denom
            # (1 + wa*(1-a)) can be negative if wa < 0 and a is small?
            # 1 + wa - wa*a. If wa = -1, 1-1+a = a. Positive.
            # If wa = -2, 1-2+2a = 2a-1. Can be negative for a<0.5.
            # However, standard CPL is usually defined for positive density.
            # We assume the evaluator uses a standard definition where the term is well-defined.
            # To be safe, we take abs if the base is negative and exponent is non-integer?
            # Usually, these models are constrained to positive energy density.
            base = 1.0 + wa * (1.0 - a)
            # If base < 0 and exponent is not integer, this is NaN.
            # We assume the test cases are in the physical domain.
            de_term = Om_de * a**(-3.0 * denom) * np.power(base, exponent)
        
        return np.sqrt(matter + de_term)

    def reference_integral(z, w0, wa, n_ref=2001):
        """High-accuracy reference integral using Simpson's rule."""
        u = np.linspace(0, 1, n_ref)
        a_vals = 1.0 / (1.0 + z * u)
        E_vals = E_of_a(a_vals, w0, wa)
        integrand = 1.0 / E_vals
        h = 1.0 / (n_ref - 1)
        # Simpson's rule requires even number of intervals, so odd number of points.
        # n_ref=2001 is odd, so 2000 intervals. Correct.
        integral = h * (integrand[0] + integrand[-1] + 
                        4.0*np.sum(integrand[1:-1:2]) + 
                        2.0*np.sum(integrand[2:-1:2]))
        return z * integral

    # 2. Pre-compute reference values
    ref_vals = []
    test_cases = []  # list of (w0, wa, z)
    
    for w0 in w0_values:
        for wa in wa_values:
            for z in z_values:
                if time.time() > deadline - 20: # Reserve 20s for optimization
                    break
                try:
                    rv = reference_integral(z, w0, wa)
                    if rv > 1e-12:
                        ref_vals.append(rv)
                        test_cases.append((w0, wa, z))
                except Exception:
                    continue
    
    if not ref_vals:
        # Fallback
        gl_nodes, gl_weights = roots_legendre(8)
        nodes0 = 0.5 * (gl_nodes + 1.0)
        weights0 = 0.5 * gl_weights
        weights0 = weights0 / np.sum(weights0)
        return {"nodes": nodes0, "weights": weights0}
    
    ref_vals = np.array(ref_vals)
    n_cases = len(ref_vals)
    
    # 3. Parameterization
    # Nodes x_i -> u_i = sigmoid(x_i)
    # Weights y_i -> w_i = softmax(y_i)
    
    def params_to_nodes_weights(params):
        x = params[:8]
        y = params[8:]
        # Sigmoid
        u = 1.0 / (1.0 + np.exp(-x))
        # Enforce strict ordering by adding a small monotonic constraint?
        # Sigmoid is monotonic if x is sorted. 
        # Differential Evolution doesn't enforce ordering.
        # We can sort the resulting nodes? No, that breaks the mapping to weights.
        # Instead, we can penalize non-ordering in the objective.
        
        # Softmax
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        w = exp_y / np.sum(exp_y)
        
        return u, w

    def compute_errors(u, w):
        """Compute relative errors for all test cases."""
        errors = np.zeros(n_cases)
        # Vectorize over cases?
        # For each case, we need to compute 1/E at the 8 nodes.
        # We can loop over cases, but vectorize the node evaluation.
        
        for i, (w0, wa, z) in enumerate(test_cases):
            a_vals = 1.0 / (1.0 + z * u)
            E_vals = E_of_a(a_vals, w0, wa)
            node_vals = 1.0 / E_vals
            quad_val = z * np.dot(w, node_vals)
            errors[i] = np.abs(quad_val - ref_vals[i]) / ref_vals[i]
        return errors

    def objective_full(params):
        """Objective for local optimization: Max relative error."""
        if time.time() > deadline - 5:
            return 1e10
        u, w = params_to_nodes_weights(params)
        # Check ordering
        if not np.all(np.diff(u) > 1e-8):
            return 1e10
        errors = compute_errors(u, w)
        return np.max(errors)

    def objective_global(params):
        """Objective for global optimization: Smooth max on a subset."""
        if time.time() > deadline - 25:
            return 1e10
        u, w = params_to_nodes_weights(params)
        if not np.all(np.diff(u) > 1e-8):
            return 1e10
        # Use a subset of cases for speed in global search
        # Pick every 2nd case or a random subset
        subset_indices = np.arange(0, n_cases, max(1, n_cases // 50))
        if len(subset_indices) < 10:
            subset_indices = np.random.choice(n_cases, 10, replace=False)
        
        errors = compute_errors(u, w)
        sub_errors = errors[subset_indices]
        
        # Smooth max
        beta = 10.0
        return (1.0 / beta) * np.log(np.sum(np.exp(beta * sub_errors)))

    # 4. Initial Parameters
    # Start with Gauss-Legendre nodes mapped to (0,1) via inverse sigmoid
    gl_nodes, gl_weights = roots_legendre(8)
    u0 = 0.5 * (gl_nodes + 1.0)
    # Clip to avoid log(0) or log(1) issues in inverse sigmoid
    u0 = np.clip(u0, 1e-6, 1 - 1e-6)
    x0 = np.log(u0 / (1 - u0))
    
    # Initial weights: uniform or GL?
    # Softmax of zeros is uniform.
    y0 = np.zeros(8)
    
    params0 = np.concatenate([x0, y0])

    # 5. Global Optimization (Differential Evolution)
    # Bounds for x: sigmoid range. x in [-10, 10] covers (0,1) well.
    # Bounds for y: softmax range. y in [-5, 5] is sufficient.
    bounds = [(-10, 10)] * 8 + [(-5, 5)] * 8
    
    best_params = params0
    best_score = objective_global(params0)
    
    try:
        # DE is stochastic. Use a small population for speed.
        # maxiter=50, popsize=10
        result_de = differential_evolution(objective_global, bounds, 
                                           maxiter=50, popsize=10, 
                                           seed=42, tol=1e-4, 
                                           polish=False)
        if result_de.fun < best_score:
            best_params = result_de.x
            best_score = result_de.fun
    except Exception:
        pass

    # 6. Local Optimization (L-BFGS-B)
    # Polish with the full objective
    try:
        result_lbfgs = minimize(objective_full, best_params, method='L-BFGS-B',
                                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if result_lbfgs.fun < objective_full(best_params):
            best_params = result_lbfgs.x
    except Exception:
        pass

    # 7. Multi-start Local Optimization from different points
    # Try a few random starts to escape local minima
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        # Random x, y
        x_rand = np.random.uniform(-5, 5, 8)
        # Ensure some spread
        x_rand = np.sort(x_rand) # Sigmoid is monotonic, so sorted x -> sorted u
        y_rand = np.random.uniform(-2, 2, 8)
        p_rand = np.concatenate([x_rand, y_rand])
        
        # Quick local polish
        try:
            res = minimize(objective_full, p_rand, method='L-BFGS-B',
                           options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < objective_full(best_params):
                best_params = res.x
        except Exception:
            continue

    # 8. Extract and Validate
    best_nodes, best_weights = params_to_nodes_weights(best_params)
    
    # Ensure strict ordering
    if not np.all(np.diff(best_nodes) > 1e-8):
        # Fallback to GL
        best_nodes = u0
        best_weights = np.ones(8) / 8.0
    else:
        # Re-normalize weights just in case
        best_weights = best_weights / np.sum(best_weights)
        
    # Final Check
    if not (np.all(best_nodes > 0) and np.all(best_nodes < 1)):
        best_nodes = u0
        best_weights = np.ones(8) / 8.0
    if not np.all(best_weights > 0):
        best_weights = np.ones(8) / 8.0
        
    best_weights = best_weights / np.sum(best_weights)
    
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = np.ones(8) / 8.0

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
