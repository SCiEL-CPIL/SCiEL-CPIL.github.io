"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# We need to find 8 nodes and 8 positive weights such that
# sum_i w_i * f(x_i) approximates integral_0^1 f(x) dx
# for f(x) = z * (1+z) / E(a) where x corresponds to a = 1/(1+z*x), so z = (1-x)/(x*something)... 
# Wait, let's re-read the formula.
# dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
# where a_i = 1/(1 + z * nodes[i])
# Let's change variable to x = nodes[i] in [0,1].
# The integral is over z from 0 to z_max.
# dL = c/H0 * (1+z) * int_0^z dz'/E(a')
# Let's change variable to a. a = 1/(1+z). z = 1/a - 1 = (1-a)/a.
# dz = -1/a^2 da.
# When z=0, a=1. When z=z, a=1/(1+z).
# Integral part: int_{a=1}^{a=1/(1+z)} (1/a - 1) * (-1/a^2) da / E(a)
# = int_{a=1/(1+z)}^{1} (1/a - 1) / a^2 da / E(a)
# = int_{a=1/(1+z)}^{1} (1/a^3 - 1/a^2) da / E(a)
# The formula given is: (1+z) * z * sum_i weights[i] / E(a_i)
# Let's check the integrand in terms of x (which is 'nodes').
# The prompt says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i]).
# This implies the quadrature is approximating an integral of the form:
# int_0^1 g(z, x) dx ?
# Let's look at the structure.
# If we define the integral I(z) = int_0^1 h(z, x) dx
# Then dL = (1+z) z I(z).
# What is h(z, x)?
# From the formula sum_i weights[i]/E(a_i), it looks like the quadrature is approximating int_0^1 dx / E(a(x)).
# Where a(x) = 1/(1+z*x).
# So the integral to approximate is J(z) = int_0^1 1/E(1/(1+z*x)) dx.
# Then dL = (1+z) z J(z).
# We need a fixed quadrature rule (nodes x_i, weights w_i) to approximate J(z) for a range of z.
# The "universal" rule must work for all z (or at least the validation z's).
# The error is relative error in dL.
# Since dL = (1+z)z J(z), and (1+z)z is a known factor, the relative error in dL is the relative error in J(z).
# So we want min_{nodes, weights} max_{z in validation} | sum w_i f_z(x_i) - int_0^1 f_z(x) dx | / | int_0^1 f_z(x) dx |
# where f_z(x) = 1/E(1/(1+z*x)).

# We don't know the exact validation cosmologies or z values.
# We must design a rule that is robust across a family of dark energy models.
# Common models: LambdaCDM (w=-1), CPL (w0, wa).
# E(a)^2 = Omega_m a^{-3} + Omega_r a^{-4} + Omega_k a^{-2} + Omega_L a^{-3(1+w(a))}
# Spatially flat, so Omega_k=0. Omega_m ~ 0.3, Omega_L ~ 0.7.
# w(a) = w0 + wa(1-a).

# Strategy:
# 1. Define a set of representative cosmological models (LambdaCDM, some CPL variations).
# 2. Define a set of representative redshifts z (e.g., logarithmic grid from 0.01 to 10).
# 3. For each model and z, compute the "true" integral J_true(z) using high-precision adaptive quadrature.
# 4. Define the objective function as the worst-case relative error over all (model, z) pairs.
# 5. Optimize nodes and weights to minimize this objective.

# Constraints:
# - nodes strictly increasing in (0, 1)
# - weights positive
# - sum weights = 1

# To ensure positivity and ordering, we can parameterize:
# nodes: x_i = sigmoid(y_i), where y_i are free parameters. Enforce y_0 < y_1 < ... < y_7.
# Or simply sort nodes and project.
# weights: w_i = exp(u_i) / sum(exp(u_j)).

# Let's use a simpler parameterization for optimization:
# Raw nodes r_i, sort them to get x_i. Add small epsilons to ensure strict increase.
# Raw weights v_i, normalize to get w_i.

# Objective function:
# For a given set of nodes/weights, compute sum w_i f_z(x_i).
# Compare to J_true(z).
# Error = |approx - true| / true.
# Maximize over all test cases.

# Since we don't have the actual validation set, we must guess a representative set.
# The task says "disjoint public validation cosmologies".
# We will use a diverse set of CPL models and a range of z.

def construct():
    start_time = time.time()
    
    # Representative models: (Omega_m, Omega_L, w0, wa)
    # LambdaCDM: w=-1 -> w0=-1, wa=0
    # CPL variations
    models = [
        (0.3, 0.7, -1.0, 0.0),  # LambdaCDM
        (0.3, 0.7, -0.5, 0.5),
        (0.3, 0.7, -1.5, 1.0),
        (0.25, 0.75, -0.8, 0.2),
        (0.35, 0.65, -1.2, -0.5),
        (0.3, 0.7, 0.0, 0.0),  # Matter-like dark energy (unphysical but tests robustness)
        (0.3, 0.7, -2.0, 2.0), # Strong evolution
    ]
    
    # Redshifts to test
    z_values = np.logspace(-2, 1, 50) # 0.01 to 10
    
    # Precompute true integrals for each model and z
    # J(z) = int_0^1 dx / E(1/(1+z*x))
    
    def compute_E(a, model):
        Om, Ol, w0, wa = model
        # w(a) = w0 + wa(1-a)
        w = w0 + wa * (1 - a)
        # E(a)^2 = Om a^-3 + Ol a^-3(1+w)
        # Note: for CPL, the integral of the DE density is not simply a^-3(1+w) unless w is constant.
        # For varying w, rho_DE(a) = rho_DE(1) * exp(int_1^a 3(1+w(a'))/a' da')
        # Let's approximate with constant w_eff or just use the standard CPL formula for E^2:
        # E(a)^2 = Om a^-3 + Ol * a^(-3) * exp(3 * wa * (1 - 1/a)) * a^(-3*wa) ? 
        # Actually, the standard CPL parameterization gives:
        # rho_DE(a) = rho_DE(0) a^{-3(1+w0)} exp(-3 wa (1 - 1/a))
        # So E(a)^2 = Om a^{-3} + Ol a^{-3(1+w0)} exp(-3 wa (1 - 1/a))
        
        if wa == 0:
            de_term = Ol * (a ** (-3 * (1 + w0)))
        else:
            de_term = Ol * (a ** (-3 * (1 + w0))) * np.exp(-3 * wa * (1 - 1/a))
        
        return np.sqrt(Om * a**(-3) + de_term)
    
    from scipy.integrate import quad
    
    true_integrals = []
    test_cases = [] # List of (model, z, true_value)
    
    for model in models:
        for z in z_values:
            def integrand(x):
                a = 1.0 / (1.0 + z * x)
                return 1.0 / compute_E(a, model)
            
            val, err = quad(integrand, 0, 1, limit=200)
            true_integrals.append(val)
            test_cases.append((model, z, val))
    
    true_integrals = np.array(true_integrals)
    
    # Optimization variables:
    # x: raw nodes (will be sorted)
    # w: raw weights (will be normalized)
    
    n = 8
    
    def objective(params):
        # First 8: nodes, next 8: weights
        raw_nodes = params[:n]
        raw_weights = params[n:]
        
        # Sort nodes
        nodes = np.sort(raw_nodes)
        
        # Ensure strictly increasing and in (0,1)
        # Add small margins
        nodes = nodes * 0.98 + 0.01
        
        # Enforce strict increase
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 1e-6
        
        # Ensure within [0,1]
        if nodes[0] <= 0:
            nodes[0] = 1e-6
        if nodes[-1] >= 1:
            nodes[-1] = 1 - 1e-6
            
        # Re-check strict increase after bounds
        for i in range(1, n):
            if nodes[i] <= nodes[i-1]:
                # Spread them out
                nodes[i] = nodes[i-1] + (1 - nodes[i-1]) * 1e-6

        # Normalize weights
        weights = np.exp(raw_weights) # Softplus for positivity
        weights = weights / np.sum(weights)
        
        # Compute approximations for all test cases
        max_error = 0.0
        
        # Vectorized computation? 
        # nodes shape (8,)
        # For each test case, we need f_z(nodes)
        
        for i_tc, (model, z, true_val) in enumerate(test_cases):
            # Compute f_z(x) for x in nodes
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = compute_E(a_vals, model)
            f_vals = 1.0 / E_vals
            
            approx_val = np.dot(weights, f_vals)
            
            rel_error = abs(approx_val - true_val) / true_val
            
            if rel_error > max_error:
                max_error = rel_error
                
        return max_error * 100 # Percentage error
    
    # Initial guess: Equally spaced nodes, equal weights
    x0_nodes = np.linspace(0.1, 0.9, n)
    x0_weights = np.zeros(n)
    x0 = np.concatenate([x0_nodes, x0_weights])
    
    # Optimization
    best_params = x0
    best_obj = objective(x0)
    
    # Try a few different starting points
    rng = np.random.default_rng(42)
    
    start_points = [x0]
    for _ in range(10):
        noise_nodes = rng.normal(0, 0.1, n)
        noise_weights = rng.normal(0, 0.1, n)
        sp = np.concatenate([x0_nodes + noise_nodes, x0_weights + noise_weights])
        start_points.append(sp)
    
    for sp in start_points:
        if time.time() - start_time > 70:
            break
        try:
            res = minimize(objective, sp, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
        
    # Extract final nodes and weights
    nodes = np.sort(best_params[:n])
    weights = np.exp(best_params[n:])
    weights = weights / np.sum(weights)
    
    # Final post-processing to ensure strict constraints
    nodes = nodes * 0.99 + 0.005 # Push into (0,1)
    for i in range(1, n):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-8
            
    if nodes[0] <= 0: nodes[0] = 1e-8
    if nodes[-1] >= 1: nodes[-1] = 1 - 1e-8
    
    # Check sum weights
    s = np.sum(weights)
    if s > 0:
        weights = weights / s
    else:
        weights = np.ones(n) / n
    weights = weights / np.sum(weights) # Ensure exact 1.0
    
    # Verify time
    if time.time() - start_time > 85:
        # Fallback to midpoint rule if timeout
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
        
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
