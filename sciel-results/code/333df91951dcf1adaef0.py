"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 8
    
    # --- Cosmology models for training ---
    # E(a) = H(a)/H0 for spatially flat universe
    # We'll use a variety of CPL-like models and other parameterizations
    
    def E_lcdm(a, w0=-1.0, wa=0.0):
        """CPL model: E(a) = a^(-3(1+w0+wa)) * ((1+z)^(3wa))^(-1) ... 
        Actually for CPL: H(z)^2/H0^2 = (1+z)^3 * ((1+z)^(3wa) * (1/(1+w0)) * ((1+z)^(1+w0) - 1)) / (1/(1+w0))
        More standard: E(z)^2 = (1+z)^3 * exp(-3wa * z/(1+z)) for w0=-1
        Let's use the general CPL form:
        E(z)^2 = (1+z)^3 * ((1+z)^(3wa) * (1/(1+w0)) * ((1+z)^(1+w0) - 1)) / (1/(1+w0))
        But simpler: for w(z) = w0 + wa*z
        E(z)^2 = (1+z)^3 * exp(-3*wa*z/(1+z)) * (1+w0)^(1/(1+w0)) ... 
        Let me just use: E(z)^2 = (1+z)^3 * (1+z)^(3wa*z/(1+z)) is wrong.
        
        Standard CPL: E(z)^2 = (1+z)^3 * exp(3wa * z/(1+z)) is not quite right either.
        
        Let's use: for w(z) = w0 + wa*z, the scale factor relation gives:
        E(a)^2 = a^{-3(1+w0+wa)} * (a^{1+w0} - 1)/(1+w0) * a^{3wa}
        where a = 1/(1+z)
        
        Actually the cleanest form:
        E(z)^2 = (1+z)^3 * ((1+z)^(3wa) * (1/(1+w0)) * ((1+z)^(1+w0) - 1)) / (1/(1+w0))
        = (1+z)^3 * (1+z)^(3wa) * ((1+z)^(1+w0) - 1)/(1+w0)
        
        For w0=-1, wa=0: E(z)^2 = (1+z)^3 * 1 * ((1+z)^0 - 1)/0 -> limit = (1+z)^3
        That's LCDM. Good.
        """
        if abs(w0 + 1.0) < 1e-10:
            # w0 = -1 case
            return np.sqrt((1.0 + z)**3 * np.exp(3.0 * wa * z / (1.0 + z)))
        else:
            term = ((1.0 + z)**(1.0 + w0) - 1.0) / (1.0 + w0)
            return np.sqrt((1.0 + z)**3 * (1.0 + z)**(3.0 * wa) * term)
    
    def E_w0wa(a, w0, wa):
        """E(a) using scale factor a = 1/(1+z), so z = 1/a - 1"""
        z = 1.0 / a - 1.0
        return E_lcdm(z, w0, wa)
    
    # Training cosmologies (w0, wa) pairs
    training_models = [
        (-1.0, 0.0),    # LCDM
        (-0.9, 0.0),
        (-0.8, 0.0),
        (-1.0, 0.5),
        (-1.0, -0.5),
        (-0.9, 0.3),
        (-0.7, -0.3),
        (-1.1, 0.2),
        (-0.5, 0.0),
        (-1.0, 1.0),
        (-1.2, 0.5),
        (-0.8, 0.5),
        (-0.9, -0.5),
        (-1.0, 0.8),
        (-0.6, 0.4),
    ]
    
    # z range for training: [0, 3]
    z_train = np.linspace(0.001, 3.0, 50)
    
    def exact_dL(z, w0, wa):
        """Compute exact dL/(c/H0) via adaptive quadrature"""
        def integrand(z_val):
            a_val = 1.0 / (1.0 + z_val)
            return (1.0 + z_val) * z_val / E_lcdm(z_val, w0, wa)
        result, _ = quad(integrand, 0, z, limit=200)
        return result
    
    def quadrature_dL(z, nodes, weights, w0, wa):
        """Compute dL/(c/H0) using the quadrature rule"""
        a_i = 1.0 / (1.0 + z * nodes)
        total = 0.0
        for i in range(n):
            total += weights[i] / E_w0wa(a_i, w0, wa)
        return z * total
    
    def objective(nodes):
        """Objective: maximize worst-case relative error (we minimize this)"""
        if time.time() > deadline - 5.0:
            return 1e10  # penalty for timeout
        
        # Ensure nodes are strictly increasing in (0,1)
        nodes = np.sort(nodes)
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e10
        for i in range(n - 1):
            if nodes[i+1] - nodes[i] < 1e-10:
                return 1e10
        
        # Solve for optimal positive weights via LP
        # We want to minimize max over all training points of |quad - exact|/exact
        # First, compute the matrix A where A[z_idx, i] = 1/E(a_i) for model m
        # Then dL_quad = z * sum_i w_i * A[m, z_idx, i]
        
        # For each (model, z) pair, we have:
        # exact = f(m, z)
        # quad = z * sum_i w_i * g(m, z, i)
        # relative error = |quad - exact| / exact
        
        # This is a linear programming problem:
        # minimize t
        # subject to: for each (m, z): -t * exact <= z * sum_i w_i * g(m,z,i) - exact <= t * exact
        #              sum_i w_i = 1
        #              w_i >= eps (small positive)
        
        from scipy.optimize import linprog
        
        # Build the constraint matrix
        constraints = []
        for w0, wa in training_models:
            for z_val in z_train:
                a_i = 1.0 / (1.0 + z_val * nodes)
                g_vals = np.array([1.0 / E_w0wa(a_i[j], w0, wa) for j in range(n)])
                exact_val = exact_dL(z_val, w0, wa)
                if exact_val < 1e-10:
                    continue
                # z * sum_i w_i * g_i - exact = z * dot(w, g) - exact
                # We want: -t*exact <= z*dot(w,g) - exact <= t*exact
                # => z*dot(w,g) - t*exact <= exact
                # => -z*dot(w,g) - t*exact <= -exact
                row1 = np.zeros(n + 1)
                row1[:n] = z_val * g_vals
                row1[n] = -exact_val
                constraints.append((row1, exact_val))
                
                row2 = np.zeros(n + 1)
                row2[:n] = -z_val * g_vals
                row2[n] = -exact_val
                constraints.append((row2, -exact_val))
        
        A_ub = np.array([c[0] for c in constraints])
        b_ub = np.array([c[1] for c in constraints])
        
        # Equality constraint: sum w_i = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        # Objective: minimize t (last variable)
        c_obj = np.zeros(n + 1)
        c_obj[n] = 1.0
        
        # Bounds: w_i >= 0 (we'll use small epsilon to ensure positivity), t >= 0
        bounds = [(1e-10, None)] * n + [(0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                          bounds=bounds, method='highs')
            if not res.success:
                return 1e10
            weights = res.x[:n]
            t_val = res.x[n]
            return t_val * 100.0  # convert to percent
        except:
            return 1e10
    
    # Differential evolution for node optimization
    bounds_de = [(0.01, 0.99)] * n
    
    best_nodes = None
    best_weights = None
    best_score = float('inf')
    
    # First try: use differential evolution
    try:
        result_de = differential_evolution(
            objective, bounds_de, 
            maxiter=50, popsize=15, 
            tol=1e-6, seed=42,
            polish=True,
            init='sobol'
        )
        if time.time() < deadline - 10.0:
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_nodes = np.sort(result_de.x)
                # Re-solve for weights
                from scipy.optimize import linprog
                constraints = []
                for w0, wa in training_models:
                    for z_val in z_train:
                        a_i = 1.0 / (1.0 + z_val * best_nodes)
                        g_vals = np.array([1.0 / E_w0wa(a_i[j], w0, wa) for j in range(n)])
                        exact_val = exact_dL(z_val, w0, wa)
                        if exact_val < 1e-10:
                            continue
                        row1 = np.zeros(n + 1)
                        row1[:n] = z_val * g_vals
                        row1[n] = -exact_val
                        constraints.append((row1, exact_val))
                        row2 = np.zeros(n + 1)
                        row2[:n] = -z_val * g_vals
                        row2[n] = -exact_val
                        constraints.append((row2, -exact_val))
                A_ub = np.array([c[0] for c in constraints])
                b_ub = np.array([c[1] for c in constraints])
                A_eq = np.zeros((1, n + 1))
                A_eq[0, :n] = 1.0
                b_eq = np.array([1.0])
                c_obj = np.zeros(n + 1)
                c_obj[n] = 1.0
                bounds_lp = [(1e-10, None)] * n + [(0, None)]
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                              bounds=bounds_lp, method='highs')
                if res.success:
                    best_weights = res.x[:n]
    except Exception as e:
        pass
    
    # If we have time, try local refinement with Nelder-Mead
    if best_nodes is not None and time.time() < deadline - 15.0:
        try:
            def obj_local(nodes):
                if time.time() > deadline - 5.0:
                    return 1e10
                return objective(nodes)
            
            result_nm = minimize(
                obj_local, best_nodes, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_nodes = np.sort(result_nm.x)
                from scipy.optimize import linprog
                constraints = []
                for w0, wa in training_models:
                    for z_val in z_train:
                        a_i = 1.0 / (1.0 + z_val * best_nodes)
                        g_vals = np.array([1.0 / E_w0wa(a_i[j], w0, wa) for j in range(n)])
                        exact_val = exact_dL(z_val, w0, wa)
                        if exact_val < 1e-10:
                            continue
                        row1 = np.zeros(n + 1)
                        row1[:n] = z_val * g_vals
                        row1[n] = -exact_val
                        constraints.append((row1, exact_val))
                        row2 = np.zeros(n + 1)
                        row2[:n] = -z_val * g_vals
                        row2[n] = -exact_val
                        constraints.append((row2, -exact_val))
                A_ub = np.array([c[0] for c in constraints])
                b_ub = np.array([c[1] for c in constraints])
                A_eq = np.zeros((1, n + 1))
                A_eq[0, :n] = 1.0
                b_eq = np.array([1.0])
                c_obj = np.zeros(n + 1)
                c_obj[n] = 1.0
                bounds_lp = [(1e-10, None)] * n + [(0, None)]
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                              bounds=bounds_lp, method='highs')
                if res.success:
                    best_weights = res.x[:n]
        except:
            pass
    
    # Fallback: use a good initial guess if optimization failed
    if best_nodes is None:
        best_nodes = np.array([0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375])
        best_weights = np.ones(n) / n
    
    # Ensure constraints are met
    best_nodes = np.sort(best_nodes)
    # Clamp to (0,1)
    best_nodes[0] = max(best_nodes[0], 1e-6)
    best_nodes[-1] = min(best_nodes[-1], 1 - 1e-6)
    # Ensure strictly increasing
    for i in range(n - 1):
        if best_nodes[i+1] - best_nodes[i] < 1e-8:
            best_nodes[i+1] = best_nodes[i] + 1e-8
            best_nodes[i+1] = min(best_nodes[i+1], 1 - 1e-6)
    
    # Ensure weights are positive and sum to 1
    best_weights = np.abs(best_weights)
    if best_weights.sum() < 1e-15:
        best_weights = np.ones(n) / n
    best_weights = best_weights / best_weights.sum()
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
