import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    # Representative cosmologies
    # E(a) = H(a)/H0
    # For flat universe: E(a) = sqrt(Omega_m * a^-3 + Omega_DE * a^(-3(1+w)) )
    # For CPL: w(a) = w0 + wa*(1-a)
    # E(a) = sqrt(Omega_m * a^-3 + Omega_DE * exp(3*wa*(1-a)) * a^(-3*(1+w0)))
    # Note: The integral is over z. a = 1/(1+z).
    # E(z) = sqrt(Omega_m * (1+z)^3 + Omega_DE * (1+z)^(3*(1+w0)) * exp(3*wa*z/(1+z)))
    # Wait, the standard CPL form is:
    # E(z) = (1+z)^(3/2(1+w0)) * (1 + w_a * (1 - 1/(1+z)))^(3/2 w_a) * sqrt(Omega_m + Omega_DE * (1+z)^(3(1+w0)) * exp(-3 w_a (1 - 1/(1+z))))?
    # Let's stick to the simpler form:
    # H(z) = H0 * (Omega_m (1+z)^3 + Omega_DE (1+z)^(3(1+w0)) exp(3 w_a (1 - 1/(1+z))))^0.5
    # This is a common approximation.
    
    Omega_m = 0.3
    Omega_DE = 0.7

    def E_z(z, w0, wa):
        # z is array or scalar
        a = 1.0 / (1.0 + z)
        term_m = Omega_m * a**(-3)
        term_de = Omega_DE * a**(-3*(1+w0)) * np.exp(3*wa*(1-a))
        return np.sqrt(term_m + term_de)

    # Models: (w0, wa)
    models = [
        (-1.0, 0.0),   # Lambda CDM
        (-0.5, 0.5),   # Quintessence-like
        (-1.5, 1.0),   # Phantom-like
        (-0.9, -0.1)   # Mild variation
    ]

    # Test redshifts
    z_tests = [0.2, 0.5, 1.0, 1.5, 2.0]

    def objective(params):
        # Unpack params
        v = params[:8]
        y = params[8:]
        
        # Map v to nodes u in (0, 1) sorted
        exp_v = np.exp(v)
        cumsum_v = np.cumsum(exp_v)
        u = cumsum_v / cumsum_v[-1]
        
        # Map y to weights w, sum to 1
        exp_y = np.exp(y)
        w = exp_y / exp_y.sum()
        
        max_err = 0.0
        
        for w0, wa in models:
            for z in z_tests:
                # True integral: (1+z)*z * int_0^1 du / E(z*u)
                # Integrand for quad: f(u) = 1 / E(z*u)
                def integrand(u):
                    return 1.0 / E_z(z * u, w0, wa)
                
                # Quad integration
                val_true, _ = quad(integrand, 0, 1)
                dL_true = (1 + z) * z * val_true
                
                # Quadrature approximation
                # Nodes in u-space are u.
                # The argument to E is z * u_i
                u_nodes = u
                z_u = z * u_nodes
                E_vals = E_z(z_u, w0, wa)
                quad_sum = np.sum(w * (1.0 / E_vals))
                
                dL_quad = (1 + z) * z * quad_sum
                
                rel_err = abs(dL_quad - dL_true) / dL_true
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # Initial guess: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    u_init = (nodes_gl + 1) / 2
    
    # Invert the mapping to get v and y
    # u_i = sum_{k=1}^i exp(v_k) / sum_{k=1}^8 exp(v_k)
    # This is equivalent to softmax of v? No, it's cumulative softmax.
    # Let p_k = exp(v_k). u_i = (p_1+...+p_i) / (p_1+...+p_8)
    # p_i = p_1 * exp(v_i - v_1)
    # This is hard to invert analytically easily.
    # Instead, just start with v=0, y=0 which gives uniform nodes and weights.
    # Or, we can use the Gauss-Legendre nodes to initialize v and y approximately.
    # For now, uniform is a safe start, or we can use random restarts.
    # Let's try to map GL nodes to v.
    # If u_i are given, we can solve for p_i.
    # p_i / P = u_i - u_{i-1} (with u_0=0, u_8=1)
    # Let d_i = u_i - u_{i-1}. d_i = p_i / P.
    # p_i = P * d_i.
    # v_i = ln(p_i) = ln(P) + ln(d_i).
    # We can set v_i = ln(d_i). The constant ln(P) cancels in softmax-like structures?
    # Wait, u = cumsum(exp(v)) / sum(exp(v)).
    # If v_i = ln(d_i), then exp(v_i) = d_i.
    # cumsum(d_i) = u_i. sum(d_i) = 1.
    # So u = cumsum(d_i) / 1 = u_i.
    # So v_i = ln(u_i - u_{i-1}) works!
    
    d_u = np.diff(np.concatenate(([0], u_init, [1])))
    v_init = np.log(d_u)
    
    # For weights, w_i = exp(y_i) / sum(exp(y_i)).
    # If we want w_i = weights_gl, then exp(y_i) proportional to weights_gl.
    # y_i = ln(weights_gl).
    y_init = np.log(weights_gl)
    
    x0 = np.concatenate((v_init, y_init))
    
    # Optimize
    start_time = time.time()
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
    
    # Check time
    if time.time() - start_time > 50:
        # Fallback to GL if optimization is too slow (unlikely with 200 iters)
        return {"nodes": u_init, "weights": weights_gl}

    # Extract optimal params
    v_opt = res.x[:8]
    y_opt = res.x[8:]
    
    exp_v = np.exp(v_opt)
    cumsum_v = np.cumsum(exp_v)
    nodes_opt = cumsum_v / cumsum_v[-1]
    
    exp_y = np.exp(y_opt)
    weights_opt = exp_y / exp_y.sum()
    
    # Ensure strict increasing (should be by construction)
    # Ensure positive weights (should be by construction)
    
    return {"nodes": nodes_opt, "weights": weights_opt}
