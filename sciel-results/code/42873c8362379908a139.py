import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety

    # Define cosmological models for training
    # (Omega_m, Omega_r, w0, wa)
    models = [
        (0.3, 0.0, -1.0, 0.0),   # Lambda CDM
        (0.3, 0.0, -1.2, 0.0),   # Phantom
        (0.3, 0.0, -0.8, 0.0),   # Quintessence
        (0.3, 0.0, -1.0, 0.5),   # CPL
        (0.4, 0.0, -1.0, 0.0),   # High Omega_m
        (0.2, 0.0, -1.0, 0.0),   # Low Omega_m
        (0.3, 0.05, -1.0, 0.0),  # With Radiation
    ]
    
    # Redshifts to test
    z_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Pre-compute reference values
    # Reference integral: I_ref = integral_0^1 1/E(a(x)) dx
    # Then dL_norm = (1+z)*z*I_ref
    # We minimize relative error of dL_norm, which is same as relative error of I_ref
    # because (1+z)*z is exact.
    
    def E_cpl(a, om_m, om_r, w0, wa):
        a = np.maximum(a, 1e-10)
        h2 = om_m * a**(-3) + om_r * a**(-4) + (1 - om_m - om_r) * a**(-3*(1 + w0 + wa))
        return np.sqrt(np.maximum(h2, 1e-30))
        
    def integrand(x, z, om_m, om_r, w0, wa):
        a = 1.0 / (1.0 + z * x)
        return 1.0 / E_cpl(a, om_m, om_r, w0, wa)
        
    ref_integrals = []
    for om_m, om_r, w0, wa in models:
        for z in z_vals:
            # High precision integration
            val, err = quad(lambda x: integrand(x, z, om_m, om_r, w0, wa), 0, 1, limit=200)
            ref_integrals.append(val)
            
    ref_integrals = np.array(ref_integrals)
    
    # Mapping from unconstrained parameters to nodes and weights
    def params_to_nodes_weights(params):
        u = params[:8]
        v = params[8:]
        
        # Nodes: cumulative sigmoid
        x = np.zeros(8)
        x[0] = 1.0 / (1.0 + np.exp(-u[0]))
        for i in range(1, 8):
            # Increment in (0, 1 - x[i-1])
            inc = (1.0 / (1.0 + np.exp(-u[i]))) * (1.0 - x[i-1])
            x[i] = x[i-1] + inc
        # Ensure strictly in (0,1)
        x = np.clip(x, 1e-9, 1.0 - 1e-9)
        
        # Weights: softmax
        w = np.exp(v - np.max(v))
        w = w / np.sum(w)
        
        return x, w
        
    def objective(params):
        x, w = params_to_nodes_weights(params)
        
        max_err = 0.0
        idx = 0
        for om_m, om_r, w0, wa in models:
            for z in z_vals:
                a = 1.0 / (1.0 + z * x)
                e = E_cpl(a, om_m, om_r, w0, wa)
                approx_int = np.sum(w / e)
                
                ref_val = ref_integrals[idx]
                idx += 1
                
                if ref_val == 0:
                    err = abs(approx_int)
                else:
                    err = abs(approx_int - ref_val) / abs(ref_val)
                    
                if err > max_err:
                    max_err = err
                    
        return max_err
        
    # Initial guess: Gauss-Legendre nodes and weights on [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # Map [-1,1] to [0,1]
    x_gl = 0.5 * x_gl + 0.5
    w_gl = 0.5 * w_gl
    
    # Invert the parameterization to get initial u, v
    # This is tricky. Instead, we can just start with random u, v 
    # or find u, v that produce x_gl.
    # Simpler: Start with u=0, v=0 (uniform nodes/weights) and let optimizer work.
    # Or better: Use the GL nodes directly in the objective if we bypass the constraint?
    # No, we must use the constrained parameterization for the optimizer.
    # Let's just start with u=0, v=0. The optimizer will move it.
    # Or we can solve for u, v approximately.
    # For u: x[0] = sig(u0) => u0 = log(x0/(1-x0))
    # x[i] = x[i-1] + sig(ui)*(1-x[i-1]) => sig(ui) = (x[i]-x[i-1])/(1-x[i-1])
    # ui = log( sig / (1-sig) )
    
    u_init = np.zeros(8)
    u_init[0] = np.log(x_gl[0] / (1 - x_gl[0]))
    for i in range(1, 8):
        sig = (x_gl[i] - x_gl[i-1]) / (1 - x_gl[i-1])
        sig = np.clip(sig, 1e-9, 1-1e-9)
        u_init[i] = np.log(sig / (1 - sig))
        
    # For v: w_i = exp(v_i) / sum(exp(v_j))
    # We can set v_i = log(w_i)
    v_init = np.log(w_gl)
    
    x0 = np.concatenate([u_init, v_init])
    
    best_params = x0
    best_score = objective(x0)
    
    # Optimization
    # Nelder-Mead is robust for this
    options = {
        'maxiter': 2000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'disp': False
    }
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception as e:
        pass
        
    # If time allows, try a second run with different start or Basin Hopping?
    # Let's just do a second Nelder-Mead run with slightly perturbed start if time remains
    if time.time() < deadline - 10:
        x0_2 = x0 + np.random.normal(0, 0.1, 16)
        try:
            res2 = minimize(objective, x0_2, method='Nelder-Mead', options=options)
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception as e:
            pass
            
    # Extract final nodes and weights
    nodes, weights = params_to_nodes_weights(best_params)
    
    # Ensure weights sum to 1 exactly (floating point)
    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
