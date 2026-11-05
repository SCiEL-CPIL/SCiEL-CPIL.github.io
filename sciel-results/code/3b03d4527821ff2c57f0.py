import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
import time
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import chebgauss

def construct():
    t0 = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    OMEGA_M = 0.3
    OMEGA_DE = 0.7

    def E_of_z(z, w0, wa):
        """H(z)/H0 for CPL model"""
        z_safe = np.maximum(z, 0.0)
        w = w0 + wa * z_safe
        term_m = OMEGA_M * (1.0 + z_safe)**3
        term_de = OMEGA_DE * (1.0 + z_safe)**(3.0 * (1.0 + w))
        E2 = term_m + term_de
        return np.sqrt(np.maximum(E2, 1e-30))

    def reference_integral(z, w0, wa):
        """High accuracy reference for \int_0^z 1/E(z') dz'"""
        val, err = quad(lambda zp: 1.0 / E_of_z(zp, w0, wa), 0, z, limit=200)
        return val

    # Diverse training models covering the plausible validation space
    models = [
        (-1.0, 0.0),    # LambdaCDM
        (-0.5, 0.0),    # Quintessence
        (-1.5, 0.0),    # Phantom
        (-0.9, 0.3),    # CPL
        (-1.1, -0.2),   # CPL
        (0.0, 0.0),     # w=0
        (-0.7, 0.7),    # Strong variation
        (-1.3, 0.5),    # Another CPL
        (-0.6, -0.4),   # Another CPL
        (-1.0, 0.5),    # Lambda + variation
        (-0.8, -0.3),   # Mild variation
        (-1.2, 0.2),    # Phantom + variation
    ]
    
    z_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    # Precompute references
    refs = []
    z_flat = []
    models_flat = []
    for z in z_list:
        for w0, wa in models:
            refs.append(reference_integral(z, w0, wa))
            z_flat.append(z)
            models_flat.append((w0, wa))
    
    refs = np.array(refs)
    z_flat = np.array(z_flat)
    n_pairs = len(z_flat)
    
    def compute_max_error(nodes, weights):
        """Compute max relative error of the integral approximation."""
        max_err = 0.0
        for i in range(n_pairs):
            z_val = z_flat[i]
            w0, wa = models_flat[i]
            
            z_nodes_i = z_val * nodes
            E_vals = E_of_z(z_nodes_i, w0, wa)
            integral_approx = z_val * np.sum(weights / E_vals)
            
            ref_val = refs[i]
            
            if ref_val > 1e-10:
                rel_err = abs(integral_approx - ref_val) / abs(ref_val)
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    def objective(params):
        """Minimize max relative error. params: 8 node params + 8 weight params."""
        if time.time() > deadline - 5:
            return 1e10
            
        node_params = params[:8]
        weight_params = params[8:16]
        
        # Nodes: base grid + tanh perturbation, then enforce ordering
        base_nodes = np.linspace(0.1, 0.9, 8)
        nodes = base_nodes + 0.15 * np.tanh(node_params)
        nodes = np.clip(nodes, 0.001, 0.999)
        # Enforce strict ordering
        for i in range(1, 8):
            if nodes[i] <= nodes[i-1]:
                nodes[i] = nodes[i-1] + 0.001
        nodes = np.clip(nodes, 0.001, 0.999)
        
        # Weights: softmax to ensure positive, then normalize
        exp_w = np.exp(weight_params - np.max(weight_params))
        weights = exp_w / np.sum(exp_w)
        
        return compute_max_error(nodes, weights)

    # Initializations
    # 1. Gauss-Legendre mapped to [0,1]
    gl_nodes, gl_weights = leggauss(8)
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    base_nodes = np.linspace(0.1, 0.9, 8)
    x0_gl_nodes = np.arctanh(np.clip((gl_nodes - base_nodes) / 0.15, -0.999, 0.999))
    x0_gl_weights = np.log(gl_weights) - np.mean(np.log(gl_weights))
    x0_gl = np.concatenate([x0_gl_nodes, x0_gl_weights])
    
    # 2. Gauss-Chebyshev mapped to [0,1]
    ch_nodes, ch_weights = chebgauss(8)
    ch_nodes = 0.5 * (ch_nodes + 1.0)
    x0_ch_nodes = np.arctanh(np.clip((ch_nodes - base_nodes) / 0.15, -0.999, 0.999))
    x0_ch_weights = np.log(ch_weights) - np.mean(np.log(ch_weights))
    x0_ch = np.concatenate([x0_ch_nodes, x0_ch_weights])
    
    best_params = x0_gl.copy()
    best_err = objective(x0_gl)
    
    # Local refinement from GL
    if time.time() < deadline - 15:
        try:
            res_gl = minimize(
                objective, x0_gl, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_gl.fun < best_err:
                best_err = res_gl.fun
                best_params = res_gl.x
        except Exception:
            pass

    # Local refinement from Chebyshev
    if time.time() < deadline - 15:
        try:
            res_ch = minimize(
                objective, x0_ch, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_ch.fun < best_err:
                best_err = res_ch.fun
                best_params = res_ch.x
        except Exception:
            pass

    # Differential evolution if time permits
    if time.time() < deadline - 30:
        try:
            bounds = [(-5, 5)] * 16
            de_res = differential_evolution(
                objective, bounds, seed=42, maxiter=60, popsize=20,
                tol=1e-8, polish=False, init='sobol'
            )
            if de_res.fun < best_err:
                best_err = de_res.fun
                best_params = de_res.x
                
                # Refine DE result
                if time.time() < deadline - 5:
                    res_de = minimize(
                        objective, best_params, method='Nelder-Mead',
                        options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    if res_de.fun < best_err:
                        best_err = res_de.fun
                        best_params = res_de.x
        except Exception:
            pass

    # Final local refinement from best
    if time.time() < deadline - 3:
        try:
            res_final = minimize(
                objective, best_params, method='Powell',
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res_final.fun < best_err:
                best_err = res_final.fun
                best_params = res_final.x
        except Exception:
            pass

    # Extract final nodes and weights
    node_params = best_params[:8]
    weight_params = best_params[8:16]
    
    base_nodes = np.linspace(0.1, 0.9, 8)
    nodes = base_nodes + 0.15 * np.tanh(node_params)
    nodes = np.clip(nodes, 0.001, 0.999)
    for i in range(1, 8):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 0.001
    nodes = np.clip(nodes, 0.001, 0.999)
    
    exp_w = np.exp(weight_params - np.max(weight_params))
    weights = exp_w / np.sum(exp_w)
    
    # Final normalization check
    weights = weights / np.sum(weights)

    return {"nodes": nodes, "weights": weights}
