"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def construct():
    # Define a set of representative cosmologies
    # E(a) = H(a)/H0
    # Models: (name, w0, wa, w) for CPL: E^2 = Omega_m (1/a)^3 + Omega_k (1/a)^2 + Omega_de (1/a)^{3(1+w0+wa(1-1/a))}
    # For simplicity, assume flat universe Omega_m=0.3, Omega_de=0.7, Omega_k=0
    
    def E_CPL(a, w0, wa, Om=0.3):
        Ode = 1.0 - Om
        # a = 1/(1+z)
        # E^2 = Om a^-3 + Ode a^(-3(1+w0+wa(1-a)))
        exp_de = -3.0 * (1.0 + w0 + wa * (1.0 - a))
        return np.sqrt(Om * a**(-3.0) + Ode * a**exp_de)
    
    def E_wCDM(a, w, Om=0.3):
        Ode = 1.0 - Om
        exp_de = -3.0 * (1.0 + w)
        return np.sqrt(Om * a**(-3.0) + Ode * a**exp_de)
        
    def E_LambdaCDM(a, Om=0.3):
        Ode = 1.0 - Om
        return np.sqrt(Om * a**(-3.0) + Ode)

    models = [
        lambda a: E_LambdaCDM(a),
        lambda a: E_wCDM(a, -0.5),
        lambda a: E_wCDM(a, 0.0),
        lambda a: E_CPL(a, -1.0, 0.5),
        lambda a: E_CPL(a, -0.5, -0.5),
        lambda a: E_CPL(a, -1.5, 1.0),
    ]
    
    z_test = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
    
    def true_distance(z, model_func):
        # Integrate from 0 to z: (1+z') / E(z')
        # z' = (1/a') - 1 => a' = 1/(1+z')
        # dz' = -da'/a'^2
        # Integral = int_{a=1}^{a=1/(1+z)} (1/z') (1/a'^2) da' ... wait
        # dL = c/H0 int_0^z (1+z')/E(z') dz'
        # Let a' = 1/(1+z'). z' = 1/a' - 1. 1+z' = 1/a'.
        # dz' = -1/a'^2 da'.
        # Limits: z'=0 -> a'=1. z'=z -> a'=1/(1+z).
        # Integral = int_{1}^{1/(1+z)} (1/a') / E(a') * (-1/a'^2) da'
        # = int_{1/(1+z)}^{1} 1 / (a'^3 E(a')) da'
        
        def integrand(a):
            return 1.0 / (a**3 * model_func(a))
            
        val, err = quad(integrand, 1.0/(1.0+z), 1.0, limit=100)
        return val

    def quad_distance(z, model_func, nodes, weights):
        # Formula: z * (1+z) * sum_i weights[i] / E(a_i)
        # a_i = 1 / (1 + z * nodes[i])
        sum_term = 0.0
        for i in range(len(nodes)):
            a_i = 1.0 / (1.0 + z * nodes[i])
            sum_term += weights[i] / model_func(a_i)
        return z * (1.0 + z) * sum_term

    def objective(params):
        # params: [log_w1, ..., log_w8, x1, ..., x8]
        log_w = params[:8]
        x = params[8:]
        
        # Enforce ordering of nodes
        # x must be strictly increasing in [0,1]
        # We can just check and return high cost if violated
        if np.any(np.diff(x) <= 0) or np.any(x <= 0) or np.any(x >= 1):
            return 1e10
            
        w = np.exp(log_w)
        # Normalize weights to sum to 1? 
        # The prompt says weights sum to 1. 
        # It's better to optimize the raw weights and then normalize, 
        # but the formula depends on the scale.
        # If we normalize, we lose the ability to scale the whole thing.
        # However, the true integral for E=1 is z+z^2/2.
        # Approx is z(1+z) * Sum(w).
        # If Sum(w)=1, Approx = z+z^2.
        # This is fixed by the constraint.
        # So we should optimize w such that they sum to 1?
        # Or optimize log_w and then normalize?
        # If we normalize, the "scale" is fixed to 1.
        # Let's assume the optimal scale is indeed 1 for the base case.
        w_norm = w / np.sum(w)
        
        max_err = 0.0
        for model in models:
            for z in z_test:
                d_true = true_distance(z, model)
                d_quad = quad_distance(z, model, x, w_norm)
                if d_true > 1e-10:
                    err = abs(d_quad - d_true) / d_true
                    if err > max_err:
                        max_err = err
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    from numpy.polynomial.legendre import leggauss
    x_gl, w_gl = leggauss(8)
    x0_nodes = (x_gl + 1.0) / 2.0
    w0_weights = w_gl / 2.0
    
    # Params: log(w), x
    p0 = np.concatenate([np.log(w0_weights), x0_nodes])
    
    # Optimize
    res = minimize(objective, p0, method='Nelder-Mead', 
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    p_opt = res.x
    log_w_opt = p_opt[:8]
    x_opt = p_opt[8:]
    
    w_opt = np.exp(log_w_opt)
    w_opt = w_opt / np.sum(w_opt)
    
    # Ensure strict increase
    # Nelder-Mead might produce non-strictly increasing nodes if constraints are not hard
    # We added a penalty, but let's enforce it
    # If not strictly increasing, fallback to Gauss-Legendre
    if not np.all(np.diff(x_opt) > 0) or not np.all(x_opt > 0) or not np.all(x_opt < 1):
        print("Optimization failed to maintain constraints, falling back to GL")
        return {"nodes": x0_nodes, "weights": w0_weights}
        
    return {"nodes": x_opt, "weights": w_opt}

# EVOLVE-BLOCK-END
