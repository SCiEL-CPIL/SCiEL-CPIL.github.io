import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Strategy:
    # Represent h as the real part of a complex exponential sequence with non-linear phase.
    # h[k] = Re( exp(i * phi(k)) )
    # phi(k) = a * k^2 + b * k  (quadratic phase for chirp-like structure)
    # The autoconvolution of a chirp is related to the Fourier transform of the phase function.
    # Specifically, if h[k] = exp(i * a * k^2), the autoconvolution c[m] has a peak at m=0 
    # whose magnitude is related to 1/sqrt(a). 
    # By choosing 'a' appropriately, we can control the peak height.
    # We also allow a linear phase shift 'b' and a DC offset 'c' (added to h) to adjust sum(h).
    # Actually, adding a constant to h changes the structure significantly.
    # Let's stick to h[k] = A * Re( exp(i * (a*k^2 + b*k)) ) + C.
    # We optimize A, a, b, C.
    
    # Parameters:
    # p = [A, a, b, C]
    # h[k] = A * cos(a*k^2 + b*k) + C
    
    # Objective: Minimize value = abs(2 * N * max(c) / sum(h)^2)
    # We use a smooth approximation of max(c) to allow gradient-based optimization.
    # max(c) ~ log-sum-exp(c / tau) * tau
    
    def compute_h(params):
        A, a, b, C = params
        k = np.arange(N)
        phase = a * k**2 + b * k
        h = A * np.cos(phase) + C
        return h

    def smooth_objective(params):
        h = compute_h(params)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        c = np.convolve(h, h)
        # Normalize c for numerical stability in log-sum-exp
        c_max = np.max(c)
        c_shifted = c - c_max
        # Temperature for smooth max
        tau = 0.1 
        lse = c_max + tau * np.log(np.sum(np.exp(c_shifted / tau)))
        # The objective is abs(2 * N * max(c) / s^2)
        # Using lse as proxy for max(c)
        val = 2 * N * abs(lse) / (s * s)
        return val

    def smooth_grad(params):
        h = compute_h(params)
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros(4)
        c = np.convolve(h, h)
        c_max = np.max(c)
        c_shifted = c - c_max
        tau = 0.1
        exp_terms = np.exp(c_shifted / tau)
        sum_exp = np.sum(exp_terms)
        
        # Gradient of lse w.r.t h
        # d lse / d h_i = sum_m (d lse / d c_m) * (d c_m / d h_i)
        # d lse / d c_m = exp(c_m/tau) / sum(exp(c/tau)) = exp_terms[m] / sum_exp
        # Note: c_max is constant w.r.t h for the derivative of the log-sum-exp part, 
        # but c_max itself depends on h. 
        # However, the derivative of max(x) is the indicator of the max element.
        # The derivative of log-sum-exp(x/tau)*tau is sum(x_j/tau * exp(x_j/tau) / sum(exp(x/tau))).
        # Let w_m = exp(c_m/tau) / sum_exp.
        # Then d lse / d c_m = w_m.
        
        weights = exp_terms / sum_exp
        
        # Now we need grad_h = sum_m weights[m] * (d c_m / d h)
        # d c_m / d h_i = h[m-i] + h[i] (with bounds)
        # This is similar to the previous gradient derivation.
        # sum_m weights[m] * (h[m-i] + h[i])
        # = sum_m weights[m] h[m-i] + h[i] sum_m weights[m]
        # sum_m weights[m] = 1.
        # So term 2 is h[i].
        # Term 1 is cross-correlation of weights and h.
        
        # weights is length 2N-1, h is length N.
        # corr = np.correlate(weights, h, mode='full')
        # We need sum_m weights[m] h[m-i].
        # This is the cross-correlation at lag i.
        # In numpy correlate(a, v), out[k] = sum_m a[m] v[m-k].
        # We want k=i.
        # Index in out = i + len(v) - 1 = i + N - 1.
        
        corr = np.correlate(weights, h, mode='full')
        R_wh = corr[N-1 : N-1+N]
        
        grad_h = R_wh + h
        
        # Now chain rule: d val / d params = d val / d h * d h / d params
        # val = 2 * N * lse / s^2
        # d val / d lse = 2 * N / s^2
        # d val / d s = -2 * 2 * N * lse / s^3 = -4 * N * lse / s^3
        
        # d lse / d params = grad_h @ (d h / d params)
        # d s / d params = sum(d h / d params)
        
        # Compute dh/dparams for p = [A, a, b, C]
        # h = A * cos(phi) + C
        # phi = a*k^2 + b*k
        # dh/dA = cos(phi)
        # dh/db = -A * sin(phi) * k
        # dh/db = -A * sin(phi)
        # dh/dC = 1
        
        k = np.arange(N)
        phi = a * k**2 + b * k
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        dh_dA = cos_phi
        dh_da = -A * sin_phi * k**2
        dh_db = -A * sin_phi * k
        dh_dC = np.ones(N)
        
        dh_dparams = np.column_stack((dh_dA, dh_da, dh_db, dh_dC))
        
        # d val / d params_i = (2*N/s^2) * sum_j (grad_h[j] * dh_dparams[j, i]) + (-4*N*lse/s^3) * sum_j (dh_dparams[j, i])
        
        lse_val = c_max + tau * np.log(sum_exp)
        
        coeff1 = 2 * N / (s * s)
        coeff2 = -4 * N * lse_val / (s * s * s)
        
        grad_params = np.zeros(4)
        for i in range(4):
            d_h = dh_dparams[:, i]
            term1 = np.dot(grad_h, d_h)
            term2 = np.sum(d_h)
            grad_params[i] = coeff1 * term1 + coeff2 * term2
            
        return grad_params

    # Initial guesses
    # We want to explore the parameter space.
    # a controls the "chirp rate". 
    # If a is small, it's close to constant (seed).
    # If a is large, it oscillates rapidly, potentially flattening the autoconvolution.
    
    starts = [
        [1.0, 0.0, 0.0, 0.0], # Seed-like
        [1.0, 0.1, 0.0, 0.0],
        [1.0, -0.1, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0],
        [1.0, -0.5, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.5, 0.1, 0.1, 0.5],
        [0.5, -0.1, -0.1, 0.5],
    ]
    
    best_params = starts[0]
    best_val = 1e10
    
    # Evaluate seed exactly to set baseline
    h_seed = np.ones(N) / N
    c_seed = np.convolve(h_seed, h_seed)
    val_seed = abs(2 * N * np.max(c_seed) / np.sum(h_seed)**2)
    best_val = val_seed
    best_h = h_seed
    
    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                smooth_objective,
                start,
                jac=smooth_grad,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            
            h_opt = compute_h(res.x)
            s = np.sum(h_opt)
            if abs(s) > 1e-10:
                c_opt = np.convolve(h_opt, h_opt)
                val_opt = abs(2 * N * np.max(c_opt) / (s * s))
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt
                    best_params = res.x
        except Exception:
            pass

    # If the optimized result is worse than seed, return seed
    if best_val > val_seed:
        return h_seed
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(len(h))
    print(np.sum(h))
    z = np.convolve(h, h)
    print(abs(2 * N * np.max(z) / np.sum(h)**2))
