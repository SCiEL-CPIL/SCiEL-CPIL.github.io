import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

def construct():
    # Pre-compute reference data for a small set of cosmologies
    # Cosmologies: (Om0, w0, wa)
    cosms = [
        (0.3, -1.0, 0.0),   # LCDM
        (0.3, -0.9, 0.1),   # Mild
        (0.3, -1.0, 0.5),   # High wa
    ]
    
    # a values
    a_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Reference data storage
    ref_f = []
    ref_D_norm = []
    ref_params = [] # (Om_a, q, u, wa, a) for ansatz evaluation
    
    for Om0, w0, wa in cosms:
        # Define ODE for D(a)
        # x = ln a, D(x)
        # D'' + (3 + (1+w)) D' - 1.5 Om_m(x) D = 0
        # w(x) = w0 + wa(1 - exp(x))
        # Om_m(x) = Om0 / (Om0 + (1-Om0) * exp(-3(1+w0+wa)x))
        
        def ode_func(x, y):
            a = np.exp(x)
            w = w0 + wa * (1 - a)
            # Om_m
            # exp(-3(1+w0+wa)x)
            # Note: for large x, this might underflow, but x is ln(a), a in [0.1, 1], x in [-2.3, 0]
            exp_term = np.exp(-3 * (1 + w0 + wa) * x)
            Om_m = Om0 / (Om0 + (1 - Om0) * exp_term)
            
            D = y[0]
            D_prime = y[1]
            
            D_double_prime = - (3 + (1 + w)) * D_prime + 1.5 * Om_m * D
            return [D_prime, D_double_prime]
        
        # Integrate from x=0 (a=1) to x_min = ln(0.2)
        # Initial conditions at a=1: D(1)=1, D'(1)=f(1)
        # f(1) for LCDM is approx 0.55 * Om0^0.55
        # For general, f(1) ~ 0.55 + 0.05 w0 + 0.05 wa
        f_1 = 0.55 + 0.05 * w0 + 0.05 * wa
        y0 = [1.0, f_1]
        
        x_min = np.log(0.2)
        x_eval = np.log(a_vals)
        
        sol = solve_ivp(ode_func, [x_min, 0], y0, t_eval=x_eval, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            continue
            
        # sol.y[0] is D, sol.y[1] is D'
        # D is normalized to 1 at a=1 (x=0)
        # f = D'
        # We need D(a)/D(1). Since D(1)=1, D is already normalized.
        
        for i, a in enumerate(a_vals):
            x = x_eval[i]
            D_val = sol.y[0][i]
            f_val = sol.y[1][i]
            
            # Calculate Om_a, q, u for ansatz
            exp_term = np.exp(-3 * (1 + w0 + wa) * x)
            Om_a = Om0 / (Om0 + (1 - Om0) * exp_term)
            q = 1 - Om_a
            u = 1 + w0 + wa * (1 - a)
            
            ref_f.append(f_val)
            ref_D_norm.append(D_val)
            ref_params.append((Om_a, q, u, wa, a))
            
    # Flatten for vectorized loss
    ref_f = np.array(ref_f)
    ref_D_norm = np.array(ref_D_norm)
    Om_a_arr, q_arr, u_arr, wa_arr, a_arr = zip(*ref_params)
    Om_a_arr = np.array(Om_a_arr)
    q_arr = np.array(q_arr)
    u_arr = np.array(u_arr)
    wa_arr = np.array(wa_arr)
    a_arr = np.array(a_arr)
    
    def loss(x):
        c0, c1, c2, c3, c4, c5 = x
        # Calculate gamma
        gamma = c0 + c1 * q_arr + c2 * q_arr**2 + c3 * u_arr + c4 * wa_arr * a_arr + c5 * u_arr * q_arr
        
        # Check constraints
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e6
        if np.any(x < -10) or np.any(x > 10):
            return 1e6
            
        # Calculate f_approx
        f_approx = Om_a_arr ** gamma
        
        # Relative error for f
        # Avoid division by zero
        f_err = np.abs((f_approx - ref_f) / ref_f)
        
        # Calculate D_approx
        # D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
        # This integral is path dependent. We need to compute it for each a.
        # Since we have discrete points, we can approximate the integral using trapezoidal rule
        # on the f_approx values.
        # However, f_approx depends on x.
        # Let's compute the integral for each a_i.
        # x goes from ln(a_i) to 0.
        # We have points at x_eval.
        # We can use cumulative trapezoidal integration.
        
        # Sort x_eval in descending order (from 0 to x_min) for integration from 0 to x
        # x_eval is sorted ascending (ln(0.2) to 0).
        # We want integral from x to 0.
        # Let's compute cumulative integral from 0 backwards.
        
        # f_approx is defined at x_eval.
        # integral(x) = - \int_x^0 f(x') dx'
        # Let's compute cumulative sum of f * dx from 0 to x_min.
        # x_eval is [x_min, ..., 0].
        # We want I(x_i) = \int_{x_i}^0 f(x') dx'
        
        # Use np.trapz on segments.
        # I(0) = 0
        # I(x_1) = \int_{x_1}^0 f dx
        # We can compute this by integrating from 0 down to x_min.
        
        x_eval_desc = x_eval[::-1] # 0 to x_min
        f_approx_desc = f_approx[::-1]
        
        # Compute cumulative integral
        # dx = x[i] - x[i+1]
        # Int = 0.5 * (f[i] + f[i+1]) * dx
        # Let's just use np.trapz for each point? Too slow.
        # Cumulative trapezoid:
        # dI = 0.5 * (f[i] + f[i+1]) * (x[i] - x[i+1])
        
        dI = np.zeros_like(x_eval_desc)
        for i in range(len(x_eval_desc) - 1):
            dI[i] = 0.5 * (f_approx_desc[i] + f_approx_desc[i+1]) * (x_eval_desc[i] - x_eval_desc[i+1])
        # dI[i] is integral from x[i] to x[i+1]
        # We want integral from 0 to x[i].
        # I[0] = 0
        # I[1] = dI[0]
        # I[2] = dI[0] + dI[1]
        I_cum = np.cumsum(dI)
        # I_cum[i] is integral from 0 to x[i] (where x[i] is in desc order)
        # Wait, x_desc[0]=0, x_desc[1]=x1, etc.
        # dI[0] is int from 0 to x1.
        # I_cum[0] = 0? No, cumsum starts with dI[0].
        # Let's adjust:
        # I_at_x_desc_i = sum(dI[0:i])
        # I[0] = 0
        # I[1] = dI[0]
        # I[2] = dI[0] + dI[1]
        
        I_vals = np.zeros_like(x_eval_desc)
        I_vals[1:] = np.cumsum(dI[:-1])
        
        # I_vals[i] is integral from 0 to x_desc[i].
        # We want D(a) = exp(-I).
        # Note: x_desc is [0, x1, x2, ...].
        # I_vals is [0, I1, I2, ...].
        
        # Map back to original order (ascending x)
        I_vals_orig = I_vals[::-1]
        
        D_approx = np.exp(-I_vals_orig)
        
        # Relative error for D
        D_err = np.abs((D_approx - ref_D_norm) / ref_D_norm)
        
        # Max relative error
        max_err = np.max(np.concatenate([f_err, D_err]))
        return max_err

    # Initial guess
    x0 = np.array([0.55, 0.0, 0.0, 0.05, 0.05, 0.0])
    
    # Optimize
    res = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
    
    if res.success:
        return res.x.tolist()
    else:
        return x0.tolist()
