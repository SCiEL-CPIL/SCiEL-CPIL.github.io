from scipy.optimize import minimize
import numpy as np
import time
# EVOLVE-BLOCK-START
def construct():
    return list(_best_coeffs())

def _best_coeffs():
    t0 = time.time()
    deadline = t0 + 80.0          # Explicit safety margin under the 90s kill
    c = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    try:
        c = _fit(deadline)
    except Exception:
        pass
    return c

def _fit(deadline):
    # ---- Reference grid ----
    om0_vals = np.array([0.22, 0.25, 0.28, 0.30, 0.35])
    w0_vals  = np.array([-1.0, -0.5, 0.0, 0.5])
    wa_vals  = np.array([-0.5, 0.0, 0.5])
    a_vals   = np.linspace(0.1, 1.0, 50)
    
    # Flatten grid for vectorization where possible, but ODE is per-model
    models = []
    for o in om0_vals:
        for w in w0_vals:
            for wa in wa_vals:
                models.append((o, w, wa))
                
    M = len(models)
    N = len(a_vals)
    
    # Precompute reference D(a) and f(a) using solve_ivp
    # System: y1 = D, y2 = D'
    # y1' = y2
    # y2' = -1.5/a * y2 + 1.5 * Om_a(a)/a^2 * y1
    
    ref_D = np.zeros((M, N))
    ref_f = np.zeros((M, N))
    
    # We will store the specific coefficients for each model to use in the objective
    # Om0_g, W0_g, Wa_g, A_g are needed for the ansatz evaluation
    Om0_g = np.zeros(M * N)
    W0_g = np.zeros(M * N)
    Wa_g = np.zeros(M * N)
    A_g = np.zeros(M * N)
    
    idx = 0
    for m_idx, (o, w, wa) in enumerate(models):
        # Fill grid arrays
        for n_idx, a in enumerate(a_vals):
            Om0_g[idx] = o
            W0_g[idx] = w
            Wa_g[idx] = wa
            A_g[idx] = a
            idx += 1
            
        # Solve ODE for this model
        def ode_func(a, y):
            D, Dp = y
            u = 1.0 + w + wa * (1.0 - a)
            # Om_a = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(3u))
            # Handle a=0? a starts at 0.1.
            denom = o * a**3 + (1.0 - o) * a**(3.0 * u)
            if denom == 0:
                Om_a = 0.0
            else:
                Om_a = (o * a**3) / denom
            
            dD = Dp
            dDp = -1.5 / a * Dp + 1.5 * Om_a / (a * a) * D
            return [dD, dDp]
        
        # Initial conditions at a=0.1
        # Matter dominated: D ~ a, D' ~ 1
        y0 = [0.1, 1.0]
        
        try:
            sol = solve_ivp(ode_func, [0.1, 1.0], y0, t_eval=a_vals, method='RK45', rtol=1e-8, atol=1e-10)
            if sol.success:
                D_vals = sol.y[0]
                Dp_vals = sol.y[1]
                # Normalize D(1) = 1
                norm = D_vals[-1]
                D_vals = D_vals / norm
                Dp_vals = Dp_vals / norm
                
                ref_D[m_idx, :] = D_vals
                # f = a * D' / D
                with np.errstate(divide='ignore', invalid='ignore'):
                    f_vals = a_vals * Dp_vals / D_vals
                # Handle any NaNs
                f_vals = np.nan_to_num(f_vals, nan=0.55)
                ref_f[m_idx, :] = f_vals
            else:
                # Fallback if solver fails
                ref_D[m_idx, :] = a_vals**0.55
                ref_f[m_idx, :] = 0.55 * np.ones(N)
        except Exception:
            ref_D[m_idx, :] = a_vals**0.55
            ref_f[m_idx, :] = 0.55 * np.ones(N)

    # Reshape reference arrays for easy comparison
    # ref_D is (M, N), ref_f is (M, N)
    # Our ansatz outputs will be (M*N,) which we reshape to (M, N)
    
    # Precompute features for the ansatz
    # u = 1 + w0 + wa*(1-a)
    # q = 1 - Om(a)
    # Note: The ansatz uses Om(a) in f = Om(a)^gamma.
    # We need to compute Om(a) for the ansatz evaluation too.
    
    def _compute_om_a():
        # Om_a = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(3u))
        u = 1.0 + W0_g + Wa_g * (1.0 - A_g)
        num = Om0_g * A_g**3
        den = num + (1.0 - Om0_g) * A_g**(3.0 * u)
        with np.errstate(divide='ignore', invalid='ignore'):
            om_a = num / den
        return np.nan_to_num(om_a, nan=1.0)

    Om_a_global = _compute_om_a()
    u_global = 1.0 + W0_g + Wa_g * (1.0 - A_g)
    q_global = 1.0 - Om_a_global # Wait, q in task is 1 - Om(a)? 
    # Task says: q=1-Om(a). 
    # My previous code used q = 1 - Om0. 
    # Let's re-read carefully: "q=1-Om(a)".
    # Okay, q depends on a.
    
    # Re-calculate q_global
    q_global = 1.0 - Om_a_global
    
    def _gamma(c):
        g = (c[0] + c[1]*q_global + c[2]*q_global**2 + c[3]*u_global
             + c[4]*Wa_g*A_g + c[5]*u_global*q_global)
        return np.clip(g, 0.1, 1.2)

    def _f_ansatz(c):
        g = _gamma(c)
        return Om_a_global ** g

    def objective(c):
        f_ana = _f_ansatz(c).reshape(M, N)
        
        # Error in f
        with np.errstate(divide='ignore', invalid='ignore'):
            err_f = np.abs(f_ana - ref_f) / np.abs(ref_f)
        err_f = np.nan_to_num(err_f, nan=1.0)
        
        # Error in D
        # D_ana(a) / D_ana(1) = exp(-int_1^a f da) ?? 
        # Task: "D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)"
        # Let's check the integral definition.
        # integral from ln(a) to 0 of f(exp(x)) dx.
        # Let t = ln(a'), a' = exp(t). da' = a' dt.
        # Standard growth: D(a) = C * exp(-int^a f(a')/a' da')?
        # f = d ln D / d ln a.
        # ln D(a) - ln D(1) = int_1^a f(a') d ln a' = int_1^a f(a') da'/a'.
        # Let x = ln a'. dx = da'/a'.
        # Limits: a'=1 -> x=0. a'=a -> x=ln a.
        # int_0^{ln a} f(exp(x)) dx.
        # So D(a)/D(1) = exp( int_0^{ln a} f(exp(x)) dx ).
        # The task says: exp(-integral_ln(a)^0 ...).
        # integral_ln(a)^0 is int from ln(a) to 0.
        # int_{ln a}^0 = - int_0^{ln a}.
        # So exp(- ( - int_0^{ln a} )) = exp( int_0^{ln a} ).
        # This matches.
        
        # We need to compute D_ana(a) / D_ana(1) for each model.
        # D_ana(a_i) / D_ana(1) = exp( int_{ln 1}^{ln a_i} f(exp(x)) dx )
        # = exp( int_{0}^{ln a_i} f(exp(x)) dx )
        
        # Let's compute this integral numerically for each model.
        # x_grid = ln(a_vals).
        # f_vals_on_x = f_ana[:, :] (since f is function of a, and a=exp(x))
        # We need int_0^{x_i} f(exp(x)) dx.
        # This is cumulative integral of f_ana with respect to x.
        # dx = da/a.
        # So integral is sum of f(a_k) * (a_{k+1}-a_k)/a_k? No, trapezoidal on x.
        
        x_vals = np.log(a_vals)
        # f_ana is (M, N)
        # We want I_i = int_{x_0}^{x_i} f(x) dx. Note x_0 = ln(0.1) != 0.
        # We need normalization to D(1).
        # D(a_i)/D(1) = exp( int_{ln 1}^{ln a_i} f dx ) = exp( int_{0}^{x_i} f dx ).
        # Let J_i = int_{x_0}^{x_i} f dx.
        # int_0^{x_i} = J_i - J_0 (where J_0 is int from x_0 to 0? No, x_0 < 0).
        # Let's define K_i = int_{x_0}^{x_i} f(exp(x)) dx.
        # Then int_{0}^{x_i} = K_i - K_0 (if we define K_0 as integral from x_0 to 0).
        # Actually, let's just compute the cumulative integral from x_0 to x_i.
        # Then D(a_i)/D(1) = exp( K_i - K_{idx_1} ), where idx_1 is index where a=1.
        
        # Compute cumulative integral of f_ana w.r.t x
        # dx = diff(x_vals)
        dx = np.diff(x_vals)
        # trap = 0.5 * (f[:, :-1] + f[:, 1:]) * dx
        # cum = cumsum(trap, axis=1)
        # cum[:, i] = int_{x_0}^{x_{i+1}}
        
        # Let's do this efficiently.
        trap = 0.5 * (f_ana[:, :-1] + f_ana[:, 1:]) * dx
        cum_int = np.cumsum(trap, axis=1)
        # cum_int[:, i] is integral from x_0 to x_{i+1}
        # We need integral from 0 to x_i.
        # Let idx_1 be index of a=1.0.
        # a_vals = [0.1, ..., 1.0]. idx_1 = N-1.
        # Integral from 0 to x_i:
        # If i < N-1: int_0^{x_i} = - int_{x_i}^0.
        # int_{x_i}^0 = int_{x_i}^{x_{N-1}} = cum_int[:, N-2] - cum_int[:, i-1] (for i>0).
        # This is getting messy.
        
        # Simpler: Compute D_ana directly by integrating f.
        # D_ana(a_i) = D_ana(a_0) * exp( int_{x_0}^{x_i} f dx ).
        # We normalize D_ana(1) = 1.
        # So D_ana(a_i) = exp( int_{x_{N-1}}^{x_i} f dx ).
        # int_{x_{N-1}}^{x_i} = - int_{x_i}^{x_{N-1}}.
        
        # Let's compute S_i = int_{x_i}^{x_{N-1}} f(exp(x)) dx for all i.
        # S_{N-1} = 0.
        # S_i = f_{i} * (x_{i+1}-x_i)/2 + ... 
        # We can compute reverse cumulative sum.
        
        # trap_rev[i] = 0.5 * (f[:, i] + f[:, i+1]) * (x[:, i+1] - x[:, i])
        # S[:, i] = sum_{k=i}^{N-2} trap_rev[:, k]
        
        trap_rev = 0.5 * (f_ana[:, :-1] + f_ana[:, 1:]) * dx
        # S[:, i] = cumsum of trap_rev from i to end.
        # S[:, N-1] = 0.
        # S[:, i] = S[:, i+1] + trap_rev[:, i]
        
        S = np.zeros((M, N))
        for i in range(N-2, -1, -1):
            S[:, i] = S[:, i+1] + trap_rev[:, i]
            
        # D_ana(a_i) / D_ana(1) = exp( - S[:, i] )
        D_ana = np.exp(-S)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            err_D = np.abs(D_ana - ref_D) / np.abs(ref_D)
        err_D = np.nan_to_num(err_D, nan=1.0)
        
        # Max relative error
        max_err_f = np.max(err_f)
        max_err_D = np.max(err_D)
        max_err = max(max_err_f, max_err_D)
        
        # Smooth max for optimization
        # Use log-sum-exp with high temperature
        beta = 100.0
        vals = np.concatenate([err_f.ravel(), err_D.ravel()])
        if vals.max() < 1e-15:
            return 0.0
        # Shift for stability
        m = vals.max()
        s = np.logaddexp.reduce((vals - m) * beta) / beta + m
        return float(s)

    # Optimization
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best = x0.copy()
    best_f = objective(best)
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        res = minimize(objective, best, method='Nelder-Mead', 
                       options=dict(xatol=1e-6, fatol=1e-8, maxiter=200, adaptive=True))
        if res.fun < best_f:
            best_f = res.fun
            best = np.clip(res.x, -10.0, 10.0)
        else:
            # Perturbation
            cand = best + 0.05 * rng.standard_normal(6)
            cand = np.clip(cand, -10.0, 10.0)
            fc = objective(cand)
            if fc < best_f:
                best_f = fc
                best = cand
            else:
                # Check time, break if no improvement in a few tries?
                pass
                
    return list(best)
# EVOLVE-BLOCK-END
