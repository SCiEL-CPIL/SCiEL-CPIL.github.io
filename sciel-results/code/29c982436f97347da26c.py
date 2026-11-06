import time
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.linalg import lstsq

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = 85.0  # 5s safety margin under 90s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _Om_m(a, Om0, Oml, w0, wa):
    """Compute matter density parameter Omega_m(a)."""
    a = np.asarray(a, dtype=float)
    a3 = a ** 3
    # Standard CPL form
    # Om_m(a) = Om0 * a^-3
    # Om_L(a) = Oml * a^-3 * (1 + 3w0 + 2wa + wa(1-a))
    return Om0 / a3 + Oml / a3 * (1.0 + 3.0 * w0 + 2.0 * wa + wa * (1.0 - a))

def _ref_gamma(a, Om0, w0, wa):
    """
    Reference growth index gamma.
    We use the GR approximation for f = Om^gamma.
    f_ref = 0.5*Om + 1.25*Om*(Om-1)/H2
    gamma_ref = log(f_ref) / log(Om)
    """
    Oml = 1.0 - Om0
    Om = _Om_m(a, Om0, Oml, w0, wa)
    H2 = Om + Oml * (1.0 + 3.0 * w0 + 2.0 * wa + wa * (1.0 - a))
    f_ref = 0.5 * Om + 1.25 * Om * (Om - 1.0) / H2
    
    # Avoid log(0) or negative values
    # Om is positive in standard cosmologies. f_ref is positive.
    # Clip to avoid numerical issues
    Om_safe = np.clip(Om, 1e-10, None)
    f_safe = np.clip(f_ref, 1e-10, None)
    
    gamma = np.log(f_safe) / np.log(Om_safe)
    return gamma

def _err(c, B, Om, refF, refD, a, loga):
    """
    Compute the error metric.
    c: (6,)
    B: (6, N, M) - Basis functions
    Om: (N, M)
    refF: (N, M)
    refD: (N, M)
    a: (M,)
    loga: (M,)
    """
    # gamma = c . B
    gamma = np.tensordot(c, B, axes=(0, 0))
    
    # Check bounds
    if gamma.min() < 0.1 or gamma.max() > 1.2:
        return 1e6
        
    # f = Om^gamma
    f = np.power(Om, gamma)
    
    if not np.all(np.isfinite(f)):
        return 1e6
        
    err_f = np.max(np.abs(f - refF) / refF)
    
    # Compute D
    # D(a) = exp(- \int_a^1 f(a') da')
    # Efficient cumulative trapezoid
    J = np.zeros_like(f)
    dx = np.diff(a)
    avg_f = 0.5 * (f[:, 1:] + f[:, :-1])
    steps = avg_f * dx[None, :]
    J[:, 1:] = np.cumsum(steps, axis=1)
    
    J_total = J[:, -1]
    I = J_total[:, None] - J
    D = np.exp(-I)
    
    if not np.all(np.isfinite(D)):
        return 1e6
        
    err_D = np.max(np.abs(D - refD) / refD)
    
    return max(err_f, err_D)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Seed coefficients (constant gamma ~ 0.55)
    seed = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    try:
        # 1. Define validation cosmologies
        # We want to cover the space effectively.
        # Om0 in [0.2, 0.4], w0 in [-1.2, 0.0], wa in [-0.5, 0.5]
        # Use a stratified grid
        om0_vals = np.array([0.2, 0.25, 0.3, 0.35, 0.4])
        w0_vals = np.array([-1.2, -1.0, -0.8, -0.5, 0.0])
        wa_vals = np.array([-0.5, -0.3, 0.0, 0.3, 0.5])
        
        models = []
        for o in om0_vals:
            for w in w0_vals:
                for wa in wa_vals:
                    models.append((o, w, wa))
        
        # Add some random outliers to ensure robustness
        rng = np.random.default_rng(42)
        for _ in range(50):
            o = rng.uniform(0.15, 0.45)
            w = rng.uniform(-1.5, 0.2)
            wa = rng.uniform(-0.8, 0.8)
            models.append((o, w, wa))
            
        models = np.array(models)
        N = len(models)
        
        # 2. Define a grid
        # Focus on a where validation is likely dense (z < 3, a > 0.25)
        # But include some low a to constrain the fit
        a = np.geomspace(0.1, 1.0, 200)
        M = len(a)
        
        # 3. Precompute Basis and References
        B = np.zeros((6, N, M))
        Om = np.zeros((N, M))
        refF = np.zeros((N, M))
        refD = np.zeros((N, M))
        
        for i, (o, w, wa) in enumerate(models):
            Oml = 1.0 - o
            Om_i = _Om_m(a, o, Oml, w, wa)
            Om[i, :] = Om_i
            
            q = 1.0 - Om_i
            u = 1.0 + w + wa * (1.0 - a)
            
            B[0, i, :] = 1.0
            B[1, i, :] = q
            B[2, i, :] = q * q
            B[3, i, :] = u
            B[4, i, :] = wa * a
            B[5, i, :] = u * q
            
            # Reference f
            H2 = Om_i + Oml * (1.0 + 3.0 * w + 2.0 * wa + wa * (1.0 - a))
            f_ref = 0.5 * Om_i + 1.25 * Om_i * (Om_i - 1.0) / H2
            refF[i, :] = f_ref
            
            # Reference D
            # Compute cumulative integral
            J = np.zeros(M)
            dx = np.diff(a)
            avg_f = 0.5 * (f_ref[1:] + f_ref[:-1])
            steps = avg_f * dx
            J[1:] = np.cumsum(steps)
            J_total = J[-1]
            I = J_total - J
            refD[i, :] = np.exp(-I)
        
        # 4. Strategy: Global Linear Least Squares on Gamma
        # We want gamma_ansatz = gamma_ref
        # gamma_ansatz = sum c_i B_i
        # This is linear in c.
        # We can solve min || B c - gamma_ref ||^2
        # But we care about relative error in f, which is approximately relative error in gamma * log(Om).
        # Let's weight the linear fit to prioritize regions where log(Om) is large (early times) 
        # or just do a standard fit on gamma.
        
        # Flatten B and gamma_ref
        # B_flat: (6, N*M)
        # gamma_ref_flat: (N*M,)
        
        # However, the task minimizes max relative error. L2 fit minimizes sum of squares.
        # This is a good starting point though.
        
        # Let's try to fit gamma directly.
        gamma_ref = np.zeros((N, M))
        for i in range(N):
            gamma_ref[i, :] = _ref_gamma(a, models[i,0], models[i,1], models[i,2])
            
        # Reshape for lstsq
        B_flat = B.reshape(6, -1)
        gamma_flat = gamma_ref.reshape(-1)
        
        # Weights: Weight by 1/log(Om) to make relative error in f uniform?
        # f = Om^gamma -> ln f = gamma ln Om
        # d(ln f) = d(gamma) ln Om + gamma d(ln Om) (d ln Om is 0 for fixed model)
        # So relative error in f is proportional to d(gamma) * ln(Om).
        # To minimize max relative error, we might want to weight points with large ln(Om) less?
        # Or weight by 1/ln(Om) to equalize the contribution to relative error.
        # Let's try unweighted first, then maybe weighted.
        
        # Unweighted L2 fit
        c_ls, residuals, rank, s = lstsq(B_flat, gamma_flat, cond=None)
        
        # Clip coefficients to valid range
        c_ls = np.clip(c_ls, -10.0, 10.0)
        
        # Check if this solution is valid (bounds on gamma)
        gamma_ls = np.tensordot(c_ls, B, axes=(0, 0))
        if gamma_ls.min() < 0.1 or gamma_ls.max() > 1.2:
            # If bounds violated, fall back to seed or project
            # For now, if invalid, we might need to constrain the fit.
            # Let's just use the seed if the LSQ solution is bad.
            c_best = np.array(seed)
        else:
            c_best = c_ls
        
        # 5. Polish with Nelder-Mead on the actual Max-Error metric
        # The L2 fit minimizes sum of squared errors in gamma.
        # The task minimizes max relative error in f and D.
        # A short Nelder-Mead search can fix the "tails" of the distribution.
        
        def objective(c):
            return _err(c, B, Om, refF, refD, a, a)
            
        # Start from LSQ solution
        c0 = c_best.copy()
        
        # Run a few iterations of Nelder-Mead
        # We have plenty of time budget (85s).
        # Let's run a robust optimization.
        
        best_c = c0
        best_err = objective(c0)
        
        # Use a multi-start approach if time permits, but LSQ is a very strong start.
        # Just polish the LSQ result.
        
        if _time_left() > 10.0:
            try:
                res = minimize(objective, c0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'maxfev': 5000, 'xatol': 1e-6,
                                        'fatol': 1e-8, 'adaptive': True})
                c_polished = np.clip(res.x, -10.0, 10.0)
                err_polished = objective(c_polished)
                if err_polished < best_err:
                    best_c = c_polished
                    best_err = err_polished
            except Exception:
                pass
        
        # Final check
        final_err = objective(best_c)
        
        # If the final error is worse than seed (unlikely), revert
        seed_err = objective(np.array(seed))
        if final_err > seed_err:
            best_c = np.array(seed)
            
        out = np.clip(best_c, -10.0, 10.0)
        if not np.all(np.isfinite(out)):
            return seed
            
        return [float(v) for v in out]
        
    except Exception:
        return seed

# Dummy entry point
if __name__ == "__main__":
    construct()
