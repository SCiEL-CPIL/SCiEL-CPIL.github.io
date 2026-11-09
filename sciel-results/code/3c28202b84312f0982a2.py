"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to Linder-Van de Stark growth index via IRLS + polish."""
    t_start = time.time()
    deadline = t_start + 75.0  # 15s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa)
    Om0_vals = np.array([0.2, 0.25, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.2, -1.1, -1.0, -0.9, -0.8])
    wa_vals = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3])

    # Scale factor grid for fitting - denser
    a_vals = np.linspace(0.05, 0.99, 60)

    # Build design matrix and target
    rows = []
    targets = []
    # Store cosmology params for later use
    cosmo_list = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0:
                        continue
                    
                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    cosmo_list.append((Om0, w0, wa, a))
                    
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)

    # IRLS to minimize max error (L-infinity approximation)
    n = len(b)
    coeffs = np.zeros(6)
    
    # Initial LSQ fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # IRLS iterations
    for it in range(20):
        if time.time() > deadline:
            break
        # Compute residuals
        pred = A @ coeffs
        resid = b - pred
        max_resid = np.max(np.abs(resid))
        if max_resid < 1e-12:
            break
        # Weights: inverse of |residual| (for L-infinity)
        weights = 1.0 / np.maximum(np.abs(resid), 1e-10)
        # Weighted LSQ
        try:
            W = np.diag(weights)
            WA = W @ A
            Wb = W @ b
            coeffs, _, _, _ = np.linalg.lstsq(WA, Wb, rcond=None)
        except Exception:
            break
        coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Polish pass: use scipy.optimize to minimize max |residual|
    if time.time() < deadline - 5.0:
        from scipy.optimize import minimize
        
        def max_error(c):
            c = np.clip(c, -10.0, 10.0)
            pred = A @ c
            return np.max(np.abs(b - pred))
        
        try:
            res = minimize(max_error, coeffs, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success or res.fun < max_error(coeffs):
                coeffs = np.clip(res.x, -10.0, 10.0)
        except Exception:
            pass
    
    # Final verification on broader grid
    valid = True
    check_Om0 = np.array([0.2, 0.28, 0.35])
    check_w0 = np.array([-1.2, -1.0, -0.8])
    check_wa = np.array([-0.5, 0.0, 0.3])
    check_a = np.linspace(0.05, 0.99, 15)
    
    for Om0 in check_Om0:
        for w0 in check_w0:
            for wa in check_wa:
                for a in check_a:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    if E2 <= 0:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    gamma = (coeffs[0] + coeffs[1]*q + coeffs[2]*q*q + 
                             coeffs[3]*u + coeffs[4]*wa*a + coeffs[5]*u*q)
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
