"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to Linder-Van de Stark growth index via least squares."""
    t_start = time.time()
    deadline = t_start + 80.0  # 10s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa)
    # Focus on physically reasonable range
    Om0_vals = np.array([0.2, 0.25, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.2, -1.1, -1.0, -0.9, -0.8])
    wa_vals = np.array([-0.5, -0.3, -0.1, 0.0, 0.1, 0.3])

    # Scale factor grid for fitting
    a_vals = np.linspace(0.1, 0.99, 40)

    # Build design matrix and target
    # gamma_target = (5/7) * (Om(a) / u)^(4/7) where u = 1+w0+wa*(1-a)
    # Our ansatz: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    # where q = 1 - Om(a)

    rows = []
    targets = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    # Compute Om(a) for flat CPL cosmology
                    # Om_m(a) = Om0 * a^3 / E(a)^2
                    # E(a)^2 = Om0*a^3 + (1-Om0)*a^(-3*(1+w0+wa/3)) ... actually:
                    # For CPL: w = w0 + wa*(1-a)
                    # E(a)^2 = Om0*a^3 + (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
                    # More precisely: E(a)^2 = Om0*a^3 + (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
                    # Actually the standard form:
                    # E(a)^2 = Om0*a^3 + (1-Om0) * exp(3*wa*(a-1)) * a^(-3*(1+w0))
                    # Let me use: E(a)^2 = Om0*a^3 + (1-Om0) * a^(-3*(1+w0)) * exp(-3*wa*(1-a))
                    
                    a_cubed = a**3
                    # Dark energy density factor
                    # rho_de(a) ∝ a^(-3*(1+w0)) * exp(-3*wa*(1-a))
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
                    
                    # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    
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

    # Solve least squares
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)

    # Verify gamma stays in [0.1, 1.2] on a test grid
    # Quick check on the same grid
    valid = True
    for Om0 in Om0_vals[:3]:
        for w0 in w0_vals[:3]:
            for wa in wa_vals[:3]:
                for a in np.linspace(0.1, 0.99, 10):
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
        # Fallback to constant if gamma out of range
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
