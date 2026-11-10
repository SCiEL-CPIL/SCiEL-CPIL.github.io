"""Improved growth index approximation using a compact ansatz."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # New Approach: Least-squares fit of the 6 coefficients to the Linder (2005) 
    # approximation for gamma, which is highly accurate for CPL models.
    # Target: gamma_ref = 0.55 + 0.05 * (1 + w(a))
    # Basis: gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    # Generate a grid of representative cosmologies and scale factors
    # Om_m0 in [0.2, 0.3], w0 in [-1.2, 0.5], wa in [-0.5, 0.5]
    # a in [0.01, 1.0]
    
    # We need to be careful: q depends on Om_m(a), which depends on all params.
    # Let's sample a dense grid.
    
    Om_m0_vals = np.linspace(0.2, 0.3, 5)
    w0_vals = np.linspace(-1.2, 0.5, 10)
    wa_vals = np.linspace(-0.5, 0.5, 10)
    a_vals = np.linspace(0.05, 1.0, 50)
    
    A_matrix = []
    y_vector = []
    
    for Om_m0 in Om_m0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    # Calculate Om_m(a)
                    # Om_m(a) = Om_m0 * a^-3 / (Om_m0 * a^-3 + Om_de0 * a^(-3*(1+w0+wa)))
                    # Om_de0 = 1 - Om_m0
                    # Note: For CPL, the DE density is Om_de0 * a^(-3*(1+w0+wa))
                    # This is an approximation for small wa, but standard for this context.
                    # Actually, exact CPL integral is harder, but for fitting gamma,
                    # the Linder approximation assumes a specific relation.
                    # Let's use the standard Omega_m(a) for constant w first, 
                    # or the CPL approximation.
                    # Omega_m(a) = (Om_m0 * a**-3) / (Om_m0 * a**-3 + (1-Om_m0) * a**(-3*(1+w0+wa)))
                    
                    w_a = w0 + wa * (1 - a)
                    # Use the CPL density relation
                    # rho_de(a) = rho_de0 * a^(-3*(1+w0+wa))
                    # This is the standard definition used in these approximations.
                    
                    if a == 0:
                        continue
                    rho_m = Om_m0 * (a ** -3)
                    rho_de = (1 - Om_m0) * (a ** (-3 * (1 + w0 + wa)))
                    Om_m_a = rho_m / (rho_m + rho_de)
                    
                    q = 1 - Om_m_a
                    u = 1 + w0 + wa * (1 - a)
                    
                    # Target Gamma from Linder (2005)
                    # gamma ~ 0.55 + 0.05 * (1 + w)
                    # This is the leading order.
                    gamma_ref = 0.55 + 0.05 * (1 + w_a)
                    
                    # Build row for A
                    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    A_matrix.append(row)
                    y_vector.append(gamma_ref)
    
    A = np.array(A_matrix)
    y = np.array(y_vector)
    
    # Solve least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
    
    # Clip coefficients to [-10, 10] as per contract
    coeffs = np.clip(coeffs, -10, 10)
    
    return coeffs.tolist()
# EVOLVE-BLOCK-END
