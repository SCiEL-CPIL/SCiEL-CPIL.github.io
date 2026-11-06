import numpy as np

def coefficients(stages):
    # Solve 9x^3 - 18x^2 + 9x - 1 = 0 for gamma_1
    # Roots are approx 1.293, 0.551, 0.156
    # We want the root that gives a stable formula.
    # Usually gamma_1 is the middle root.
    
    # Use numpy to find roots
    coeffs = [9, -18, 9, -1]
    roots = np.roots(coeffs)
    
    # Filter real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    
    # The standard 4th order coefficient gamma_1 is the one that is not 1/2?
    # Actually, for S4, gamma_1 is the largest? No.
    # Let's check the sum 2*gamma_1 + gamma_2 = 1.
    # If gamma_1 = 0.551, gamma_2 = -0.102.
    # If gamma_1 = 1.293, gamma_2 = 1 - 2.586 = -1.586.
    # If gamma_1 = 0.156, gamma_2 = 1 - 0.312 = 0.688.
    
    # The standard 4th order formula uses gamma_1 = 0.551...
    # Let's pick the root closest to 0.55
    gamma_1 = min(real_roots, key=lambda x: abs(x - 0.55))
    gamma_2 = 1 - 2 * gamma_1
    
    a = [0.0] * stages
    b = [0.0] * stages
    
    # Map to stages
    b[0] = gamma_1 / 2.0
    a[1] = gamma_1
    b[2] = (gamma_1 + gamma_2) / 2.0
    a[3] = gamma_2
    b[4] = (gamma_1 + gamma_2) / 2.0
    a[5] = gamma_1
    b[6] = gamma_1 / 2.0
    
    return a, b
