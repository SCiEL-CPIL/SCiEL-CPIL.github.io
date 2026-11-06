"""Combined program: Odd-power polynomial closure with standardized scaling and robust fitting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)

    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.zeros(6)

    # If very few points, fall back to linear or zeros
    if x.size < 3:
        # Cannot fit 3 parameters reliably, fall back to linear if possible
        if x.size >= 2:
            coeff = np.zeros(6)
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff[:2] = c_lin
            return coeff
        else:
            return np.zeros(6)

    # 1. Standardize x to improve conditioning
    mu = np.mean(x)
    sigma = np.std(x)
    
    if sigma < 1e-12:
        # Degenerate case: all x are the same
        # Fit a constant (but we must return 6 coeffs)
        # The tendency is likely 0 or constant, but odd function implies 0 if x is constant?
        # If x is constant, the "odd" symmetry argument is weak. Just return mean y as constant term?
        # But the interface expects a polynomial.
        # Let's just return zeros or a simple linear fit if possible (it won't be).
        # Actually, if sigma is 0, we can't define z.
        # Return zeros is safer than garbage.
        return np.zeros(6)

    # 2. Define scaled variable z
    # z = (x - mu) / sigma
    # We fit y = a1*z + a3*z^3 + a5*z^5
    
    # Build design matrix for scaled odd powers
    z = (x - mu) / sigma
    A_scaled = np.column_stack([z, z**3, z**5])
    
    # 3. Solve for coefficients a = [a1, a3, a5]
    # Use ridge regression for stability in the scaled space
    # The scale of A_scaled.T @ A_scaled is roughly N (number of points)
    # A small lambda relative to the diagonal is appropriate.
    try:
        # Compute normal equations
        AT_A = A_scaled.T @ A_scaled
        AT_y = A_scaled.T @ y
        
        # Regularization parameter
        # Use a small fraction of the mean diagonal
        trace_AT_A = np.trace(AT_A)
        if trace_AT_A > 0:
            lam = 1e-8 * trace_AT_A / 3.0 # Very small, just for stability
        else:
            lam = 1e-8
            
        # Solve (AT_A + lam*I) a = AT_y
        I = np.eye(3)
        a_scaled = np.linalg.solve(AT_A + lam * I, AT_y)
        
    except np.linalg.LinAlgError:
        # Fallback to lstsq if solve fails
        a_scaled, _, _, _ = np.linalg.lstsq(A_scaled, y, rcond=None)

    # 4. Convert scaled coefficients to original x coefficients
    # P(x) = a1 * (x-mu)/sigma + a3 * ((x-mu)/sigma)^3 + a5 * ((x-mu)/sigma)^5
    # Let alpha = 1/sigma, beta = -mu/sigma
    # z = alpha * x + beta
    # P(x) = a1*(alpha*x + beta) + a3*(alpha*x + beta)^3 + a5*(alpha*x + beta)^5
    
    alpha = 1.0 / sigma
    beta = -mu / sigma
    
    coeff = np.zeros(6)
    
    # Helper to add contribution of a_j * (alpha*x + beta)^j to coeff
    # (alpha*x + beta)^j = sum_{k=0}^j binom(j,k) (alpha*x)^k beta^(j-k)
    #                  = sum_{k=0}^j binom(j,k) alpha^k beta^(j-k) x^k
    
    def add_term(a_j, j):
        if a_j == 0:
            return
        for k in range(j + 1):
            # Calculate binom(j, k)
            # j is small (1, 3, 5), so we can compute directly
            binom_val = 1.0
            if k > 0:
                # binom(j, k) = j! / (k! (j-k)!)
                # Iterative calculation
                num = 1
                for m in range(k):
                    num *= (j - m)
                den = 1
                for m in range(1, k + 1):
                    den *= m
                binom_val = num / den
            
            term_coeff = a_j * binom_val * (alpha ** k) * (beta ** (j - k))
            coeff[k] += term_coeff

    add_term(a_scaled[0], 1) # a1 * z
    add_term(a_scaled[1], 3) # a3 * z^3
    add_term(a_scaled[2], 5) # a5 * z^5

    # Verify finite
    if not np.all(np.isfinite(coeff)):
        # Fallback to linear fit in original space
        try:
            c_lin = np.polynomial.polynomial.polyfit(x, y, 1)
            coeff = np.zeros(6)
            coeff[:2] = c_lin
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
