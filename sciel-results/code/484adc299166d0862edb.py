"""Improved program for the suite task; fits a robust degree-five polynomial closure."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Returns 6 coefficients in ascending powers of x.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Ensure inputs are float64 numpy arrays
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Handle edge cases
    if x.size == 0:
        return np.zeros(6)
    if x.size < 6:
        # If not enough points for degree 5, fall back to lower degree or zeros
        # But the task implies sufficient data. Let's just return zeros or a simple fit.
        # Given the context of "spun-up trajectories", data should be plentiful.
        pass

    # Check for time
    if time.time() > deadline:
        return np.zeros(6)

    # 1. Scale x to [-1, 1] for better conditioning with Legendre polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        # Degenerate case
        return np.zeros(6)
    
    # u = (2x - (x_max + x_min)) / (x_max - x_min)
    # u = alpha * x + beta
    alpha = 2.0 / (x_max - x_min)
    beta = -(x_max + x_min) / (x_max - x_min)
    
    u = alpha * x + beta

    # 2. Construct Legendre Vandermonde matrix for degree 5
    # P_0(u) = 1
    # P_1(u) = u
    # P_2(u) = 0.5*(3u^2 - 1)
    # ...
    # We can use numpy.polynomial.legendre.legvander
    
    try:
        V = np.polynomial.legendre.legvander(u, 5)
    except Exception:
        return np.zeros(6)

    # 3. Least squares fit
    # c_leg are the coefficients in the Legendre basis: y ≈ sum(c_leg[i] * P_i(u))
    try:
        c_leg, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    except Exception:
        return np.zeros(6)

    # 4. Convert Legendre coefficients to monomial coefficients in terms of u
    # p_u(u) = sum(a_i * u^i)
    a_u = np.polynomial.legendre.leg2poly(c_leg)
    
    # Ensure a_u has exactly 6 elements
    if len(a_u) < 6:
        a_u = np.pad(a_u, (0, 6 - len(a_u)))
    else:
        a_u = a_u[:6]

    # 5. Substitute u = alpha * x + beta into the polynomial
    # We want coefficients b_i such that sum(b_i * x^i) = sum(a_i * (alpha*x + beta)^i)
    # This is equivalent to expanding the polynomial.
    
    # We can do this by evaluating the polynomial at a set of points and using polyfit,
    # but direct algebraic expansion is more precise and fast.
    
    # Let's use the fact that we can compute the new coefficients by convolving or iterating.
    # Alternatively, use numpy.polynomial.polynomial.polymul or similar logic.
    
    # Simpler robust way: Evaluate the Legendre-based polynomial at a set of distinct points x_eval, 
    # then use polyfit to get monomial coefficients. 
    # However, polyfit is also Vandermonde based. 
    # Let's stick to algebraic expansion for precision.
    
    # p(x) = a_0 + a_1(alpha x + beta) + a_2(alpha x + beta)^2 + ... + a_5(alpha x + beta)^5
    
    # We can build the polynomial in x by adding terms.
    # Start with p(x) = 0
    # For each i from 0 to 5:
    #   term_i(x) = a_i * (alpha x + beta)^i
    #   Expand (alpha x + beta)^i and add a_i * coeff
    
    # Coefficients of (alpha x + beta)^i:
    # This is a binomial expansion: sum_{k=0}^i binom(i,k) (alpha x)^k beta^{i-k}
    # = sum_{k=0}^i binom(i,k) alpha^k beta^{i-k} x^k
    
    b_x = np.zeros(6)
    
    for i in range(6):
        a_i = a_u[i]
        if a_i == 0:
            continue
        # Expand (alpha x + beta)^i
        # Coefficient of x^k is binom(i,k) * alpha^k * beta^(i-k)
        for k in range(i + 1):
            # Binomial coefficient
            binom_coeff = 1.0
            if i > 0:
                # Calculate binom(i, k)
                # Using iterative calculation to avoid overflow or precision issues for small i
                # Since i <= 5, we can just compute it directly
                num = 1
                for m in range(k):
                    num *= (i - m)
                den = 1
                for m in range(1, k + 1):
                    den *= m
                binom_coeff = num / den
            else:
                binom_coeff = 1.0
            
            coeff_val = a_i * binom_coeff * (alpha ** k) * (beta ** (i - k))
            b_x[k] += coeff_val

    # Check time again
    if time.time() > deadline:
        return b_x

    return b_x

# EVOLVE-BLOCK-END
