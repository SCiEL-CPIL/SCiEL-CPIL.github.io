"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_legendre

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    
    # Flatten inputs
    x = np.asarray(training_x).flatten()
    y = np.asarray(training_subgrid_tendency).flatten()
    
    # Filter out any non-finite values just in case
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 6:
        return np.zeros(6)

    # 1. Scale x to [-1, 1] for stable Legendre evaluation
    # Using quantiles (5% and 95%) to be robust against outliers
    x_min = np.percentile(x, 5)
    x_max = np.percentile(x, 95)
    
    if x_max - x_min < 1e-12:
        x_scaled = np.zeros_like(x)
    else:
        # Map [x_min, x_max] to [-1, 1]
        x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        
    # 2. Construct Legendre design matrix (degrees 0 to 5)
    # P_k(x) are the Legendre polynomials
    # We want to fit y = sum_{k=0}^5 c_k P_k(x_scaled)
    degrees = 6
    # eval_legendre takes an array of degrees and an array of x values
    # Returns a matrix of shape (n_samples, n_degrees) if x is 1D and deg is array
    # Actually, scipy.special.eval_legendre(deg, x) returns an array of the same shape as x if deg is scalar,
    # or if deg is array, it broadcasts.
    # Let's build it explicitly for clarity and control
    X_leg = np.zeros((len(x), degrees))
    for k in range(degrees):
        X_leg[:, k] = eval_legendre(k, x_scaled)
        
    # 3. Regularized Least Squares (Ridge Regression)
    # Minimize ||X_leg * c - y||^2 + alpha * ||c||^2
    # The orthogonal basis makes the design matrix well-conditioned.
    # Regularization helps smooth the fit for better generalization to F=10.
    
    # We can try a small alpha. Since the basis is orthogonal-ish, 
    # the eigenvalues of X^T X are well spread, but high degree terms might still be noisy.
    # Let's solve the normal equations with a small ridge parameter.
    
    # A simple approach: Solve (X^T X + alpha * I) c = X^T y
    # However, standard numpy lstsq doesn't do ridge directly.
    # We can use np.linalg.solve on the normal equations.
    
    XtX = X_leg.T @ X_leg
    Xty = X_leg.T @ y
    
    # Choose alpha. 
    # The scale of XtX depends on N. 
    # A relative ridge penalty is safer.
    # alpha = lambda * trace(XtX) / n_features ? 
    # Let's try a fixed small alpha relative to the diagonal scale.
    # Since Legendre polynomials are orthogonal on [-1,1], the off-diagonals are small.
    # The diagonal entries are roughly N/(2k+1).
    # So the condition number is not huge, but regularization prevents overfitting to noise.
    
    # Let's use a small alpha, e.g., 1e-4 * mean(diag(XtX))
    diag_mean = np.mean(np.diag(XtX))
    alpha = 1e-4 * diag_mean if diag_mean > 0 else 1e-4
    
    # Add ridge to diagonal
    # Do not regularize the constant term (index 0) as much? 
    # Standard ridge regularizes all. Let's regularize all for simplicity, 
    # but maybe reduce penalty on k=0.
    # Actually, for closure, the mean offset is important. 
    # Let's use a diagonal matrix for penalty: 
    # penalty = alpha * diag([0, 1, 1, 1, 1, 1])
    
    penalty = alpha * np.ones(degrees)
    penalty[0] = 0.0 # Do not penalize the mean
    
    A = XtX + np.diag(penalty)
    
    try:
        c_leg = np.linalg.solve(A, Xty)
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular (should not happen with ridge)
        c_leg, _, _, _ = np.linalg.lstsq(X_leg, y, rcond=None)
        
    # Check time
    if time.time() - start_time > 140:
        # Convert and return best effort
        pass # Conversion is fast, so we just proceed

    # 4. Convert Legendre coefficients to Monomial coefficients
    # We need to express P_k(x_scaled) in terms of powers of x.
    # x_scaled = (2/(x_max-x_min)) * x - (2*x_min/(x_max-x_min)) - 1
    # Let a = 2/(x_max-x_min), b = -2*x_min/(x_max-x_min) - 1
    # x_scaled = a*x + b
    
    # We want y = sum_{k=0}^5 c_k P_k(a*x + b)
    # P_k(z) is a polynomial in z of degree k.
    # We can compute the monomial coefficients of P_k(a*x + b) for each k,
    # then multiply by c_k and sum them up.
    
    # Alternatively, we can construct the monomial basis in terms of x_scaled,
    # solve for coefficients in that basis, and then transform.
    # But we already have c_leg.
    
    # Let's use the property that P_k(z) = sum_{j=0}^k p_{k,j} z^j
    # where p_{k,j} are the coefficients of the k-th Legendre polynomial.
    # We can compute these coefficients once.
    
    # Coefficients of Legendre polynomials
    # P_0(z) = 1
    # P_1(z) = z
    # P_2(z) = 0.5*(3z^2 - 1)
    # ...
    
    # Let's generate the monomial coefficients for P_k(z)
    leg_monomial_coeffs = np.zeros((degrees, degrees))
    for k in range(degrees):
        # Get coefficients of P_k(z)
        # Using scipy.special.eval_legendre doesn't give coefficients directly.
        # We can use numpy.polynomial.legendre.leg2poly
        from numpy.polynomial import legendre
        coeffs_k = legendre.leg2poly([0]*k + [1]) # Coeffs of P_k in Legendre basis -> Monomial basis
        # leg2poly returns coefficients in ascending powers of z
        # Pad to length degrees
        leg_monomial_coeffs[k, :len(coeffs_k)] = coeffs_k
        
    # Now, P_k(x_scaled) = sum_j leg_monomial_coeffs[k,j] * (x_scaled)^j
    # And x_scaled = a*x + b
    # So we need to expand (a*x + b)^j into powers of x.
    
    # Let's build the transformation matrix T such that:
    # [ (x_scaled)^0, (x_scaled)^1, ..., (x_scaled)^5 ]^T = T @ [ 1, x, x^2, ..., x^5 ]^T
    # Actually, we want:
    # sum_k c_k P_k(x_scaled) = sum_k c_k sum_j p_{k,j} (a*x+b)^j
    # = sum_j (sum_k c_k p_{k,j}) (a*x+b)^j
    
    # Let d_j = sum_k c_k p_{k,j}. This is c_leg @ leg_monomial_coeffs.T
    d = c_leg @ leg_monomial_coeffs.T
    
    # Now we have y = sum_{j=0}^5 d_j (a*x + b)^j
    # We need to expand (a*x + b)^j = sum_{m=0}^j binom(j,m) a^m b^{j-m} x^m
    # So the coefficient of x^m in the final polynomial is:
    # C_m = sum_{j=m}^5 d_j * binom(j,m) * a^m * b^{j-m}
    
    a = 2.0 / (x_max - x_min)
    b = -2.0 * x_min / (x_max - x_min) - 1.0
    
    C = np.zeros(degrees)
    for m in range(degrees):
        coeff_m = 0.0
        for j in range(m, degrees):
            # binom(j, m)
            from math import comb
            binom_val = comb(j, m)
            term = d[j] * binom_val * (a ** m) * (b ** (j - m))
            coeff_m += term
        C[m] = coeff_m
        
    # C contains the coefficients for powers x^0 to x^5
    # The task requires ASCENDING powers: c_0 + c_1*x + ...
    # So C is already in the correct order.
    
    # Final check for finite values
    if not np.all(np.isfinite(C)):
        # Fallback to simple linear fit if something went wrong
        coeff = np.zeros(6)
        try:
            coeff[:2] = np.polyfit(x, y, 1)[::-1] # polyfit returns descending, so reverse
        except:
            coeff = np.zeros(6)
        return coeff
        
    return C

# EVOLVE-BLOCK-END
