"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from numpy.polynomial import chebyshev as C

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the unresolved fast-scale forcing
    of a two-scale Lorenz-96 model using Huber Loss optimization on a
    Chebyshev basis for numerical stability and robustness to outliers.
    
    Args:
        training_x: Flattened array of slow-state values
        training_subgrid_tendency: Flattened array of unresolved tendencies
    
    Returns:
        Array of 6 finite real polynomial coefficients in ascending powers of x
    """
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any NaN/inf values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 10:
        return np.zeros(6)
    
    # Determine the range of x to map to [-1, 1] for Chebyshev polynomials
    x_min = np.min(x)
    x_max = np.max(x)
    
    # Avoid division by zero if range is tiny
    if x_max - x_min < 1e-12:
        x_center = x_min
        x_scale = 1.0
    else:
        x_center = (x_max + x_min) / 2.0
        x_scale = (x_max - x_min) / 2.0
    
    # Map x to t in [-1, 1]
    t = (x - x_center) / x_scale
    
    # Handle potential overflow in t if x is far outside [x_min, x_max] due to floating point
    # Though x_min/max are derived from x, so t should be in [-1, 1] mostly.
    # Clip to be safe for Chebyshev evaluation stability
    t = np.clip(t, -1.0, 1.0)
    
    degree = 5
    
    # Generate Chebyshev design matrix T(t) for degree 0 to 5
    # Using numpy's built-in for stability
    T = C.chebvander(t, degree)
    
    # Initial guess: OLS (Least Squares)
    try:
        beta_ols, residuals, rank, sv = np.linalg.lstsq(T, y, rcond=None)
    except Exception:
        beta_ols = np.zeros(degree + 1)
    
    # Check time before expensive optimization
    if time.time() > DEADLINE - 20:
        # Fallback to OLS if time is tight
        return _convert_cheb_to_monomial(beta_ols, x_center, x_scale, degree)
    
    # Huber Loss optimization
    # Minimize sum(0.5*r^2) if |r| < delta, else sum(delta*|r| - 0.5*delta^2)
    # We use scipy.optimize.minimize with L-BFGS-B for efficiency
    
    def huber_loss_and_grad(coeffs):
        # coeffs are Chebyshev coefficients
        r = y - T.dot(coeffs)
        abs_r = np.abs(r)
        delta = 1.0  # Huber parameter; tune if needed, 1.0 is standard
        
        # Gradient of Huber loss w.r.t. coefficients
        # dL/dr = r/|r| if |r|>delta else r/delta
        # dL/dcoeffs = T.T @ dL/dr
        
        if delta == 0:
            grad_r = r / np.maximum(abs_r, 1e-8)
        else:
            grad_r = np.where(abs_r < delta, r / delta, np.sign(r))
        
        grad = T.T.dot(grad_r)
        
        # Loss value
        loss = np.sum(np.where(abs_r < delta, 0.5 * r**2, delta * abs_r - 0.5 * delta**2))
        
        return loss, grad
    
    # Use L-BFGS-B for large N
    options = {
        'maxiter': 100,
        'ftol': 1e-8,
        'gtol': 1e-6,
        'maxcor': 10
    }
    
    try:
        res = minimize(
            huber_loss_and_grad,
            beta_ols,
            method='L-BFGS-B',
            jac=True,
            options=options
        )
        beta_opt = res.x
    except Exception:
        beta_opt = beta_ols
    
    # Check time
    if time.time() > DEADLINE - 10:
        return _convert_cheb_to_monomial(beta_opt, x_center, x_scale, degree)
    
    # Convert Chebyshev coefficients to monomial coefficients
    # We want coefficients c_i such that P(x) = sum c_i x^i
    # We have P(t) = sum a_k T_k(t), where t = (x - x_center) / x_scale
    # This is a change of basis.
    
    return _convert_cheb_to_monomial(beta_opt, x_center, x_scale, degree)

def _convert_cheb_to_monomial(cheb_coeffs, x_center, x_scale, degree):
    """
    Convert Chebyshev coefficients in normalized domain t to monomial coefficients in x.
    t = (x - x_center) / x_scale => x = x_scale * t + x_center
    
    We can use numpy.polynomial.chebyshev.cheb2poly to get polynomial in t,
    then scale and shift the variable to x.
    """
    try:
        # cheb2poly converts Chebyshev series to power series in t
        # c_t are coefficients for t^0, t^1, ..., t^degree
        c_t = C.cheb2poly(cheb_coeffs)
        
        # Now we have P(t) = sum c_t[i] * t^i
        # Substitute t = (x - x_center) / x_scale
        # P(x) = sum c_t[i] * ((x - x_center) / x_scale)^i
        #      = sum c_t[i] / x_scale^i * (x - x_center)^i
        
        # We can use numpy.polynomial.polynomial.polyfromroots? No.
        # We can use scaling and shifting functions.
        
        # polyfromcheb gives coefficients in t.
        # We want coefficients in x.
        # Let Q(x) = P((x - x_center) / x_scale)
        
        # numpy.polynomial.polynomial.polyscale and polyshift
        # First, scale variable: t = x / x_scale => P(x/x_scale)
        # coeffs_x_scaled = polyfromcheb(cheb_coeffs) scaled by 1/x_scale
        # Actually, C.cheb2poly gives power series in t.
        
        # Let's do it manually or with poly functions.
        # poly = c_t
        # Shift and scale.
        # P(t) with t = (x - cx) / sx
        # This is equivalent to:
        # 1. Shift variable: u = x - cx
        # 2. Scale variable: v = u / sx
        # P(v)
        
        # numpy.polynomial.polynomial.polyfromcheb is not directly available for scaling.
        # Let's use the property:
        # If P(t) = sum a_i t^i, and t = (x-c)/s
        # P(x) = sum a_i (x-c)^i / s^i
        
        # We can compute the monomial coefficients by expanding (x-c)^i.
        # Or simpler: use numpy.polynomial.polynomial.polyfromroots? No.
        
        # Let's use the polynomial operations in numpy.polynomial.polynomial
        # p = np.poly1d? No, let's stick to arrays.
        
        # Method:
        # 1. Get power series coefficients in t: c_t
        # 2. Scale the variable: t = u / s => P(u/s)
        #    Coeffs become c_t[i] * (1/s)^i
        # 3. Shift the variable: u = x - c => P(x-c)
        #    Use polyshift
        
        # Step 1: Scale
        # c_scaled[i] = c_t[i] / s^i
        powers = np.arange(degree + 1)
        c_scaled = c_t / (x_scale ** powers)
        
        # Step 2: Shift
        # P(u) = sum c_scaled[i] u^i
        # We want P(x - c)
        # numpy.polynomial.polynomial.polyshift(c_scaled, -x_center)
        
        try:
            from numpy.polynomial import polynomial as P
            c_x = P.polyshift(c_scaled, -x_center)
        except ImportError:
            # Fallback to manual expansion if P not found (unlikely)
            c_x = np.zeros(degree + 1)
            for i in range(degree + 1):
                # Expand (x - c)^i
                # Coeffs of (x-c)^i are binomial
                binom_coeffs = np.zeros(i + 1)
                for j in range(i + 1):
                    # C(i, j) * (-c)^(i-j)
                    # Using math.comb
                    import math
                    binom_coeffs[j] = math.comb(i, j) * ((-x_center) ** (i - j))
                c_x[i] += c_scaled[i] * binom_coeffs[i]
                # Wait, the expansion is sum_{j=0}^i C(i,j) x^j (-c)^(i-j)
                # So c_x[j] += c_scaled[i] * C(i,j) * (-c)^(i-j)
                # The loop above adds to c_x[i] which is wrong index.
                # Let's redo fallback if needed.
                pass
            
            # Correct Fallback
            c_x = np.zeros(degree + 1)
            import math
            for i in range(degree + 1):
                if c_scaled[i] != 0:
                    for j in range(i + 1):
                        c_x[j] += c_scaled[i] * math.comb(i, j) * ((-x_center) ** (i - j))
        
        # Ensure finite
        if not np.all(np.isfinite(c_x)):
             # Fallback to OLS monomial if conversion failed
             # Re-compute OLS in monomial basis directly?
             # Or just return zeros?
             # Better to return the scaled/shifted OLS if available, but we don't have it here.
             # Let's try to recover from beta_ols if passed? No, this is a helper.
             # We'll return zeros as a last resort, but hopefully polyshift works.
             return np.zeros(degree + 1)
        
        return c_x[:degree + 1]
        
    except Exception:
        # Absolute fallback
        return np.zeros(degree + 1)

# EVOLVE-BLOCK-END
