import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial approximation to the unresolved fast-scale forcing.
    Uses Ridge Regression on centered and scaled inputs for numerical stability and 
    regularization against noise.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64).flatten()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).flatten()
    
    N = len(x)
    if N == 0:
        return np.zeros(6)

    # Center and scale x for stability
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        sigma = 1.0
    z = (x - mu) / sigma

    # Build Vandermonde matrix for z
    # V[:, i] = z^i
    V = np.vander(z, N=6, increasing=True)

    # Ridge Regression: min ||V a - y||^2 + lambda ||a||^2
    # (V^T V + lambda I) a = V^T y
    
    VtV = V.T @ V
    Vty = V.T @ y
    
    # Choose lambda
    # A common heuristic is to look at the eigenvalues of VtV.
    # Or simply use a small value relative to the diagonal.
    diag_mean = np.mean(np.diag(VtV))
    
    # We can try a small grid of lambda values to pick the best one based on a simple criterion,
    # but given the time constraint, a robust fixed heuristic is safer.
    # Let's try lambda = 1e-6 * diag_mean.
    # If the data is very noisy, we might need more.
    # Let's use a slightly stronger regularization to act as a smoother.
    lambda_reg = 1e-5 * diag_mean
    
    A = VtV + lambda_reg * np.eye(6)
    
    try:
        a = np.linalg.solve(A, Vty)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular (should not happen with regularization)
        a = np.linalg.lstsq(V, y, rcond=None)[0]

    if not np.all(np.isfinite(a)):
        return np.zeros(6)

    # Transform coefficients from z-basis to x-basis
    # P(z) = sum_{i=0}^5 a_i z^i
    # z = (x - mu) / sigma
    # P(x) = sum_{i=0}^5 a_i sigma^{-i} (x - mu)^i
    # Let b_i = a_i * sigma^{-i}
    # P(x) = sum_{i=0}^5 b_i (x - mu)^i
    # (x - mu)^i = sum_{j=0}^i binom(i, j) x^j (-mu)^{i-j}
    
    b = np.zeros(6)
    for i in range(6):
        b[i] = a[i] * (sigma ** (-i))
        
    c = np.zeros(6)
    for i in range(6):
        if b[i] == 0:
            continue
        for j in range(i + 1):
            # Term is b_i * binom(i, j) * x^j * (-mu)^(i-j)
            coeff = b[i] * _binom(i, j) * ((-mu) ** (i - j))
            c[j] += coeff

    if not np.all(np.isfinite(c)):
        return np.zeros(6)

    return c

def _binom(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # Use symmetry
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result *= (n - i)
        result //= (i + 1)
    return result
