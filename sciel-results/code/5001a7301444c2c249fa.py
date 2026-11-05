import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    """
    Fits a degree-5 polynomial to the training data.
    Returns coefficients for ascending powers of the original x.
    """
    start_time = time.time()
    
    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    
    # Flatten just in case
    x = x.ravel()
    y = y.ravel()
    
    # 1. Scale x for numerical stability
    # Use robust statistics? Mean/Std is standard.
    mu = np.mean(x)
    sigma = np.std(x)
    
    # Avoid division by zero if std is 0 (unlikely for chaotic system)
    if sigma < 1e-12:
        sigma = 1.0
        
    z = (x - mu) / sigma
    
    # 2. Construct Vandermonde matrix for z (degree 5)
    # V[:, j] = z^j
    # Using np.vander with increasing=True
    V = np.vander(z, N=6, increasing=True)
    
    # 3. Solve least squares
    # Using np.linalg.lstsq for robustness
    # a are the coefficients in the scaled basis z
    try:
        a, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to simpler method if lstsq fails (highly unlikely for 6xN)
        # Use normal equations with small regularization
        A = V.T @ V
        b = V.T @ y
        a = np.linalg.solve(A + 1e-12 * np.eye(6), b)

    # 4. Convert coefficients from scaled basis z to original basis x
    # P(x) = sum_{k=0}^5 a_k * ((x - mu) / sigma)^k
    # P(x) = sum_{k=0}^5 (a_k / sigma^k) * (x - mu)^k
    # (x - mu)^k = sum_{j=0}^k binom(k, j) x^j (-mu)^(k-j)
    # c_j = sum_{k=j}^5 (a_k / sigma^k) * binom(k, j) * (-mu)^(k-j)
    
    c = np.zeros(6)
    for k in range(6):
        scale_k = a[k] / (sigma ** k)
        for j in range(k + 1):
            # binom(k, j)
            from math import comb
            binom_kj = comb(k, j)
            term = scale_k * binom_kj * ((-mu) ** (k - j))
            c[j] += term
            
    # 5. Deadline guard (not strictly necessary for this speed, but required by directive)
    if time.time() - start_time > 110:
         # Return best result found so far
         return c
         
    return c
