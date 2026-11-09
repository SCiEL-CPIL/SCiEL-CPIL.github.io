import numpy as np
from scipy.special import roots_laguerre
import time

def _compute_score(roots):
    if len(roots) != 12:
        return float('inf')
    if np.any(roots <= 0):
        return float('inf')
    prod = np.prod(roots)
    if prod <= 0 or not np.isfinite(prod):
        return float('inf')
    return 1.0 / np.sqrt(prod)

def construct():
    start_time = time.time()
    deadline = start_time + 580
    
    # Strategy 1: Roots of L_12(x)
    try:
        roots_l12 = roots_laguerre(12)[0]
        score_l12 = _compute_score(roots_l12)
        best_roots = roots_l12
        best_score = score_l12
    except Exception:
        best_roots = np.arange(1, 13) * 2.0
        best_score = _compute_score(best_roots)

    # Strategy 2: Roots of orthogonal polynomial with weight e^-x * L_12(x)^2
    if time.time() < deadline - 10:
        try:
            # Get coefficients of L_12(x)
            # L_m(x) = sum_{k=0}^m (-1)^k * binomial(m, k) / k! * x^k
            m = 12
            coeffs = []
            for k in range(m, -1, -1):
                c = ((-1)**k) * (np.math.factorial(m) / (np.math.factorial(k) * np.math.factorial(m-k))) / np.math.factorial(k)
                coeffs.append(c)
            coeffs = np.array(coeffs)
            
            # Square the polynomial
            # P(x) = (L_12(x))^2
            p_coeffs = np.convolve(coeffs, coeffs)
            
            # Compute moments M_k = integral x^k e^-x P(x) dx
            # P(x) = sum_{j=0}^{2m} d_j x^j
            # M_k = sum_{j=0}^{2m} d_j (k+j)!
            max_k = 2 * m - 1 # We need moments up to 2m-2 for N=m Hankel
            # Actually for N roots, we need moments M_0 ... M_{2N-2}
            # N=12, so M_0 ... M_22
            moments = []
            for k in range(2 * m - 1):
                val = 0.0
                for j, d_j in enumerate(p_coeffs):
                    if d_j != 0:
                        val += d_j * np.math.factorial(k + j)
                moments.append(val)
            
            # Build Hankel matrix H (12x12)
            H = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    H[i, j] = moments[i + j]
            
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                # Add small regularization if not positive definite
                L = np.linalg.cholesky(H + 1e-10 * np.eye(m))
            
            # Construct Jacobi matrix J
            J = np.zeros((m, m))
            # Diagonal alpha_i
            for i in range(m):
                if i == 0:
                    # alpha_0 = M_1 / M_0
                    # From Cholesky: L[0,0] = sqrt(M_0), L[1,0] = M_1/L[0,0]
                    # alpha_0 = L[1,0] / L[0,0]
                    J[i, i] = L[1, 0] / L[0, 0]
                else:
                    # alpha_i = L[i, i-1] / L[i-1, i-1] - L[i-1, i-1] / L[i-2, i-2] ?
                    # Let's use the derived relation:
                    # alpha_i = (L[i, i-1] / L[i-1, i-1]) + (L[i-1, i-1] / L[i-2, i-2])? No.
                    # Standard formula:
                    # alpha_i = L[i, i-1] / L[i-1, i-1] - L[i-1, i-1] / L[i-2, i-2] is incorrect indexing.
                    
                    # Let's use the property:
                    # beta_i = (L[i, i] / L[i-1, i-1])^2 ? No.
                    
                    # Reliable formula from literature:
                    # alpha_i = (L[i, i-1] / L[i-1, i-1]) + (L[i-1, i-1] / L[i-2, i-2])?
                    
                    # Let's re-derive for i=1 (index 1):
                    # alpha_1 = L[2,1]/L[1,1] - L[1,1]/L[0,0]
                    # beta_1 = (L[2,2]/L[1,1])^2? No, beta_1 = (L[1,1]/L[0,0])^2?
                    
                    # Actually, the off-diagonal element J[i, i+1] = sqrt(beta_{i+1}).
                    # J[i, i] = alpha_i.
                    
                    # Let's use the explicit values:
                    # J[0,0] = L[1,0]/L[0,0]
                    # J[0,1] = L[1,1]/L[0,0]
                    # J[1,1] = L[2,1]/L[1,1] - L[1,1]/L[0,0]
                    # J[1,2] = L[2,2]/L[1,1]
                    
                    # General pattern:
                    # J[i, i] = L[i+1, i] / L[i, i] - L[i, i] / L[i-1, i-1] (for i>0)
                    # J[i, i+1] = L[i+1, i+1] / L[i, i]
                    
                    J[i, i] = (L[i, i-1] / L[i-1, i-1]) - (L[i-1, i-1] / L[i-2, i-2]) if i > 1 else (L[i, i-1] / L[i-1, i-1]) - (L[i-1, i-1] / L[0, 0])
                    if i < m - 1:
                        J[i, i+1] = L[i+1, i+1] / L[i, i]
            
            # Symmetrize J
            J = (J + J.T) / 2.0
            
            # Eigenvalues are the roots
            roots_refined = np.linalg.eigvalsh(J)
            
            # Filter positive roots
            roots_refined = roots_refined[roots_refined > 0]
            
            if len(roots_refined) == 12:
                score_refined = _compute_score(roots_refined)
                if score_refined < best_score:
                    best_score = score_refined
                    best_roots = roots_refined
        except Exception:
            pass

    if time.time() > deadline:
        return list(best_roots)

    return list(best_roots)
