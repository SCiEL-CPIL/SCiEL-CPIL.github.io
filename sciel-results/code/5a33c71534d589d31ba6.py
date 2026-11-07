import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using an orthonormal polynomial basis.
    
    We use Legendre polynomials scaled to be orthonormal on the interval [-1, 1].
    The input x is in [-2, 2], so we map it to t = x/2 in [-1, 1].
    The basis functions are phi_k(t) = sqrt((2k+1)/2) * L_k(t).
    This ensures the design matrix is well-conditioned (near-identity Gram matrix),
    allowing BFGS to converge rapidly and reliably to the global least-squares solution.
    """
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    N = len(t)
    # We need 10 basis functions (indices 0 to 9)
    K = 10
    
    # Initialize basis matrix
    # basis[:, k] will hold phi_k(t)
    basis = np.zeros((N, K))
    
    # L_0(t) = 1
    L_prev = np.ones(N)
    # phi_0(t) = sqrt(1/2) * L_0(t)
    basis[:, 0] = np.sqrt(0.5) * L_prev
    
    if K > 1:
        # L_1(t) = t
        L_curr = t.copy()
        # phi_1(t) = sqrt(3/2) * L_1(t)
        basis[:, 1] = np.sqrt(1.5) * L_curr
        
        for k in range(2, K):
            # Recurrence: L_k(t) = ((2k-1)t L_{k-1}(t) - (k-1) L_{k-2}(t)) / k
            L_next = ((2 * k - 1) * t * L_curr - (k - 1) * L_prev) / k
            
            # Orthonormal scaling factor for L_k
            scale = np.sqrt((2 * k + 1) / 2.0)
            basis[:, k] = scale * L_next
            
            L_prev = L_curr
            L_curr = L_next
            
    # Linear combination: y = sum(params[k] * basis[:, k])
    return basis @ params
