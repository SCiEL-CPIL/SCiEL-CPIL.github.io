# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_entropies(P, slopes):
    """
    Computes Shannon entropies for projections along given slopes.
    P is a 10x10 array.
    slopes is a list of floats.
    Returns a dict mapping slope to entropy.
    """
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    entropies = {}
    for r in slopes:
        # Compute values x + r*y
        values = X + r * Y
        # Unique values and their probabilities
        # Since values might be floats, we need to be careful with uniqueness if r is not integer.
        # Here slopes are -1, 0, 1, 2, which are integers, so x + r*y are integers.
        # We can use np.unique with return_inverse to bin them.
        unique_vals, inverse = np.unique(values, return_inverse=True)
        
        # Sum probabilities for each unique value
        probs = np.zeros(len(unique_vals))
        np.add.at(probs, inverse, P)
        
        # Filter out zero probabilities for entropy calculation
        probs = probs[probs > 1e-12]
        if len(probs) == 0:
            entropies[r] = 0.0
        else:
            # Shannon entropy in bits
            entropies[r] = -np.sum(probs * np.log2(probs))
            
    return entropies

def _objective(z):
    """
    Objective function for optimization.
    z is a flat array of length 100.
    Returns - (D / M) to be minimized.
    """
    P = z.reshape(10, 10)
    # Convert to probabilities using softmax to ensure P > 0 and sum(P) = 1
    # Subtract max for numerical stability
    P_exp = np.exp(P - np.max(P))
    P_norm = P_exp / np.sum(P_exp)
    
    # Compute entropies
    Hs = _compute_entropies(P_norm, [-1.0, 0.0, 1.0, 2.0])
    
    D = Hs[-1.0]
    M = max(Hs[0.0], Hs[1.0], Hs[2.0])
    
    if M < 1e-9:
        return 1e9 # Large penalty
    
    return -D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    This implementation uses numerical optimization (L-BFGS-B) over log-probabilities
    to maximize the ratio D/M. Multiple random restarts are used to avoid local minima.
    """
    n = 10
    best_P = None
    best_score = -np.inf
    
    # Number of restarts
    n_restarts = 5
    
    for i in range(n_restarts):
        # Initialize z (log-probabilities) randomly
        # Start near uniform (z=0) or random
        if i == 0:
            z0 = np.zeros(n * n)
        else:
            z0 = np.random.normal(0, 1, size=n * n)
            
        try:
            res = minimize(_objective, z0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Convert optimal z back to P
                P_opt = res.x.reshape(n, n)
                P_exp = np.exp(P_opt - np.max(P_opt))
                P_norm = P_exp / np.sum(P_exp)
                
                # Verify validity
                if np.any(np.isnan(P_norm)) or np.any(np.isinf(P_norm)):
                    continue
                if np.sum(P_norm) < 1e-9:
                    continue
                if np.any(P_norm < -1e-9): # Should not happen with softmax
                    continue
                    
                # Calculate actual score for this P
                Hs = _compute_entropies(P_norm, [-1.0, 0.0, 1.0, 2.0])
                D = Hs[-1.0]
                M = max(Hs[0.0], Hs[1.0], Hs[2.0])
                if M > 1e-9:
                    score = D / M
                    if score > best_score:
                        best_score = score
                        best_P = P_norm.copy()
                        
        except Exception as e:
            pass
            
    if best_P is None:
        # Fallback to uniform distribution
        best_P = np.ones((n, n)) / (n * n)
        
    return best_P


# EVOLVE-BLOCK-END
