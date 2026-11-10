# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_entropies(P):
    """
    Compute H_{-1} (D) and H_0, H_1, H_2 from the joint distribution P[y, x].
    P has shape (10, 10) where P[y, x] is the probability of (X=x, Y=y).
    """
    n = P.shape[0]
    P = np.clip(P, 0, None)
    P = P / P.sum() if P.sum() > 0 else P

    # Slope r: value = x + r*y
    # r = -1: x - y
    # r = 0: x
    # r = 1: x + y
    # r = 2: x + 2y

    def entropy_of_projection(P, r):
        n = P.shape[0]
        ys = np.arange(n)
        xs = np.arange(n)
        values = xs[None, :] + r * ys[:, None]
        
        # Get unique values and their probabilities
        # Since values are integers (for integer r), we can use bincount
        # But r can be non-integer in general, though here r is -1, 0, 1, 2
        # For integer r, values are integers in range [min, max]
        
        # To be safe, use a dictionary or unique
        flat_vals = values.ravel()
        flat_probs = P.ravel()
        
        # For integer slopes, we can use bincount
        if r == 0:
            # values = x, range 0..9
            probs = np.bincount(flat_vals.astype(int), weights=flat_probs, minlength=n)
        elif r == 1:
            # values = x + y, range 0..18
            probs = np.bincount(flat_vals.astype(int), weights=flat_probs, minlength=2*n-1)
        elif r == 2:
            # values = x + 2y, range 0..27
            probs = np.bincount(flat_vals.astype(int), weights=flat_probs, minlength=2*n-1+9) # max is 9+2*9=27, so 28
        elif r == -1:
            # values = x - y, range -9..9
            shifted = flat_vals.astype(int) + n - 1
            probs = np.bincount(shifted, weights=flat_probs, minlength=2*n-1)
        else:
            # General case
            unique_vals, inverse = np.unique(flat_vals, return_inverse=True)
            probs = np.bincount(inverse, weights=flat_probs)
        
        # Shannon entropy in bits, ignoring zero probabilities
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    D = entropy_of_projection(P, -1)
    H0 = entropy_of_projection(P, 0)
    H1 = entropy_of_projection(P, 1)
    H2 = entropy_of_projection(P, 2)
    
    M = max(H0, H1, H2)
    
    return D, M

def _value(P):
    """
    Compute the objective value: D / M.
    """
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-9:
        return 0.0
    P = P / total
    
    D, M = _compute_entropies(P)
    if M <= 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Initial guess: uniform
    P0 = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P0.copy()
    best_val = _value(P0)
    
    # Flatten for optimization
    x0 = P0.ravel().copy()
    
    # We want to maximize D / M.
    # This is equivalent to minimizing -D / M.
    # However, the objective is not smooth due to the max and the ratio.
    # Let's try to use scipy.optimize.minimize with SLSQP or Trust-Constr.
    # We need to handle the non-negativity and sum-to-1 constraints.
    
    # Alternative approach:
    # Maximize log(D) - log(M).
    # This is still non-smooth due to max.
    
    # Let's try a direct optimization with a smooth approximation of max.
    # Soft-max: M_beta = (1/beta) * log(sum(exp(beta * H_i)))
    # As beta -> inf, M_beta -> max(H_i).
    
    def objective_smooth(x, beta=10.0):
        P = x.reshape(n, n)
        P = np.clip(P, 0, None)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        
        # Compute entropies
        D, H0, H1, H2 = _compute_entropies(P)
        # Wait, _compute_entropies returns D, M. I need individual Hs.
        # Let's refactor to return individual Hs.
        
        # For now, let's just use the raw value if we can't easily get individual Hs.
        # But soft-max needs individual Hs.
        
        # Let's compute H0, H1, H2 individually inside the objective.
        # This is inefficient but correct for the prototype.
        
        Hs = []
        for r in [0, 1, 2]:
            Hs.append(_entropy_single(P, r))
        Hs = np.array(Hs)
        
        # Soft max
        max_H = np.max(Hs)
        if max_H < 1e-9:
            return 0.0
        # Log-sum-exp trick for stability
        M_soft = (1.0 / beta) * np.log(np.sum(np.exp(beta * (Hs - max_H)))) + max_H
        
        if D < 1e-9 or M_soft < 1e-9:
            return 0.0
            
        return -np.log(D) + np.log(M_soft) # Minimize -log(D/M) = -log(D) + log(M)
        
    def _entropy_single(P, r):
        n = P.shape[0]
        ys = np.arange(n)
        xs = np.arange(n)
        values = xs[None, :] + r * ys[:, None]
        flat_vals = values.ravel().astype(int)
        flat_probs = P.ravel()
        
        if r == 0:
            probs = np.bincount(flat_vals, weights=flat_probs, minlength=n)
        elif r == 1:
            probs = np.bincount(flat_vals, weights=flat_probs, minlength=2*n-1)
        elif r == 2:
            probs = np.bincount(flat_vals, weights=flat_probs, minlength=3*n-2)
        else:
            # Fallback
            unique_vals, inverse = np.unique(flat_vals, return_inverse=True)
            probs = np.bincount(inverse, weights=flat_probs)
            
        probs = probs[probs > 1e-15]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))

    # Constraints
    # Sum to 1
    # Non-negativity is handled by bounds
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones_like(x)}
    ]
    
    bounds = [(0, None) for _ in range(n*n)]
    
    # Try optimization with different beta values
    betas = [5.0, 10.0, 20.0, 50.0]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        res = minimize(
            objective_smooth,
            current_x,
            args=(beta,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-8}
        )
        
        if res.success:
            current_x = res.x
            P_candidate = current_x.reshape(n, n)
            val = _value(P_candidate)
            if val > best_val:
                best_val = val
                best_P = P_candidate.copy()
    
    # Final polish with the actual objective (hard max)
    def objective_hard(x):
        P = x.reshape(n, n)
        val = _value(P)
        return -val
    
    if time.time() < deadline:
        res = minimize(
            objective_hard,
            current_x,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if res.success:
            current_x = res.x
            P_candidate = current_x.reshape(n, n)
            val = _value(P_candidate)
            if val > best_val:
                best_val = val
                best_P = P_candidate.copy()
    
    # Ensure validity
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-9:
        best_P = best_P / total
    
    return best_P

# EVOLVE-BLOCK-END
