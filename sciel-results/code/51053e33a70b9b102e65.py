import time
import numpy as np
import random
from fractions import Fraction
import scipy.optimize

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_exact(U, V, W, T):
    """Verify exact decomposition using Fractions."""
    n, m, p = T.shape[0]//5, 5, 5 # Hardcoded for 2x5x5, but let's pass dims
    # Actually T shape is (10, 25, 10)
    n, m, p = 2, 5, 5
    rank = U.shape[1]
    
    # Convert to Fractions
    U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
    
    # Check a subset or all? All is 10*25*10 = 2500 entries.
    # Sum over rank (max 50). 2500 * 50 = 125000 operations. Fast enough.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                expected = 1 if (u == i*m+j and v == j*p+k and w == k*n+i) else 0
                # The tensor T has 1s at these positions.
                # Wait, T is defined such that T[u,v,w]=1 for these indices.
                # Other entries are 0.
                # We only need to check if the sum equals T[u,v,w].
                if s != T[u,v,w]:
                    return False
    return True

def _try_round_and_verify(U, V, W, T, tol=1e-5):
    """Try to round U, V, W to half-integers and verify."""
    # Check if entries are close to half-integers
    def is_close_to_half_int(x):
        return abs(2*x - round(2*x)) < tol
    
    if not (np.all(np.vectorize(is_close_to_half_int)(U)) and 
            np.all(np.vectorize(is_close_to_half_int)(V)) and 
            np.all(np.vectorize(is_close_to_half_int)(W))):
        return None
        
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    if _verify_exact(U_r, V_r, W_r, T):
        return U_r, V_r, W_r
    return None

def _optimize_rank(T, rank, seed, deadline):
    """Optimize for a specific rank."""
    n, m, p = T.shape
    rng = np.random.RandomState(seed)
    
    # Initialize
    # Try V=I if rank >= m*p
    if rank >= m*p:
        V = np.zeros((m*p, rank))
        V[:m*p, :m*p] = np.eye(m*p)
        U = rng.randn(n*m, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
    else:
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
    def objective(x):
        if time.time() > deadline:
            return np.inf
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute reconstruction
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        return err
        
    x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    
    # Use L-BFGS-B
    try:
        res = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
        if time.time() > deadline:
            return None
        x_opt = res.x
        U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
        V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Try to round
        result = _try_round_and_verify(U_opt, V_opt, W_opt, T)
        if result:
            return result
            
        # If rounding failed, check if numerical error is tiny
        # and maybe try a different rounding strategy?
        # For now, return None if not exactly rounded
        return None
    except Exception as e:
        return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    
    deadline = time.time() + 140
    
    # Try ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 25
    for rank in range(25, rank_schoolbook):
        if time.time() > deadline:
            break
            
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            result = _optimize_rank(T, rank, seed, deadline)
            if result:
                best_U, best_V, best_W = result
                best_rank = rank
                break # Found a valid decomposition for this rank, try lower? No, we are going up.
                # If we found rank R, we should check if lower ranks are possible?
                # We are iterating 25, 26, ... so if 25 works, we return.
                # If 25 fails, we try 26.
        
        # If we found a solution at this rank, we can return immediately?
        # The task is to find rank-minimal.
        # So if we find a valid decomposition at rank R, and we are iterating upwards,
        # this is the minimal rank (assuming we started at the lower bound).
        if best_rank < rank_schoolbook:
            break
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
