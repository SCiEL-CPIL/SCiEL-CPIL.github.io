import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Build target tensor
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, i*p+k] = 1.0
    
    # Schoolbook fallback
    def schoolbook():
        rank = nm * mp * np_
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((np_, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[i*p+k, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Try to find lower rank
    best = (U_sb, V_sb, W_sb, rank_sb)
    
    for r in range(28, 45):
        if time.time() > deadline - 15:
            break
        
        # Flatten U, V, W into a single vector
        # x = [U.flatten(), V.flatten(), W.flatten()]
        def residual(x):
            U = x[:nm*r].reshape(nm, r)
            V = x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = x[nm*r+mp*r:].reshape(np_, r)
            # Compute T_recon
            T_recon = np.einsum('at,bt,ct->abc', U, V, W)
            return (T_recon - T).flatten()
        
        # Initial guess: use schoolbook truncated
        x0 = np.zeros(nm*r + mp*r + np_*r)
        # Initialize with small random values
        np.random.seed(123 + r)
        x0 = np.random.randn(len(x0)) * 0.1
        
        result = least_squares(residual, x0, method='lm', max_nfev=1000, xtol=1e-10, ftol=1e-10)
        
        if result.success and result.cost < 1e-8:
            U = result.x[:nm*r].reshape(nm, r)
            V = result.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W = result.x[nm*r+mp*r:].reshape(np_, r)
            best = (U, V, W, r)
            break
    
    U, V, W, rank = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
