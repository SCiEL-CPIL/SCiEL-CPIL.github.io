import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Fallback: Schoolbook
    U_school = np.zeros((nm, 24), dtype=int)
    V_school = np.zeros((mp, 24), dtype=int)
    W_school = np.zeros((pn, 24), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
                
    # Try ranks from 18 to 23
    # Known lower bounds suggest rank > 17 for 2x3x4?
    # Let's try 18, 19, 20, 21, 22, 23
    
    def als_optimize(U, V, W, T, nm, mp, pn, max_iter=200, deadline=None):
        """Run ALS optimization and return final U, V, W."""
        for it in range(max_iter):
            if deadline and time.time() > deadline:
                break
                
            # Update U
            # A is (mp*pn, r) where A[idx, r] = V[b, r] * W[c, r]
            A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, -1)
            r = A.shape[1]
            T_flat = T.reshape(nm, mp * pn)
            
            try:
                U, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                U = U.T
            except:
                break
                
            # Update V
            # A is (nm*pn, r) where A[idx, r] = U[a, r] * W[c, r]
            A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, -1)
            T_flat = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
            
            try:
                V, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                V = V.T
            except:
                break
                
            # Update W
            # A is (nm*mp, r) where A[idx, r] = U[a, r] * V[b, r]
            A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, -1)
            T_flat = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
            
            try:
                W, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                W = W.T
            except:
                break
                
        return U, V, W

    def try_snap_and_verify(U, V, W, T, nm, mp, pn, scale=2):
        """Try to snap to half-integers and verify."""
        # Snap to half-integers
        U_snap = np.round(U * scale) / scale
        V_snap = np.round(V * scale) / scale
        W_snap = np.round(W * scale) / scale
        
        # Verify snapped version
        reconstructed_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        if np.allclose(reconstructed_snap, T, atol=1e-9):
            # Convert to lists
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            return {"U": U_list, "V": V_list, "W": W_list}
        return None

    def try_polish(U, V, W, T, nm, mp, pn, deadline=None):
        """Try to polish the solution with tighter tolerances."""
        U_polish, V_polish, W_polish = als_optimize(U, V, W, T, nm, mp, pn, max_iter=50, deadline=deadline)
        
        # Try different scales
        for scale in [2, 4, 8]:
            result = try_snap_and_verify(U_polish, V_polish, W_polish, T, nm, mp, pn, scale=scale)
            if result:
                return result
        return None

    # Try ranks from 18 to 23
    for r in range(18, 24):
        if time.time() > deadline - 20:
            break
            
        # Number of trials per rank
        trials = 30
        if r < 20:
            trials = 50
            
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if trial % 3 == 0:
                # Random Gaussian
                U = np.random.randn(nm, r) * 0.1
                V = np.random.randn(mp, r) * 0.1
                W = np.random.randn(pn, r) * 0.1
            elif trial % 3 == 1:
                # Random with some zeros
                U = np.random.randn(nm, r) * 0.1
                V = np.random.randn(mp, r) * 0.1
                W = np.random.randn(pn, r) * 0.1
                # Zero out some entries
                for mat in [U, V, W]:
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            if random.random() < 0.3:
                                mat[i, j] = 0
            else:
                # Structured initialization based on schoolbook
                U = np.zeros((nm, r), dtype=float)
                V = np.zeros((mp, r), dtype=float)
                W = np.zeros((pn, r), dtype=float)
                
                # Try to create a structured initial guess
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = (i * m + j) * p + (j * p + k) - j * p + k
                            col = (i * m + j) * p + k
                            if col < r:
                                U[i*m+j, col] = 1.0
                                V[j*p+k, col] = 1.0
                                W[k*n+i, col] = 1.0
                
                # Add small random perturbation
                U += np.random.randn(nm, r) * 0.01
                V += np.random.randn(mp, r) * 0.01
                W += np.random.randn(pn, r) * 0.01
            
            # Run ALS optimization
            U, V, W = als_optimize(U, V, W, T, nm, mp, pn, max_iter=100, deadline=deadline)
            
            # Check convergence and try to snap
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            
            if np.allclose(reconstructed, T, atol=1e-5):
                # Try to snap and verify
                result = try_snap_and_verify(U, V, W, T, nm, mp, pn)
                if result:
                    return result
                
                # Try polish
                result = try_polish(U, V, W, T, nm, mp, pn, deadline=deadline)
                if result:
                    return result

    # If no better rank found, return schoolbook
    U_list = [[int(x) for x in row] for row in U_school]
    V_list = [[int(x) for x in row] for row in V_school]
    W_list = [[int(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
