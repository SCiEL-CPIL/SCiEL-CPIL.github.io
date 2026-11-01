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
                
    def snap_to_half_integer(x):
        """
        Snap to the closest half-integer (k/2).
        """
        return np.round(x * 2) / 2.0

    def verify_exact(U, V, W, T):
        """
        Verify if U, V, W exactly reconstruct T.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Reconstruct
        recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Check against T
        if not np.allclose(recon, T, atol=1e-9):
            return False
        
        return True

    # Try ranks from 19 to 23
    for r in range(19, 24):
        if time.time() > deadline - 20:
            break
            
        # Number of trials per rank
        trials = 50
        if r == 19:
            trials = 100
        
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Sparse Random Initialization
            # Values: 0, 1, -1
            # Prob: 0.5, 0.25, 0.25
            U = np.random.choice([0, 1, -1], size=(nm, r), p=[0.5, 0.25, 0.25]).astype(float)
            V = np.random.choice([0, 1, -1], size=(mp, r), p=[0.5, 0.25, 0.25]).astype(float)
            W = np.random.choice([0, 1, -1], size=(pn, r), p=[0.5, 0.25, 0.25]).astype(float)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                A = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, r)
                T_flat = T.reshape(nm, mp * pn)
                
                try:
                    U, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    U = U.T
                except:
                    break
                    
                # Update V
                A = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, r)
                T_flat = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
                
                try:
                    V, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    V = V.T
                except:
                    break
                    
                # Update W
                A = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, r)
                T_flat = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
                
                try:
                    W, _, _, _ = np.linalg.lstsq(A.T, T_flat.T, rcond=None)
                    W = W.T
                except:
                    break
                    
            # Check convergence and snap
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            
            # If it's close, try to snap to half-integers
            if np.allclose(reconstructed, T, atol=1e-4):
                U_snap = np.array([[snap_to_half_integer(x) for x in row] for row in U])
                V_snap = np.array([[snap_to_half_integer(x) for x in row] for row in V])
                W_snap = np.array([[snap_to_half_integer(x) for x in row] for row in W])
                
                # Verify snapped version
                reconstructed_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
                
                if np.allclose(reconstructed_snap, T, atol=1e-9):
                    # Convert to lists
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    U_list = U_school.tolist()
    V_list = V_school.tolist()
    W_list = W_school.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
