import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to check if a set of factors is valid
    def is_valid(U, V, W, r, n, m, p):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def snap_to_half_int(A):
        return np.round(A * 2) / 2.0

    # Function to find 2x3x3 decomposition
    def find_2x3x3(r_target):
        n2, m2, p2 = 2, 3, 3
        best_factors = None
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            U = np.random.randn(n2*m2, r_target)
            V = np.random.randn(m2*p2, r_target)
            W = np.random.randn(p2*n2, r_target)
            
            # Normalize to avoid huge/small values
            U /= np.linalg.norm(U) * np.sqrt(r_target)
            V /= np.linalg.norm(V) * np.sqrt(r_target)
            W /= np.linalg.norm(W) * np.sqrt(r_target)
            
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 20:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Efficiently compute residuals and gradients
                # Residual R[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1
                # We can loop over c to accumulate gradients
                for c in range(r_target):
                    # Tensor T_c = U[:,c] (n*m) x V[:,c] (m*p) x W[:,c] (p*n)
                    # Contribution to Residual at (i,j,k) is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    
                    # This loop is O(n*m*p*r)
                    # For 2x3x3, n*m*p = 18. r=15. 18*15 = 270 ops per step. Very fast.
                    
                    # Let's just compute the full residual tensor first for simplicity in code structure
                    pass
                
                # Better: Vectorized residual computation
                # R[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # We can reshape U, V, W and use einsum
                # U: (n, m, r), V: (m, p, r), W: (p, n, r)
                U_reshaped = U.reshape(n2, m2, r_target)
                V_reshaped = V.reshape(m2, p2, r_target)
                W_reshaped = W.reshape(p2, n2, r_target)
                
                # R[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                R = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                error = R - 1.0
                loss = np.sum(error**2)
                
                if loss < 1e-8:
                    # Check for exactness after snapping
                    U_snap = snap_to_half_int(U)
                    V_snap = snap_to_half_int(V)
                    W_snap = snap_to_half_int(W)
                    if is_valid(U_snap, V_snap, W_snap, r_target, n2, m2, p2):
                        return U_snap, V_snap, W_snap
                    # If not exact but very close, maybe continue or accept if close enough for parent?
                    # The task requires EXACT. So we must snap and verify.
                    # If snap fails, we might need to adjust.
                    # For now, return None to try another restart.
                    pass

                # Gradients
                # dU[i,j,c] = 2 * sum_k error[i,j,k] V[j,k,c] W[k,i,c]
                # dV[j,k,c] = 2 * sum_i error[i,j,k] U[i,j,c] W[k,i,c]
                # dW[k,i,c] = 2 * sum_j error[i,j,k] U[i,j,c] V[j,k,c]
                
                dU_reshaped = 2 * np.einsum('ijk,jkc,kic->ijc', error, V_reshaped, W_reshaped)
                dV_reshaped = 2 * np.einsum('ijk,ijc,kic->jkc', error, U_reshaped, W_reshaped)
                dW_reshaped = 2 * np.einsum('ijk,ijc,jkc->kic', error, U_reshaped, V_reshaped)
                
                U -= lr * dU_reshaped.reshape(n2*m2, r_target)
                V -= lr * dV_reshaped.reshape(m2*p2, r_target)
                W -= lr * dW_reshaped.reshape(p2*n2, r_target)
                
                # Optional: Normalize again to prevent drift?
                # U /= np.linalg.norm(U) * np.sqrt(r_target)
                # V /= np.linalg.norm(V) * np.sqrt(r_target)
                # W /= np.linalg.norm(W) * np.sqrt(r_target)
            
            # End of steps for this attempt
            # Check if we converged to something close
            U_reshaped = U.reshape(n2, m2, r_target)
            V_reshaped = V.reshape(m2, p2, r_target)
            W_reshaped = W.reshape(p2, n2, r_target)
            R = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            if np.max(np.abs(R - 1.0)) < 1e-4:
                U_snap = snap_to_half_int(U)
                V_snap = snap_to_half_int(V)
                W_snap = snap_to_half_int(W)
                if is_valid(U_snap, V_snap, W_snap, r_target, n2, m2, p2):
                    return U_snap, V_snap, W_snap
        
        return None

    # Try to find 2x3x3 decomposition
    factors_233 = None
    for r in [15, 16, 17, 18]:
        factors_233 = find_233(r)
        if factors_233:
            break
        if time.time() > deadline - 30:
            break

    if factors_233:
        U2, V2, W2 = factors_233
        r2 = U2.shape[1]
        # Construct 2x3x9 by repeating 3 times
        r9 = r2 * 3
        U9 = np.zeros((2*3, r9))
        V9 = np.zeros((3*9, r9))
        W9 = np.zeros((9*2, r9))
        
        for block in range(3):
            c_offset = block * r2
            # U is same for all blocks
            U9[:, c_offset:c_offset+r2] = U2
            
            # V: rows are j*p+k. For block 'block', k is in [3*block, 3*block+2]
            # V2 has rows j*3 + k_local.
            # We need to map V2 rows to V9 rows.
            for j in range(3):
                for k_local in range(3):
                    k_global = 3*block + k_local
                    V9[j*9 + k_global, c_offset:c_offset+r2] = V2[j*3 + k_local, :]
            
            # W: rows are k*n+i.
            # W2 has rows k_local*2 + i.
            for k_local in range(3):
                k_global = 3*block + k_local
                for i in range(2):
                    W9[k_global*2 + i, c_offset:c_offset+r2] = W2[k_local*2 + i, :]
        
        if is_valid(U9, V9, W9, r9, 2, 3, 9):
            return {"U": U9.tolist(), "V": V9.tolist(), "W": W9.tolist()}

    # Fallback: Direct search for 2x3x9
    # Or just return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
