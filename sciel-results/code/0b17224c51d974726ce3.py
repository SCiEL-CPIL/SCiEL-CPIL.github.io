import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    deadline = time.time() + 100
    
    # Try to find a lower rank
    # We can try ranks from a lower bound up to schoolbook - 1
    # Lower bound is max(n*m, m*p, p*n) = 24
    # Let's try a few random initializations for ranks < 72
    
    # To make it fast, we only try a few ranks and iterations
    for target_rank in [30, 40, 50, 60, 70]:
        if time.time() > deadline:
            break
            
        for _ in range(10): # 10 restarts per rank
            if time.time() > deadline:
                break
                
            U = np.random.randint(-2, 3, (n * m, target_rank))
            V = np.random.randint(-2, 3, (m * p, target_rank))
            
            # Solve for W
            # T = sum_c U_c V_c^T W_c^T ? No.
            # T_{(i,j),(j',k'),(i'',k'')} = sum_c U_{(i,j),c} V_{(j',k'),c} W_{(i'',k''),c}
            # Let's reshape U, V to matrices.
            # U is (nm, R), V is (mp, R).
            # The product U_c V_c^T is (nm, mp).
            # We need sum_c (U_c V_c^T) \otimes W_c^T = T_flat.
            
            # Build the linear system for W.
            # Let X_c = U[:, c] V[:, c]^T. Shape (nm, mp).
            # We want sum_c X_c \otimes W[:, c]^T = T_flat.
            # This is a linear system in W.
            
            # T_flat is the tensor flattened.
            # T_{i,j,j',k',i'',k''} = 1 if i=i'', j=j', k=k'' else 0.
            
            # Let's construct the matrix A_system such that A_system * W_flat = T_flat.
            # W_flat is of size (pn * R).
            # For each c, and each (i,j,j',k',i'',k''), the coefficient of W_{i'',k'',c} is U_{i,j,c} V_{j',k',c}.
            
            # This matrix is huge.
            # Dimensions: (nm * mp * pn) x (pn * R)
            # 3*3 * 3*8 * 8*3 = 9 * 24 * 24 = 5184 equations.
            # Variables: 24 * R.
            
            # Let's build the matrix.
            A_sys = np.zeros((n*m*m*p*p*n, p*n*target_rank))
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m): # j'
                        for kp in range(p): # k'
                            for i2 in range(n): # i''
                                for k2 in range(p): # k''
                                    # Coefficient for W_{i2, k2, c} is U_{i,j,c} V_{jp,kp,c}
                                    # Note: j' is the second index of A? No, A is (i,j). B is (j',k').
                                    # The tensor is non-zero if i=i2, j=jp, k=k2.
                                    # Wait, the indices in the prompt:
                                    # U: (n*m, R). Index i*m+j.
                                    # V: (m*p, R). Index j*p+k.
                                    # W: (p*n, R). Index k*n+i.
                                    # T_{(i,j), (j',k'), (i'',k'')} = 1 if i=i'', j=j', k=k''?
                                    # My previous analysis: T = delta_{i,i''} delta_{j,j'} delta_{k,k''}.
                                    # So for a fixed (i,j,jp,kp,i2,k2), the target is 1 if i==i2 and j==jp and k2==kp? No.
                                    # k is the column of B, which is the column of C.
                                    # So k (in V) must match k'' (in W/C).
                                    # And j (in U) must match j' (in V).
                                    # And i (in U) must match i'' (in W/C).
                                    
                                    target_val = 1.0 if (i == i2 and j == jp and kp == k2) else 0.0
                                    
                                    for c in range(target_rank):
                                        coeff = U[i*m+j, c] * V[jp*p+kp, c]
                                        A_sys[idx, k2*n + i2 * target_rank + c] = coeff # W index is k*n+i
                                    idx += 1
                                    
            # Solve A_sys W_flat = T_flat
            # T_flat is just the target values.
            T_flat = np.zeros(n*m*m*p*p*n)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            for i2 in range(n):
                                for k2 in range(p):
                                    T_flat[idx] = 1.0 if (i == i2 and j == jp and kp == k2) else 0.0
                                    idx += 1
                                    
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, T_flat, rcond=None)
                W = W_flat.reshape(target_rank, p*n).T
                # Check if W is close to integer/half-integer
                # And check the error
                err = np.linalg.norm(A_sys @ W_flat - T_flat)
                if err < 1e-5:
                    # Check if entries are close to half-integers
                    W_scaled = W * 2
                    if np.allclose(W_scaled, np.round(W_scaled), atol=1e-3):
                         # We found a solution!
                         # Convert to lists
                         U_list = U.tolist()
                         V_list = V.tolist()
                         W_list = (W/2).tolist() # Wait, W was solved directly.
                         # The problem asks for integer/half-integer entries.
                         # If W is close to half-integers, we can round.
                         best_rank = target_rank
                         best_U, best_V, best_W = U, V, W
                         break
            except:
                pass
                 
    # Return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
