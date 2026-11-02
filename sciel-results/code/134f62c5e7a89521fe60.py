"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
from itertools import product
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Build the tensor
    tensor = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find lower rank decompositions
    # We'll search for ranks from 29 down to 15
    # Using a simple ALS (Alternating Least Squares) approach
    # with rational rounding
    
    def try_find_decomposition(target_rank):
        """
        Attempts to find a rank-target_rank decomposition using ALS.
        Returns (U, V, W) as lists of lists of Fractions if successful, else None.
        """
        # Initialize U, V, W randomly
        # We can also try initializing with random subsets of schoolbook terms
        for attempt in range(10):  # 10 random restarts
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # ALS iterations
            for it in range(100):
                # Solve for U
                # T = sum U[:,c] (V[:,c] (W[:,c]))
                # Flatten T to (n*m, m*p*p*n)
                # T_flat_u[a, b, c]
                # We want U such that T = U (V^T (W^T)) ?
                # Actually, T[a,b,c] = sum_c U[a,c] V[b,c] W[c,c]
                # Fix V, W.
                # T[a, b, c] = sum_c U[a,c] (V[b,c] W[c,c])
                # Let M_u[a, (b,c)] = T[a,b,c]
                # Let N_u[(b,c), c'] = sum_{b,c} V[b,c'] W[c,c'] ? No.
                # The equation is linear in U.
                # T[a, b, c] = sum_{c'} U[a, c'] V[b, c'] W[c, c']
                # This is not a standard linear system in U if V and W are fixed,
                # because the index c' is summed over, but b and c are fixed indices of T.
                # Actually, for each (b,c), we have a vector equation for U[:, :].
                # T[:, b, c] = sum_{c'} U[:, c'] * (V[b, c'] W[c, c'])
                # Let vec = V[b, :] * W[c, :] (element-wise product).
                # T[:, b, c] = U @ vec
                # So we have a system:
                # [ T[:, b1, c1], T[:, b2, c2], ... ] = U @ [ vec1, vec2, ... ]
                # This is a least squares problem: U @ X = Y
                # Where X is (target_rank, m*p*p*n) and Y is (n*m, m*p*p*n)
                
                X_u = np.zeros((target_rank, m * p * p * n))
                Y_u = np.zeros((n * m, m * p * p * n))
                idx = 0
                for b in range(m * p):
                    for c in range(p * n):
                        vec = V[:, b] * W[:, c]  # Wait, V is (m*p, rank), W is (p*n, rank)
                        # V[b, c'] and W[c, c']
                        # vec[c'] = V[b, c'] * W[c, c']
                        X_u[:, idx] = vec
                        Y_u[:, idx] = tensor[:, b, c]
                        idx += 1
                
                # Solve U @ X_u = Y_u for U
                # U = Y_u @ X_u^T @ (X_u @ X_u^T)^-1
                try:
                    U_new, residuals, rank_u, s = np.linalg.lstsq(X_u.T, Y_u.T, rcond=None)
                    U_new = U_new.T
                except:
                    continue
                
                # Solve for V
                X_v = np.zeros((target_rank, n * m * p * n))
                Y_v = np.zeros((m * p, n * m * p * n))
                idx = 0
                for a in range(n * m):
                    for c in range(p * n):
                        vec = U[:, a] * W[:, c]
                        X_v[:, idx] = vec
                        Y_v[:, idx] = tensor[a, :, c]
                        idx += 1
                try:
                    V_new, residuals, rank_v, s = np.linalg.lstsq(X_v.T, Y_v.T, rcond=None)
                    V_new = V_new.T
                except:
                    continue

                # Solve for W
                X_w = np.zeros((target_rank, n * m * m * p))
                Y_w = np.zeros((p * n, n * m * m * p))
                idx = 0
                for a in range(n * m):
                    for b in range(m * p):
                        vec = U[:, a] * V[:, b]
                        X_w[:, idx] = vec
                        Y_w[:, idx] = tensor[a, b, :]
                        idx += 1
                try:
                    W_new, residuals, rank_w, s = np.linalg.lstsq(X_w.T, Y_w.T, rcond=None)
                    W_new = W_new.T
                except:
                    continue

                U, V, W = U_new, V_new, W_new
                
                # Check error
                err = 0.0
                for c in range(target_rank):
                    term = np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                    err += np.sum((tensor - term)**2)
                    # Actually, we should accumulate the full tensor
                # Better: compute full reconstruction
                T_recon = np.zeros_like(tensor)
                for c in range(target_rank):
                    T_recon += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                err = np.linalg.norm(tensor - T_recon)
                
                if err < 1e-9:
                    # Try to round to rationals
                    # Scale to integers?
                    # The entries might be small.
                    # Try denominators 1, 2, 4, 8
                    for denom in [1, 2, 4, 8, 16]:
                        U_frac = [[Fraction(int(round(u * denom)), denom) for u in U[:, c]] for c in range(target_rank)]
                        # Transpose to match expected format (n*m, rank)
                        U_list = [[0]*target_rank for _ in range(n*m)]
                        V_list = [[0]*target_rank for _ in range(m*p)]
                        W_list = [[0]*target_rank for _ in range(p*n)]
                        
                        for c in range(target_rank):
                            for i in range(n*m):
                                U_list[i][c] = Fraction(int(round(U[i, c] * denom)), denom)
                            for j in range(m*p):
                                V_list[j][c] = Fraction(int(round(V[j, c] * denom)), denom)
                            for k in range(p*n):
                                W_list[k][c] = Fraction(int(round(W[k, c] * denom)), denom)
                        
                        # Verify
                        T_verify = np.zeros((n*m, m*p, p*n), dtype=object)
                        for c in range(target_rank):
                            for i in range(n*m):
                                for j in range(m*p):
                                    for k in range(p*n):
                                        T_verify[i,j,k] += U_list[i][c] * V_list[j][c] * W_list[k][c]
                        
                        is_exact = True
                        for i in range(n*m):
                            for j in range(m*p):
                                for k in range(p*n):
                                    if T_verify[i,j,k] != tensor[i,j,k]:
                                        is_exact = False
                                        break
                                if not is_exact: break
                            if not is_exact: break
                        if is_exact:
                            return U_list, V_list, W_list
            # End attempts
        return None

    # Search for lower ranks
    # Start from 29 down to 15
    for r in range(29, 14, -1):
        result = try_find_decomposition(r)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
