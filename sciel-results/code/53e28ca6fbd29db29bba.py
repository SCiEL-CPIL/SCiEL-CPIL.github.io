import time
import numpy as np
from fractions import Fraction
import scipy.optimize
import sys

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank), dtype=object)
    V_school = np.zeros((m * p, schoolbook_rank), dtype=object)
    W_school = np.zeros((p * n, schoolbook_rank), dtype=object)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = Fraction(1)
                V_school[j * p + k, col] = Fraction(1)
                W_school[k * n + i, col] = Fraction(1)
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if s != Fraction(1):
                        return False
        return True

    def snap_to_rational(val, max_den=20):
        """Snap a float to the nearest rational with denominator <= max_den"""
        best_frac = None
        best_err = float('inf')
        for d in range(1, max_den + 1):
            num = round(val * d)
            frac = Fraction(num, d)
            err = abs(float(frac) - val)
            if err < best_err:
                best_err = err
                best_frac = frac
        return best_frac

    def try_als_rank(R, seed, max_iter=100):
        """
        Try to find a rank R decomposition using Alternating Least Squares.
        """
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1

        # Target tensor T
        T = np.ones((n, m, p))

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Update U: For each i,j, solve least squares for U[i*m+j, :]
            # T[i,j,:] = sum_r U[i*m+j, r] * V[j*p: (j+1)*p, r] * W[:, r] (reshaped)
            # This is a linear system for each (i,j).
            # Let's vectorize.
            # We want to minimize || T - U V W ||_F^2
            # Fix V, W. Update U.
            # For a fixed i, j:
            # T[i, j, :] = U[i*m+j, :] @ (V[j, :] * W.T) ? No.
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Let's update U row by row (or block by block).
            # For a fixed i, j, the equation is:
            # T[i, j, :] = sum_r U[i*m+j, r] * (V[j, :] * W[:, r]) ... wait, W depends on i too.
            # W[k*n+i, r] is the r-th component of the vector W[:, i] (if W was p x n x R).
            # Let's reshape W to (p, n, R). Then W[k, i, r] is the entry.
            # Let's reshape V to (m, p, R).
            # Let's reshape U to (n, m, R).
            
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            
            # Update U[i, j, :]:
            # T[i, j, :] = sum_r U[i, j, r] * (V[j, :, r] * W[:, i, r])
            # Let A[i, j, k, r] = V[j, k, r] * W[k, i, r]
            # Then T[i, j, k] = sum_r U[i, j, r] * A[i, j, k, r]
            # This is a linear system for each (i, j).
            
            # We can solve this using np.linalg.lstsq.
            
            # Reshape for ease
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)

            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct matrix A such that T[i,j,:] = U[i,j,:] @ A.T
                    # A[k, r] = V[j, k, r] * W[k, i, r]
                    A = V_3d[j, :, :] * W_3d[:, i, :] # shape (p, R)
                    # T[i, j, :] is shape (p,)
                    # We want U[i, j, :] (shape R,) such that A.T @ U[i, j, :] = T[i, j, :]
                    # Wait, T[i, j, k] = sum_r U[i, j, r] * A[k, r]
                    # So T[i, j, :] = A.T @ U[i, j, :]
                    # Solve A.T @ x = T[i, j, :]
                    x, _, _, _ = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)
                    U_3d[i, j, :] = x

            # Update V
            for j in range(m):
                for k in range(p):
                    # T[:, j, k] = sum_r V[j, k, r] * U[:, j, r] * W[k, :, r]
                    # Let B[i, r] = U[i, j, r] * W[k, i, r]
                    # T[:, j, k] = B.T @ V[j, k, :]
                    B = U_3d[:, j, :] * W_3d[k, :, :] # shape (n, R)
                    x, _, _, _ = np.linalg.lstsq(B.T, T[:, j, k], rcond=None)
                    V_3d[j, k, :] = x

            # Update W
            for k in range(p):
                for i in range(n):
                    # T[i, :, k] = sum_r W[k, i, r] * U[i, :, r] * V[:, k, r]
                    # Let C[j, r] = U[i, j, r] * V[j, k, r]
                    # T[i, :, k] = C.T @ W[k, i, :]
                    C = U_3d[i, :, :] * V_3d[:, k, :] # shape (m, R)
                    x, _, _, _ = np.linalg.lstsq(C.T, T[i, :, k], rcond=None)
                    W_3d[k, i, :] = x

            # Check convergence
            # Calculate error
            T_approx = np.zeros((n, m, p))
            for r in range(R):
                T_approx += np.einsum('ij,jk,ki->ijk', 
                                      U_3d[:, :, r], 
                                      V_3d[:, :, r], 
                                      W_3d[:, :, r])
            err = np.linalg.norm(T - T_approx)
            
            if err < 1e-6:
                # Convert back to 2D
                U_final = U_3d.reshape(n*m, R)
                V_final = V_3d.reshape(m*p, R)
                W_final = W_3d.reshape(p*n, R)
                
                # Snap to rationals
                U_rat = np.zeros((n*m, R), dtype=object)
                V_rat = np.zeros((m*p, R), dtype=object)
                W_rat = np.zeros((p*n, R), dtype=object)
                
                for i in range(n*m):
                    for r in range(R):
                        U_rat[i, r] = snap_to_rational(U_final[i, r])
                for i in range(m*p):
                    for r in range(R):
                        V_rat[i, r] = snap_to_rational(V_final[i, r])
                for i in range(p*n):
                    for r in range(R):
                        W_rat[i, r] = snap_to_rational(W_final[i, r])
                
                if verify_exact(U_rat, V_rat, W_rat, R):
                    return U_rat, V_rat, W_rat, R

        return None

    best_result = None
    
    # Search ranks from low to high
    # Known lower bound is 20.
    for R in range(20, 36):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple seeds
        num_seeds = 50 if R < 25 else 20
        for seed in range(num_seeds):
            if time.time() > deadline - 5:
                break
                
            result = try_als_rank(R, seed)
            if result:
                U_rat, V_rat, W_rat, rank = result
                best_result = (U_rat, V_rat, W_rat, rank)
                found = True
                break
        
        if found:
            break

    if best_result:
        U_rat, V_rat, W_rat, rank = best_result
        return {"U": [[float(x) for x in row] for row in U_rat], 
                "V": [[float(x) for x in row] for row in V_rat], 
                "W": [[float(x) for x in row] for row in W_rat]}
    else:
        return {"U": [[float(x) for x in row] for row in U_school], 
                "V": [[float(x) for x in row] for row in V_school], 
                "W": [[float(x) for x in row] for row in W_school]}
