"""Matrix multiplication 4,4,6 — rank-minimal exact tensor decomposition.

Strategy:
  • Use a computational search (alternating minimization with exactness checking)
    to find a decomposition with rank < 96.
  • The tensor T satisfies: T[i,j,k] = sum_s U[i,j,s]*V[j,k,s]*W[k,i,s]
  • For matrix multiplication, T[i,j,k] = 1 if j is the "inner" index connecting
    A[i,j] and B[j,k] to C[i,k].
  • We search for the minimal R such that the system is solvable.
"""

import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    """Search for a rank-minimal exact tensor decomposition for 4x4x6."""
    t_start = time.time()
    DEADLINE = 150.0  # safety margin
    
    n, m, p = 4, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 16, 24, 24
    
    # The target tensor: T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # This represents: C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = delta (all ones since every (i,j,k) triple is valid)
    
    # We need: for all i,j,k: sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] = 1
    
    # Best known: schoolbook rank = n*m*p = 96
    # Try to find something better via search.
    
    def build_schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return U, V, W, R
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact."""
        n, m, p = 4, 4, 6
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_fast(U_arr, V_arr, W_arr, tol=1e-9):
        """Fast verification using numpy."""
        n, m, p = 4, 4, 6
        R = U_arr.shape[1]
        # U: (16, R), V: (24, R), W: (24, R)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Reshape approach: create tensor and contract
        # U_reshaped: (n, m, R)
        # V_reshaped: (m, p, R)  
        # W_reshaped: (p, n, R)
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        
        # Sum over r: U[i,j,r]*V[j,k,r]*W[k,i,r]
        # = einsum('ijr,jkr,kir->ik', U_r, V_r, W_r) but we need per (i,j,k)
        # Actually we need the full tensor, not contracted
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use broadcasting:
        # U_r: (n,m,R), V_r: (m,p,R), W_r: (p,n,R)
        # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        # = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Check T is all ones
        return np.allclose(T, 1.0, atol=tol)
    
    # Start with schoolbook as baseline
    U_sb, V_sb, W_sb, R_sb = build_schoolbook()
    best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb
    best_score = R_sb
    
    # Check time
    if time.time() - t_start > DEADLINE - 30:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find lower rank decompositions using optimization
    # Approach: for a given rank R, initialize randomly and try to solve
    # the system sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all (i,j,k)
    
    # Use a simple iterative approach:
    # Fix two of {U,V,W}, solve for the third via least squares, repeat.
    
    def try_rank(R, num_restarts=3, max_iter=200):
        """Try to find a rank-R decomposition."""
        nonlocal best_U, best_V, best_W, best_R, best_score
        nonlocal t_start, DEADLINE
        
        for restart in range(num_restarts):
            if time.time() - t_start > DEADLINE - 10:
                return
            
            # Initialize randomly
            U = np.random.randn(nm, R) * 0.5
            V = np.random.randn(mp, R) * 0.5
            W = np.random.randn(pn, R) * 0.5
            
            # Scale to make the product have reasonable magnitude
            # Target: each (i,j,k) should sum to 1
            # Heuristic scaling
            scale = (n * m * p) / (R * (nm * mp * pn) ** 0.5)
            U *= scale
            V *= scale
            W *= scale
            
            for it in range(max_iter):
                if time.time() - t_start > DEADLINE - 10:
                    return
                
                # Solve for U given V, W
                # For each (i,j): sum_r U[i*m+j, r] * V[j*p+?:?]... 
                # Actually: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # For fixed (i,j): sum_k T[i,j,k] is not directly helpful
                # Let's think: for fixed (i,j), we have p equations (one per k):
                #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for k=0..p-1
                # This is linear in U[i*m+j, :]
                # Let's build the system:
                # For each k: dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :]) = 1
                # So: M[i,j] @ U[i*m+j, :]^T = ones(p) where M[i,j][k, r] = V[j*p+k, r]*W[k*n+i, r]
                
                # Update U
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Build M: (p, R) where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.outer(V[j*p:(j+1)*p, :], W[(0*p+i):(p*n+i):n, :].T)
                        # Wait, W is indexed by (k*n+i), so for fixed i, k varies:
                        # W[k*n+i, r] for k=0..p-1
                        W_sub = W[np.array([k*n+i for k in range(p)]), :]  # (p, R)
                        V_sub = V[j*p:(j+1)*p, :]  # (p, R)
                        M = V_sub * W_sub  # (p, R) element-wise
                        # Solve M @ u = ones(p)
                        try:
                            u = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            u = U[i*m+j]
                        U_new[i*m+j, :] = u
                
                # Update V
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # For fixed (j,k): n equations (one per i):
                        # sum_r U[i*m+j, r] * W[k*n+i, r] * V[j*p+k, r] = 1
                        # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        U_sub = U[np.array([i*m+j for i in range(n)]), :]  # (n, R)
                        W_sub = W[np.array([k*n+i for i in range(n)]), :]  # (n, R)
                        M = U_sub * W_sub  # (n, R)
                        try:
                            v = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            v = V[j*p+k]
                        V_new[j*p+k, :] = v
                
                # Update W
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # For fixed (k,i): m equations (one per j):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        U_sub = U[np.array([i*m+j for j in range(m)]), :]  # (m, R)
                        V_sub = V[np.array([j*p+k for j in range(m)]), :]  # (m, R)
                        M = U_sub * V_sub  # (m, R)
                        try:
                            w = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            w = W[k*n+i]
                        W_new[k*n+i, :] = w
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence / exactness
                if it % 20 == 0 or it == max_iter - 1:
                    U_r = U.reshape(n, m, R)
                    V_r = V.reshape(m, p, R)
                    W_r = W.reshape(p, n, R)
                    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    err = np.max(np.abs(T - 1.0))
                    if err < 1e-6:
                        # Found a valid decomposition!
                        # Convert to lists
                        U_list = [[float(U[i, r]) for r in range(R)] for i in range(nm)]
                        V_list = [[float(V[i, r]) for r in range(R)] for i in range(mp)]
                        W_list = [[float(W[i, r]) for r in range(R)] for i in range(pn)]
                        if R < best_R:
                            best_U, best_V, best_W, best_R = U_list, V_list, W_list, R
                            best_score = R
                        return  # Found one for this R
        
    # Try ranks from 95 down to some minimum
    # Known lower bounds: for 4x4x6, rank is at least... 
    # The rank of 4x4x4 is at least 28 (Lader), and 4x4x2 is at least 32/2=16? No.
    # Actually, the rank of the 4x4x6 tensor is at least the max of ranks of sub-tensors.
    # Let's just try a few ranks.
    
    # First, let's try if we can do better than 96 with a moderate search
    # Try R = 95, 90, 85, 80
    
    for R in [95, 90, 88, 85, 80, 78, 75]:
        if time.time() - t_start > DEADLINE - 15:
            break
        try_rank(R, num_restarts=2, max_iter=150)
        if best_R < R_sb:
            # Found something better, try to push further
            continue
    
    # Final verification of best result
    if best_R < R_sb:
        # Double-check
        U_arr = np.array(best_U)
        V_arr = np.array(best_V)
        W_arr = np.array(best_W)
        if verify_fast(U_arr, V_arr, W_arr, tol=1e-6):
            # Valid!
            pass
        else:
            # Fall back to schoolbook
            best_U, best_V, best_W, best_R = U_sb, V_sb, W_sb, R_sb
    
    # Ensure we have a valid result
    U_arr = np.array(best_U)
    V_arr = np.array(best_V)
    W_arr = np.array(best_W)
    if not verify_fast(U_arr, V_arr, W_arr, tol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
