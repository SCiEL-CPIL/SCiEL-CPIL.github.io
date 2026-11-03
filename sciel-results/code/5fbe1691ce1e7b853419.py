import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [3, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def iu(i, j): return i * m + j
    def iv(j, k): return j * p + k
    def iw(k, i): return k * n + i
    
    def verify(U, V, W):
        R = U.shape[1]
        U_r = U.reshape((n, m, R))
        V_r = V.reshape((m, p, R))
        W_r = W.reshape((p, n, R))
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        return np.allclose(T_recon, T_target, atol=1e-9)
    
    # Build rank-26 decomposition
    R26 = 26
    U26 = np.zeros((n * m, R26))
    V26 = np.zeros((m * p, R26))
    W26 = np.zeros((p * n, R26))
    
    # Strassen's 7 multiplications for 2x2 core
    # m1 = (a00+a11)(b00+b11)
    r = 0
    U26[iu(0,0), r] = 1; U26[iu(1,1), r] = 1
    V26[iv(0,0), r] = 1; V26[iv(1,1), r] = 1
    W26[iw(0,0), r] = 1; W26[iw(1,1), r] = 1
    
    # m2 = (a10+a11)b00
    r = 1
    U26[iu(1,0), r] = 1; U26[iu(1,1), r] = 1
    V26[iv(0,0), r] = 1
    W26[iw(0,1), r] = 1; W26[iw(1,1), r] = -1
    
    # m3 = a00(b01-b11)
    r = 2
    U26[iu(0,0), r] = 1
    V26[iv(0,1), r] = 1; V26[iv(1,1), r] = -1
    W26[iw(1,0), r] = 1; W26[iw(1,1), r] = 1
    
    # m4 = a11(b10-b00)
    r = 3
    U26[iu(1,1), r] = 1
    V26[iv(1,0), r] = 1; V26[iv(0,0), r] = -1
    W26[iw(0,0), r] = 1; W26[iw(0,1), r] = 1
    
    # m5 = (a00+a01)b11
    r = 4
    U26[iu(0,0), r] = 1; U26[iu(0,1), r] = 1
    V26[iv(1,1), r] = 1
    W26[iw(0,0), r] = -1; W26[iw(1,0), r] = 1
    
    # m6 = (a10-a00)(b00+b01)
    r = 5
    U26[iu(1,0), r] = 1; U26[iu(0,0), r] = -1
    V26[iv(0,0), r] = 1; V26[iv(0,1), r] = 1
    W26[iw(1,1), r] = 1
    
    # m7 = (a01-a11)(b10+b11)
    r = 6
    U26[iu(0,1), r] = 1; U26[iu(1,1), r] = -1
    V26[iv(1,0), r] = 1; V26[iv(1,1), r] = 1
    W26[iw(0,0), r] = 1
    
    # Border terms (19 terms)
    col = 7
    border_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[iu(i,j), col] = 1
                V26[iv(j,k), col] = 1
                W26[iw(k,i), col] = 1
                border_indices.append((i, j, k, col))
                col += 1
    
    # Verify rank-26 solution
    if not verify(U26, V26, W26):
        # Fallback to schoolbook 27
        R27 = n * m * p
        U27 = np.zeros((n * m, R27))
        V27 = np.zeros((m * p, R27))
        W27 = np.zeros((p * n, R27))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[iu(i,j), col] = 1
                    V27[iv(j,k), col] = 1
                    W27[iw(k,i), col] = 1
                    col += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    best_rank = 26
    best_result = (U26, V26, W26)
    
    # Try to reduce rank by optimizing: allow Strassen terms to also
    # contribute to border entries, then see if we can drop border terms.
    # 
    # Strategy: For each candidate rank R < 26, try to find U,V,W of 
    # shape (9,R),(9,R),(9,R) such that einsum reproduces the tensor.
    # 
    # Approach: Use iterative optimization. Start from rank-26, try to
    # zero out one column at a time by adjusting remaining columns.
    
    import random
    random.seed(42)
    
    def try_reduce_rank(U_init, V_init, W_init, target_rank, time_limit=30.0):
        """Try to reduce the rank by optimizing remaining columns."""
        R_init = U_init.shape[1]
        if target_rank >= R_init:
            return None
        
        # Select which columns to keep (random subsets)
        n_trials = 50
        best_residual = np.inf
        best_U = None
        best_V = None
        best_W = None
        
        for trial in range(n_trials):
            if time.time() > time_limit:
                break
            
            # Randomly select target_rank columns to keep
            keep_cols = random.sample(range(R_init), target_rank)
            keep_cols.sort()
            
            U_sub = U_init[:, keep_cols].copy()
            V_sub = V_init[:, keep_cols].copy()
            W_sub = W_init[:, keep_cols].copy()
            
            # Now optimize: we want to adjust these columns so they
            # exactly reproduce the tensor. Use iterative method.
            # The tensor equation: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
            # for all (i,j,k) in the support.
            
            # Use alternating least squares / gradient descent
            U_opt = U_sub.copy()
            V_opt = V_sub.copy()
            W_opt = W_sub.copy()
            
            # Iterate: fix two, solve for third
            n_iter = 200
            for it in range(n_iter):
                if time.time() > time_limit:
                    break
                    
                # Solve for U given V, W
                # For each (i,j): U[i,j,:] should satisfy
                # sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r].sum over k... )
                # Actually: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # Fix V,W: for each (i,j), we need:
                # sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r]) contracted with T[i,j,:]
                # Let M[i,j,r] = sum_k V[j,k,r] * W[k,i,r]
                # Then T[i,j,:] = U[i,j,:] @ M[i,j,:]  (dot product over r)
                # So U[i,j,:] = argmin ||U[i,j,:] @ M[i,j,:] - T[i,j,:]||^2
                # This is: U[i,j,:] = (M[i,j,:]^T M[i,j,:])^{-1} M[i,j,:]^T T[i,j,:]
                # But T[i,j,:] is a vector of length p=3, and M is (p,R)
                # Actually M[i,j] is a matrix of shape (p, R)
                # T[i,j,:] is shape (p,)
                # We want U[i,j,:] (shape R) such that M[i,j] @ U[i,j,:] = T[i,j,:]
                # This is underdetermined if R > p, overdetermined if R < p
                # For R=23, p=3: overdetermined, use least squares
                
                for i in range(n):
                    for j in range(m):
                        # M[i,j,k,r] = V[j,k,r] * W[k,i,r] summed over... no
                        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # Let A[i,j,k,r] = V[j,k,r] * W[k,i,r]
                        # T[i,j,:] = sum_r U[i,j,r] * A[i,j,:,r]
                        # So A[i,j] is (p, R) matrix, T[i,j] is (p,) vector
                        # U[i,j,:] solves: A[i,j] @ U[i,j,:] = T[i,j,:]
                        A_ij = np.zeros((p, target_rank))
                        for k in range(p):
                            A_ij[k, :] = V_opt[j * p + k, :] * W_opt[k * n + i, :]
                        T_ij = np.ones(p)  # target
                        # Least squares
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_ij, T_ij, rcond=None)
                            U_opt[i * m + j, :] = sol
                        except:
                            pass
                
                # Solve for V given U, W
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # Fix U,W: for each (j,k), need V[j,k,:] such that
                        # sum_i sum_r U[i,j,r] * W[k,i,r] * V[j,k,r] = T[:,j,k]
                        # Let B[j,k,i,r] = U[i,j,r] * W[k,i,r]
                        # T[:,j,k] = sum_r V[j,k,r] * B[j,k,:,r]
                        # B[j,k] is (n, R) matrix, T[:,j,k] is (n,) vector
                        B_jk = np.zeros((n, target_rank))
                        for i in range(n):
                            B_jk[i, :] = U_opt[i * m + j, :] * W_opt[k * n + i, :]
                        T_jk = np.ones(n)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(B_jk, T_jk, rcond=None)
                            V_opt[j * p + k, :] = sol
                        except:
                            pass
                
                # Solve for W given U, V
                for k in range(p):
                    for i in range(n):
                        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                        # Fix U,V: for each (k,i), need W[k,i,:] such that
                        # sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,:,k]
                        # Let C[k,i,j,r] = U[i,j,r] * V[j,k,r]
                        # T[i,:,k] = sum_r W[k,i,r] * C[k,i,:,r]
                        # C[k,i] is (m, R) matrix, T[i,:,k] is (m,) vector
                        C_ki = np.zeros((m, target_rank))
                        for j in range(m):
                            C_ki[j, :] = U_opt[i * m + j, :] * V_opt[j * p + k, :]
                        T_ki = np.ones(m)
                        try:
                            sol, _, _, _ = np.linalg.lstsq(C_ki, T_ki, rcond=None)
                            W_opt[k * n + i, :] = sol
                        except:
                            pass
            
            # Check residual
            U_r = U_opt.reshape((n, m, target_rank))
            V_r = V_opt.reshape((m, p, target_rank))
            W_r = W_opt.reshape((p, n, target_rank))
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            residual = np.max(np.abs(T_recon - 1.0))
            
            if residual < best_residual:
                best_residual = residual
                best_U = U_opt.copy()
                best_V = V_opt.copy()
                best_W = W_opt.copy()
            
            if residual < 1e-9:
                break
        
        if best_residual < 1e-9:
            return (best_U, best_V, best_W)
        return None
    
    # Try rank 25, 24, 23 in order
    for target_rank in [25, 24, 23]:
        if time.time() > deadline - 10:
            break
        
        print(f"Trying rank {target_rank}...")
        result = try_reduce_rank(U26, V26, W26, target_rank, time_limit=min(40, deadline - time.time() - 10))
        
        if result is not None:
            U_new, V_new, W_new = result
            # Verify
            if verify(U_new, V_new, W_new):
                best_rank = target_rank
                best_result = (U_new, V_new, W_new)
                print(f"Found valid rank-{target_rank} decomposition!")
                break
    
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
