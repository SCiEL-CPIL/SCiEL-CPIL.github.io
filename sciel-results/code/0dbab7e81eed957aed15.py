"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that U, V, W decompose T."""
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # M[i,j,r] = U[i,r]*V[j,r], then M @ W should give T
    M = U[:, None, :] * V[None, :, :]
    result = M @ W
    return np.allclose(result, T, atol=tol)

def _to_rational_approx(vals, max_den=100):
    """Try to convert float array to rational numbers with small denominators."""
    n = len(vals)
    result = np.zeros(n)
    for i in range(n):
        v = vals[i]
        if abs(v) < 1e-10:
            result[i] = 0.0
            continue
        # Try to find p/q close to v with q <= max_den
        best = None
        best_err = abs(v)
        for q in range(1, max_den+1):
            p = round(v * q)
            if abs(v - p/q) < best_err:
                best_err = abs(v - p/q)
                best = p/q
        if best is not None and best_err < 1e-6:
            result[i] = best
        else:
            result[i] = v
    return result

def _strassen_3x3x3():
    """Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
    U: (9, 7), V: (9, 7), W: (9, 7)
    Indexing: i in 0..2 (row), j in 0..2 (col) for U/V; k in 0..2 for W
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    # Strassen's 7 formulas for 3x3*3x3 -> 3x3
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B12
    # M3 = A11*(B11-B21)
    # M4 = A22*(B22-B12)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 3x3x3: indices (i,j,k) where i=row of A, j=col of A/row of B, k=col of B
    # U[i*3+j, r] = coefficient of A[i,j] in M_r
    # V[j*3+k, r] = coefficient of B[j,k] in M_r  
    # W[k*3+i, r] = coefficient of M_r in C[i,k]
    
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # M1 = (A11+A22)*(B11+B22)
    # A indices: (0,0),(1,1) -> U[0*3+0,0]=1, U[1*3+1,0]=1
    # B indices: (0,0),(1,1) -> V[0*3+0,0]=1, V[1*3+1,0]=1
    # C indices: C[0,0]+=M1, C[1,1]+=M1, C[0,1]-=M1, C[1,0]-=M1
    U[0,0] = 1; U[4,0] = 1
    V[0,0] = 1; V[4,0] = 1
    W[0,0] = 1; W[4,0] = 1; W[3,0] = -1; W[1,0] = -1
    
    # M2 = (A21+A22)*B12
    # A: (1,0),(1,1) -> U[3,1]=1, U[4,1]=1
    # B: (0,1) -> V[1,1]=1
    # C: C[1,1]+=M2, C[1,0]-=M2
    U[3,1] = 1; U[4,1] = 1
    V[1,1] = 1
    W[4,1] = 1; W[3,1] = -1
    
    # M3 = A11*(B11-B21)
    # A: (0,0) -> U[0,2]=1
    # B: (0,0),(1,0) -> V[0,2]=1, V[3,2]=-1
    # C: C[0,0]+=M3, C[0,1]-=M3
    U[0,2] = 1
    V[0,2] = 1; V[3,2] = -1
    W[0,2] = 1; W[1,2] = -1
    
    # M4 = A22*(B22-B12)
    # A: (1,1) -> U[4,3]=1
    # B: (1,1),(0,1) -> V[4,3]=1, V[1,3]=-1
    # C: C[1,1]+=M4, C[1,0]-=M4
    U[4,3] = 1
    V[4,3] = 1; V[1,3] = -1
    W[4,3] = 1; W[3,3] = -1
    
    # M5 = (A11+A12)*B22
    # A: (0,0),(0,1) -> U[0,4]=1, U[1,4]=1
    # B: (1,1) -> V[4,4]=1
    # C: C[0,1]+=M5, C[0,0]-=M5
    U[0,4] = 1; U[1,4] = 1
    V[4,4] = 1
    W[1,4] = 1; W[0,4] = -1
    
    # M6 = (A21-A11)*(B11+B12)
    # A: (1,0),(0,0) -> U[3,5]=1, U[0,5]=-1
    # B: (0,0),(0,1) -> V[0,5]=1, V[1,5]=1
    # C: C[1,0]+=M6, C[1,1]-=M6
    U[3,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1; W[4,5] = -1
    
    # M7 = (A12-A22)*(B21+B22)
    # A: (0,1),(1,1) -> U[1,6]=1, U[4,6]=-1
    # B: (1,0),(1,1) -> V[3,6]=1, V[4,6]=1
    # C: C[0,1]+=M7, C[1,1]-=M7
    U[1,6] = 1; U[4,6] = -1
    V[3,6] = 1; V[4,6] = 1
    W[1,6] = 1; W[4,6] = -1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_tensor()
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    # Try to find lower rank using random search with refinement
    # Start from reasonable ranks
    for r in range(15, rank_sb):
        if time.time() > deadline - 40:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline - 40:
                break
            if found:
                break
                
            rng = np.random.RandomState(seed)
            U = rng.randn(n*m, r)
            V = rng.randn(m*p, r)
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline - 40:
                    break
                    
                # Solve for W: T = U[:,:,None] * V[None,:,:] @ W
                M = U[:, None, :] * V[None, :, :]
                M_flat = M.reshape(n*m * m*p, r)
                T_flat = T.reshape(n*m * m*p, p*n)
                
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=1e-12)
                except:
                    break
                
                # Check if we got a good solution
                residual = np.linalg.norm(M_flat @ W_sol - T_flat)
                if residual < 1e-9:
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W_sol.T.copy()
                    found = True
                    break
                
                # Update U
                W_curr = W_sol.T  # shape (p*n, r)
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed V, W: U[i,r] * sum_{j,k} V[j,r]*W[k,r]*T[i,j,k]... 
                # Actually: T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
                # So U[i,r] should satisfy: for each r, U[i,r] * V[j,r] * W[k,r] contributes to T[i,j,k]
                # This is a tensor equation. Let's do it differently.
                
                # Given V and W, solve for U:
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For each i: T[i,:,:] = sum_r U[i,r] * (V[:,r] * W[:,r])
                # So T[i,:,:].reshape(-1) = U[i,:] @ (V[:,r]*W[:,r] for each r)
                # Let B[j,k,r] = V[j,r]*W[k,r], then T[i,j,k] = sum_r U[i,r]*B[j,k,r]
                # T_i = U_i @ B where T_i is (m*p) vector, U_i is (r,), B is (r, m*p)
                
                B_mat = np.zeros((r, m*p))
                for r_idx in range(r):
                    B_mat[r_idx, :] = (V[:, r_idx] * W_curr[:, :]).reshape(-1)
                    # Wait, W_curr is (p*n, r), so W_curr[:, r_idx] is (p*n,)
                    # We need V[j,r] * W[k*n+i, r] for the (j,k) pair... this is getting complex.
                    # Let me just use the direct approach.
                
                # Simpler: given V and W, the equation is T = U @ (V ⊗ W) in some sense
                # Let's just skip U update and V update for now, only do W update
                # and check convergence.
                
                # Actually let's do proper ALS:
                # Given V, W: solve for U
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # For fixed i: sum_r U[i,r] * (V[:,r] * W[:,r] reshaped) = T[i,:,:]
                # Let's build the matrix properly
                U_mat = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        # T[i,j,k] for k in 0..p-1
                        # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        # So for row (i*m+j): T[i*m+j, :, :] is not right...
                        # T is (n*m, m*p, p*n)
                        # T[i*m+j, j*p+k, k*n+i] = 1 for the tensor
                        # We want: T_flat[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                        
                        # This is a 3D tensor equation. Let me flatten differently.
                        pass
                
                # This is getting complicated. Let me use a different approach:
                # Just do W update and check. If not converging, try different seeds.
                # For a proper ALS, I'd need to handle the 3D structure.
                
                # Alternative: use the fact that we can reshape
                # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
                # U[i,j,r], V[j,k,r], W[k,i,r]
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                
                # Given V, W: for each (i,j), T[i,j,:] = sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
                # V[j,:,r] is vector over k: V[j*p+k, r]
                # W[:,i,r] is vector over k: W[k*n+i, r]
                # So (V[j,:,r] * W[:,i,r])[k] = V[j*p+k,r] * W[k*n+i,r]
                
                # Let's build this properly
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        # For this (i,j), we have p equations (one for each k)
                        # T[i,j,k] = sum_r U[i,j,r] * V[j*p+k, r] * W[k*n+i, r]
                        A_mat = np.zeros((p, r))
                        b_vec = np.zeros(p)
                        for k in range(p):
                            b_vec[k] = T[i*m+j, j*p+k, k*n+i]
                            for r_idx in range(r):
                                A_mat[k, r_idx] = V[j*p+k, r_idx] * W_curr[k*n+i, r_idx]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                            U_new[i*m+j, :] = sol
                        except:
                            pass
                U = U_new
                
                # Given U, W: solve for V
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For each (j,k): T[:,j,k] = sum_r V[j,k,r] * (U[:,j,r] * W[k,:,r])
                # U[:,j,r] is vector over i: U[i*m+j, r]
                # W[k,:,r] is vector over i: W[k*n+i, r]
                
                V_new = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, r))
                        b_vec = np.zeros(n)
                        for i in range(n):
                            b_vec[i] = T[i*m+j, j*p+k, k*n+i]
                            for r_idx in range(r):
                                A_mat[i, r_idx] = U[i*m+j, r_idx] * W_curr[k*n+i, r_idx]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                            V_new[j*p+k, :] = sol
                        except:
                            pass
                V = V_new
                
                # Given U, V: solve for W
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # For each (k,i): T[:,j,k]... wait
                # For each (k,i): sum over j: T[i,j,k] = sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r])
                # U[i,j,r] is vector over j: U[i*m+j, r]
                # V[j,k,r] is vector over j: V[j*p+k, r]
                
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, r))
                        b_vec = np.zeros(m)
                        for j in range(m):
                            b_vec[j] = T[i*m+j, j*p+k, k*n+i]
                            for r_idx in range(r):
                                A_mat[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                        try:
                            sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=1e-12)
                            W_new[k*n+i, :] = sol
                        except:
                            pass
                W_curr = W_new
            
            # Final check after ALS
            if time.time() <= deadline - 40:
                M = U[:, None, :] * V[None, :, :]
                M_flat = M.reshape(n*m * m*p, r)
                T_flat = T.reshape(n*m * m*p, p*n)
                if W_curr is not None:
                    residual = np.linalg.norm(M_flat @ W_curr - T_flat)
                    if residual < 1e-9:
                        best_rank = r
                        best_U, best_V, best_W = U.copy(), V.copy(), W_curr.copy()
                        found = True
                        break
        
        if found:
            break
    
    # Try to clean up to rational values
    if best_rank < rank_sb:
        # Try to find nearby rational solution
        for max_den in [1, 2, 4, 8]:
            U_rat = _to_rational_approx(best_U.flatten(), max_den).reshape(best_U.shape)
            V_rat = _to_rational_approx(best_V.flatten(), max_den).reshape(best_V.shape)
            W_rat = _to_rational_approx(best_W.flatten(), max_den).reshape(best_W.shape)
            if _verify(U_rat, V_rat, W_rat, T, tol=1e-6):
                best_U, best_V, best_W = U_rat, V_rat, W_rat
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END

