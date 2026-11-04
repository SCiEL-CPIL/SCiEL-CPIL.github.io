"""
Matrix multiplication 6x6x6: rank-189 exact tensor decomposition.

Structure: 6x6 matrices partitioned into 3x3 grid of 2x2 blocks.
Each block product (2x2 * 2x2) uses Strassen's 7-multiplication algorithm.
Total rank = 3 (i_blocks) * 3 (j_blocks) * 3 (k_blocks) * 7 (Strassen) = 189.

This beats schoolbook rank 216 (6*6*6).
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START


def _build_strassen_6x6():
    """
    Build the rank-189 Strassen-based decomposition for 6x6 matrix multiplication.
    
    Conventions:
      A is 6x6: U rows indexed by i*m + j (row i, col j of A)
      B is 6x6: V rows indexed by j*p + k (row j, col k of B)  
      C = A*B: W rows indexed by k*n + i (col k, row i of C) -- trace convention W[k*n+i, r]
    
    For each block triple (I, J, K) in {0,1,2}^3:
      A_block[I,K] is the 2x2 submatrix A[2I:2I+2, 2K:2K+2]
      B_block[K,J] is the 2x2 submatrix B[2K:2K+2, 2J:2J+2]
      C_block[I,J] receives contributions from A_block[I,K] * B_block[K,J]
    
    Strassen's 7 products for 2x2 * 2x2:
      M1 = (a11+a22)*(b11+b22)
      M2 = (a21+a22)*b11
      M3 = a11*(b12-b22)
      M4 = a22*(b21-b11)
      M5 = (a11+a12)*b22
      M6 = (a21-a11)*(b11+b12)
      M7 = (a12-a22)*(b21+b22)
      
      c11 = M1 + M4 - M5 + M7
      c12 = M3 + M5
      c21 = M2 + M4
      c22 = M1 - M2 + M3 + M6
    """
    n = m = p = 6
    rank = 189
    
    # Use numpy arrays for speed
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Helper: U row for A[2I+a, 2K+b]
                # U row index = (2I+a)*m + (2K+b)
                # Helper: V row for B[2K+c, 2J+d]
                # V row index = (2K+c)*p + (2J+d)
                # Helper: W row for C[2I+e, 2J+f]
                # W row index = (2J+f)*n + (2I+e)
                
                def u_idx(a, b):
                    return (2*I + a) * m + (2*K + b)
                
                def v_idx(c, d):
                    return (2*K + c) * p + (2*J + d)
                
                def w_idx(e, f):
                    return (2*J + f) * n + (2*I + e)
                
                # M1 = (a11+a22)*(b11+b22)
                # U: a11=1, a22=1
                U[u_idx(0,0), col] = 1
                U[u_idx(1,1), col] = 1
                # V: b11=1, b22=1
                V[v_idx(0,0), col] = 1
                V[v_idx(1,1), col] = 1
                # W: c11 += 1, c22 += 1
                W[w_idx(0,0), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M2 = (a21+a22)*b11
                U[u_idx(1,0), col] = 1
                U[u_idx(1,1), col] = 1
                V[v_idx(0,0), col] = 1
                # W: c21 += 1, c22 -= 1
                W[w_idx(1,0), col] = 1
                W[w_idx(1,1), col] = -1
                col += 1
                
                # M3 = a11*(b12-b22)
                U[u_idx(0,0), col] = 1
                V[v_idx(0,1), col] = 1
                V[v_idx(1,1), col] = -1
                # W: c12 += 1, c22 += 1
                W[w_idx(0,1), col] = 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M4 = a22*(b21-b11)
                U[u_idx(1,1), col] = 1
                V[v_idx(1,0), col] = 1
                V[v_idx(0,0), col] = -1
                # W: c11 += 1, c21 += 1
                W[w_idx(0,0), col] = 1
                W[w_idx(1,0), col] = 1
                col += 1
                
                # M5 = (a11+a12)*b22
                U[u_idx(0,0), col] = 1
                U[u_idx(0,1), col] = 1
                V[v_idx(1,1), col] = 1
                # W: c11 -= 1, c12 += 1
                W[w_idx(0,0), col] = -1
                W[w_idx(0,1), col] = 1
                col += 1
                
                # M6 = (a21-a11)*(b11+b12)
                U[u_idx(1,0), col] = 1
                U[u_idx(0,0), col] = -1
                V[v_idx(0,0), col] = 1
                V[v_idx(0,1), col] = 1
                # W: c22 += 1
                W[w_idx(1,1), col] = 1
                col += 1
                
                # M7 = (a12-a22)*(b21+b22)
                U[u_idx(0,1), col] = 1
                U[u_idx(1,1), col] = -1
                V[v_idx(1,0), col] = 1
                V[v_idx(1,1), col] = 1
                # W: c11 += 1
                W[w_idx(0,0), col] = 1
                col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=6, m=6, p=6, deadline=140.0):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    Uses vectorized NumPy for speed.
    """
    t0 = time.time()
    rank = U.shape[1]
    
    # Vectorized check: for each (i,j,k), compute sum over r
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # We need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # Efficient approach: iterate over k, vectorize over i and j
    max_err = 0.0
    for k in range(p):
        # W[k*n + i, :] for all i -> shape (n, rank)
        W_k = W[k*n : k*n + n, :]  # (n, rank)
        
        for j in range(m):
            # V[j*p + k, :] -> shape (rank,)
            V_jk = V[j*p + k, :]  # (rank,)
            
            # U[i*m + j, :] for all i -> shape (n, rank)
            U_imj = U[:, j]  # This isn't right, let me fix
            
            # Actually U[i*m+j, r] for fixed j, varying i:
            # row indices: 0*m+j, 1*m+j, ..., (n-1)*m+j
            U_col = U[::m, j]  # (n, rank) - rows 0, m, 2m, ..., (n-1)m, column j
            
            # Product: U_col[i, r] * V_jk[r] * W_k[i, r]
            # = (U_col * W_k) * V_jk element-wise, then sum over r
            prod = (U_col * W_k) * V_jk[np.newaxis, :]  # (n, rank)
            sums = prod.sum(axis=1)  # (n,)
            
            err = np.max(np.abs(sums - 1.0))
            max_err = max(max_err, err)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return max_err


def _schoolbook_fallback(n=6, m=6, p=6):
    """Return the trivial rank-n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix
    by a 6x6 matrix.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists (for JSON compatibility).
    U: shape (n*m, rank) = (36, 189)
    V: shape (m*p, rank) = (36, 189)
    W: shape (p*n, rank) = (36, 189)
    """
    t0 = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # all 6
    
    # Build the Strassen-based rank-189 decomposition
    U, V, W = _build_strassen_6x6()
    
    # Verify with a fast vectorized check
    if time.time() - t0 < DEADLINE - 20.0:
        max_err = _verify_decomposition(U, V, W, n, m, p, deadline=DEADLINE - 5.0)
        
        if max_err > 1e-10:
            # Verification failed - fall back to schoolbook
            U, V, W = _schoolbook_fallback(n, m, p)
    
    # Check deadline
    if time.time() - t0 > DEADLINE:
        # Emergency: return schoolbook immediately
        U, V, W = _schoolbook_fallback(n, m, p)
    
    # Convert numpy arrays to lists of lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
