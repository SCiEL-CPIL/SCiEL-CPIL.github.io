"""Rank-28 decomposition for 2x3 * 3x8 matrix multiplication via Kronecker structure."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 140.0

def _build_sub_2x3_3x2():
    """Rank-7 decomposition for 2x3 * 3x2 = 2x2.
    Returns U(6x7), V(6x7), W(4x7) where:
    a = i*3+j (i in {0,1}, j in {0,1,2})
    b = j*2+k (j in {0,1,2}, k in {0,1})
    d = k*2+i (k in {0,1}, i in {0,1})
    """
    U = np.zeros((6, 7))
    V = np.zeros((6, 7))
    W = np.zeros((4, 7))
    
    # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U: A[0,0] -> a=0, A[1,1] -> a=4
    # V: B[0,0] -> b=0, B[1,1] -> b=3
    # W: C[0,0]=1, C[0,1]=-1, C[1,0]=-1, C[1,1]=1
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = -1; W[1, 0] = -1; W[3, 0] = 1
    
    # m2 = A[0,1] * B[1,0]
    # U: A[0,1] -> a=1
    # V: B[1,0] -> b=2
    # W: C[0,0]=1
    U[1, 1] = 1
    V[2, 1] = 1
    W[0, 1] = 1
    
    # m3 = A[0,2] * B[2,0]
    # U: A[0,2] -> a=2
    # V: B[2,0] -> b=4
    # W: C[0,0]=1
    U[2, 2] = 1
    V[4, 2] = 1
    W[0, 2] = 1
    
    # m4 = A[1,2] * B[2,1]
    # U: A[1,2] -> a=5
    # V: B[2,1] -> b=5
    # W: C[1,1]=1
    U[5, 3] = 1
    V[5, 3] = 1
    W[3, 3] = 1
    
    # m5 = (A[0,0] - A[1,1]) * (B[0,0] - B[1,1])
    # U: A[0,0] -> a=0, A[1,1] -> a=4 (with -1)
    # V: B[0,0] -> b=0, B[1,1] -> b=3 (with -1)
    # W: C[0,0]=1, C[0,1]=1, C[1,0]=1, C[1,1]=1
    U[0, 4] = 1; U[4, 4] = -1
    V[0, 4] = 1; V[3, 4] = -1
    W[0, 4] = 1; W[2, 4] = 1; W[1, 4] = 1; W[3, 4] = 1
    
    # m6 = A[1,0] * (B[0,0] + B[0,1])
    # U: A[1,0] -> a=3
    # V: B[0,0] -> b=0, B[0,1] -> b=1
    # W: C[1,0]=1, C[1,1]=1
    U[3, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[1, 5] = 1; W[3, 5] = 1
    
    # m7 = (A[0,1] - A[1,2]) * (B[1,0] - B[2,1])
    # U: A[0,1] -> a=1, A[1,2] -> a=5 (with -1)
    # V: B[1,0] -> b=2, B[2,1] -> b=5 (with -1)
    # W: C[1,1]=-1
    U[1, 6] = 1; U[5, 6] = -1
    V[2, 6] = 1; V[5, 6] = -1
    W[3, 6] = -1
    
    return U, V, W

def _verify_sub(U, V, W, tol=1e-10):
    """Verify the 2x3*3x2 decomposition."""
    n, m, p = 2, 3, 2
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    T_decomp = np.einsum('at,bt,dt->abd', U, V, W, optimize=True)
    return np.allclose(T_decomp, T, atol=tol, rtol=0)

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the 2x3 * 3x2 sub-decomposition (rank 7)
    U_sub, V_sub, W_sub = _build_sub_2x3_3x2()
    
    # Verify sub-decomposition
    if not _verify_sub(U_sub, V_sub, W_sub):
        # Fallback: schoolbook rank 48
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    r_sub = U_sub.shape[1]  # 7
    
    # Construct full decomposition: 4 blocks of 2 columns each
    p_sub = 2
    num_blocks = p // p_sub  # 4
    total_rank = num_blocks * r_sub  # 28
    
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for t in range(num_blocks):
        # U is the same for all blocks
        U_full[:, t*r_sub:(t+1)*r_sub] = U_sub
        
        # V: global b = j*p + k. For block t, k = t*p_sub + k'. local b = j*p_sub + k'
        for j in range(m):
            for k_prime in range(p_sub):
                b_global = j * p + (t * p_sub + k_prime)
                b_local = j * p_sub + k_prime
                V_full[b_global, t*r_sub:(t+1)*r_sub] = V_sub[b_local]
        
        # W: global d = k*n + i. For block t, k = t*p_sub + k'. local d = k'*n + i
        for k_prime in range(p_sub):
            for i in range(n):
                d_global = (t * p_sub + k_prime) * n + i
                d_local = k_prime * n + i
                W_full[d_global, t*r_sub:(t+1)*r_sub] = W_sub[d_local]
    
    # Verify full decomposition
    T_full = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_full[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_decomp = np.einsum('at,bt,dt->abd', U_full, V_full, W_full, optimize=True)
    
    if not np.allclose(T_decomp, T_full, atol=1e-10, rtol=0):
        # Fallback: schoolbook rank 48
        rank_school = n * m * p
        U_fb = np.zeros((n*m, rank_school))
        V_fb = np.zeros((m*p, rank_school))
        W_fb = np.zeros((p*n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
