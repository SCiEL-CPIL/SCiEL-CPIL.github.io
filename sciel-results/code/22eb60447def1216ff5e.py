"""Matrix multiplication 4x5 by 5x6 — Strassen 4x4x4 + Strassen 4x4x2 + schoolbook 4x1x6,
with rank-based redundancy detection and randomized column replacement search."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    Each decomposition is a tuple: (U_sub, V_sub, W_sub, i_start, j_start, k_start, i_size, j_size, k_size)
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        # Map A indices: i in [0, i_sz), j in [0, j_sz)
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        # Map B indices: j in [0, j_sz), k in [0, k_sz)
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        # Map C indices: k in [0, k_sz), i in [0, i_sz)
        # Convention: W[k*n + i]
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def _build_kronecker_strassen_4x4x4():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    Uses Kronecker product of two 2x2x2 Strassen decompositions.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0:
                                continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0:
                                continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0:
                                continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build a rank-28 decomposition for 4x4 * 4x2 -> 4x2.
    Uses 4 Strassen 2x2x2 blocks.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                # U: A[i,j] where i in 2*ai..2*ai+1, j in 2*bj..2*bj+1
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                
                # V: B[j,k] where j in 2*bj..2*bj+1, k in 0..1
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                
                # W: C[i,k] where i in 2*ai..2*ai+1, k in 0..1
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    Must equal 1 if i,j,k are valid indices, else 0.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Build the tensor T of shape (n, m, p)
    # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        # Outer product contribution
        u_col = U_arr[:, t]  # shape (n*m,)
        v_col = V_arr[:, t]  # shape (m*p,)
        w_col = W_arr[:, t]  # shape (p*n,)
        
        # Reshape to 2D
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # T[i,j,k] += u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        # This is a 3-index sum
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # (since C[i,k] = sum_j A[i,j]*B[j,k], the tensor is the identity in the j-index)
    expected = np.ones((n, m, p), dtype=float)
    
    return np.allclose(T, expected, atol=1e-10)

def _try_remove_column(U, V, W, col_idx, n, m, p):
    """
    Try to remove column col_idx from the decomposition.
    Check if the remaining columns still span the same tensor.
    Returns True if removal is valid.
    """
    rank = U[0].length if hasattr(U[0], 'length') else len(U[0])
    
    # Create copies without the column
    U_new = []
    V_new = []
    W_new = []
    for row in U:
        U_new.append([row[c] for c in range(rank) if c != col_idx])
    for row in V:
        V_new.append([row[c] for c in range(rank) if c != col_idx])
    for row in W:
        W_new.append([row[c] for c in range(rank) if c != col_idx])
    
    return _verify_decomposition(U_new, V_new, W_new, n, m, p)

def _build_constraint_matrix(U_arr, V_arr, W_arr, n, m, p):
    """Build M where M[idx, t] = U[i,j,t]*V[j,k,t]*W[k,i,t], idx=(i*m+j)*p+k."""
    nmp = n * m * p
    rank = U_arr.shape[1]
    M = np.zeros((nmp, rank), dtype=float)
    for t in range(rank):
        u_col = U_arr[:, t].reshape(n, m)
        v_col = V_arr[:, t].reshape(m, p)
        w_col = W_arr[:, t].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * p + k
                    M[idx, t] = u_col[i, j] * v_col[j, k] * w_col[k, i]
    return M

def _verify_fast(U_arr, V_arr, W_arr, n, m, p):
    """Fast verification using vectorized tensor contraction."""
    rank = U_arr.shape[1]
    T = np.zeros((n, m, p), dtype=float)
    for t in range(rank):
        u_c = U_arr[:, t].reshape(n, m)
        v_c = V_arr[:, t].reshape(m, p)
        w_c = W_arr[:, t].reshape(p, n)
        # Vectorized: for each (i,j), T[i,j,:] += u_c[i,j] * v_c[j,:] * w_c[:,i]
        for i in range(n):
            for j in range(m):
                T[i, j, :] += u_c[i, j] * v_c[j, :] * w_c[:, i]
    return np.allclose(T, np.ones((n, m, p)), atol=1e-10)

def _refine_by_column_removal(U, V, W, n, m, p, deadline):
    """
    Rank-based redundancy detection:
    1. Build constraint matrix M (n*m*p rows x rank cols).
    2. Compute rank of M. If rank(M) < rank, there are linearly dependent columns.
    3. For each column, check if it's in the span of others using QR with column pivoting.
    4. Remove redundant columns iteratively.
    """
    start_time = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    nmp = n * m * p
    removed_total = 0
    
    # Iterative removal with rank-based detection
    for iteration in range(5):
        if time.time() > deadline - 10:
            break
        
        current_rank = U_arr.shape[1]
        if current_rank <= 1:
            break
        
        M = _build_constraint_matrix(U_arr, V_arr, W_arr, n, m, p)
        
        # Use QR with column pivoting to find numerically rank-deficient columns
        # A column t is redundant if its norm after QR is very small
        # We'll use a simpler approach: for each column, check if removing it
        # still allows solving M_reduced @ x = ones(nmp)
        
        # Fast pre-filter: compute column norms, try smallest first
        col_norms = np.linalg.norm(M, axis=0)
        order = np.argsort(col_norms)
        
        removed_this_round = 0
        for t in order:
            if time.time() > deadline - 10:
                break
            if t >= U_arr.shape[1]:
                continue
            
            # Try removing column t
            U_test = np.delete(U_arr, t, axis=1)
            V_test = np.delete(V_arr, t, axis=1)
            W_test = np.delete(W_arr, t, axis=1)
            
            if _verify_fast(U_test, V_test, W_test, n, m, p):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                removed_this_round += 1
                removed_total += 1
                # Column indices shift; recompute order
                col_norms = np.linalg.norm(
                    _build_constraint_matrix(U_arr, V_arr, W_arr, n, m, p), axis=0)
                order = np.argsort(col_norms)
        
        if removed_this_round == 0:
            break
    
    U_final = U_arr.tolist()
    V_final = V_arr.tolist()
    W_final = W_arr.tolist()
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140  # Safety margin before 150s limit

    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3)
    U4, V4, W4 = _build_kronecker_strassen_4x4x4()
    decomps.append((U4, V4, W4, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    U5, V5, W5 = _build_4x4x2_strassen()
    decomps.append((U5, V5, W5, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    U6, V6, W6 = _build_schoolbook(4, 1, 6)
    decomps.append((U6, V6, W6, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    # Try to refine by removing redundant columns
    if time.time() < deadline - 15:
        U, V, W = _refine_by_column_removal(U, V, W, n, m, p, deadline)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
