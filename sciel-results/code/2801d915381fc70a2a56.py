"""Matrix multiplication 4x5 by 5x6 — Hybrid Strassen/schoolbook decomposition with rank reduction."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

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

def _reduce_rank(U, V, W, n, m, p):
    """
    Attempt to reduce the rank of the decomposition by identifying linearly
    dependent rank-1 terms. Each rank-1 term is a vector in Q^(n*m*p).
    We compute the column space rank and keep only a basis.
    Returns (U_reduced, V_reduced, W_reduced) with potentially lower rank.
    """
    start_time = time.time()
    deadline = 130  # safety margin
    
    rank = len(U[0])
    dim = n * m * p  # 120
    
    # Build the matrix M of shape (dim, rank) where column t is the
    # vectorization of the rank-1 term U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # But we don't need to build it explicitly; we can do Gaussian elimination
    # on the columns.
    
    # Convert to Fractions for exact arithmetic
    # Each column t: for each (i,j,k) in [0,n)x[0,m)x[0,p),
    #   M[i*m*p + j*p + k, t] = U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # To save time, let's first do a quick check with floating point
    # to see if rank reduction is even possible
    M_float = np.zeros((dim, rank))
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][t]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][t]
                    if w_val == 0:
                        continue
                    M_float[i*m*p + j*p + k, t] = u_val * v_val * w_val
    
    # Compute rank via SVD (fast)
    try:
        _, s, _ = np.linalg.svd(M_float, full_matrices=False)
        tol = 1e-10 * max(s) if len(s) > 0 else 1e-10
        num_independent = sum(1 for sv in s if sv > tol)
    except Exception:
        num_independent = rank
    
    if num_independent >= rank:
        # No reduction possible
        return U, V, W
    
    # Now do exact rank computation to find which columns to keep
    # Use Gaussian elimination over Q
    M_exact = [[Fraction(0) for _ in range(rank)] for _ in range(dim)]
    
    for t in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i*m + j][t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p + k][t]
                    if v_val == 0:
                        continue
                    w_val = W[k*n + i][t]
                    if w_val == 0:
                        continue
                    M_exact[i*m*p + j*p + k][t] = Fraction(u_val) * Fraction(v_val) * Fraction(w_val)
    
    # Check time
    if time.time() - start_time > (deadline - 30):
        return U, V, W
    
    # Gaussian elimination to find pivot columns
    # We want to find a set of pivot columns that form a basis
    # Work on the transpose conceptually: find which columns are independent
    
    # Perform row-reduction style: we'll track which columns are pivots
    pivot_cols = []
    pivot_rows = []
    # We'll do elimination on the matrix, tracking column pivots
    # Standard approach: for each column, check if it adds to the span
    
    # Better: do Gaussian elimination on the matrix M (dim x rank)
    # to find the rank and pivot columns
    
    # Copy M for elimination
    M_work = [row[:] for row in M_exact]
    
    col_pivots = []
    current_row = 0
    
    for col in range(rank):
        if time.time() - start_time > (deadline - 10):
            # Time's up, return original
            return U, V, W
        
        # Find pivot in column `col` from row `current_row` down
        pivot = None
        for r in range(current_row, dim):
            if M_work[r][col] != 0:
                pivot = r
                break
        
        if pivot is None:
            continue
        
        # Swap rows
        M_work[current_row], M_work[pivot] = M_work[pivot], M_work[current_row]
        
        # Normalize
        pivot_val = M_work[current_row][col]
        if pivot_val != 1:
            for c in range(col, rank):
                M_work[current_row][c] /= pivot_val
        
        # Eliminate
        for r in range(current_row + 1, dim):
            factor = M_work[r][col]
            if factor != 0:
                for c in range(col, rank):
                    M_work[r][c] -= factor * M_work[current_row][c]
        
        col_pivots.append(col)
        pivot_rows.append(current_row)
        current_row += 1
        
        if current_row >= dim:
            break
    
    # col_pivots contains the indices of independent columns
    kept_cols = col_pivots
    new_rank = len(kept_cols)
    
    if new_rank >= rank:
        return U, V, W
    
    # Build reduced matrices
    U_new = [[0] * new_rank for _ in range(n * m)]
    V_new = [[0] * new_rank for _ in range(m * p)]
    W_new = [[0] * new_rank for _ in range(p * n)]
    
    for new_idx, old_idx in enumerate(kept_cols):
        for r in range(n * m):
            U_new[r][new_idx] = U[r][old_idx]
        for r in range(m * p):
            V_new[r][new_idx] = V[r][old_idx]
        for r in range(p * n):
            W_new[r][new_idx] = W[r][old_idx]
    
    return U_new, V_new, W_new

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = 140

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
    
    # Attempt rank reduction
    if time.time() - start_time < deadline - 40:
        U, V, W = _reduce_rank(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
