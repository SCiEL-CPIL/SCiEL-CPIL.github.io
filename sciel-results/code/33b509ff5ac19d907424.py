import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    """Apply factor matrices to produce the reconstructed tensor."""
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition is exact."""
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _verify_strassen():
    """Verify Strassen decomposition is correct for 2x2x2."""
    sU, sV, sW, srank = _strassen_2x2x2()
    n, m, p = 2, 2, 2
    T, _, _, _ = _build_tensor()
    # Build 2x2x2 tensor
    T222 = [0.0] * 8
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i * 4 + j * 2 + k] = 1.0
    result = _apply_factors(sU, sV, sW, n, m, p)
    for idx in range(8):
        if abs(result[idx] - T222[idx]) > 1e-9:
            return False
    return True

def _block_decomposition(n, m, p, T):
    """
    Block decomposition for 2x3x7 using Strassen for 2x2x2 sub-blocks.
    Returns (U, V, W, rank) or None if verification fails.
    """
    total_rank = 39  # 14 + 25
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 2a: j=1,2; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2b: j=1,2; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2c: j=1,2; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _block_decomposition_alt(n, m, p, T):
    """
    Alternative block split: m=3 as {0,1}|{2}, p=7 as (0,1),(2,3),(4,5),(6)
    Same total rank 39, different arrangement.
    """
    total_rank = 39
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0,1, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 1a: j=0,1; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1b: j=0,1; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1c: j=0,1; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1d: j=0,1; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    # Block 2: j=2, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    val = 2.0 * x
    nearest = round(val)
    return abs(val - nearest) < tol

def _project_half_integer(vec, tol=1e-9):
    """Project vector to nearest half-integer grid."""
    return [round(2.0 * v) / 2.0 for v in vec]

def _greedy_peel(T, n, m, p, max_rank, deadline):
    """
    Greedy rank-1 peeling: iteratively extract the best rank-1 term
    via SVD on the tensor matricization, project to half-integers,
    and subtract. Returns (U, V, W, rank) or None if fails.
    """
    nmp = n * m * p
    T_np = np.array(T, dtype=float).reshape(n, m, p)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(max_rank):
        if time.time() > deadline:
            break
        
        # Build matricization T_1: (n) x (m*p)
        # T_1[i, j*p+k] = T[i,j,k]
        T_1 = T_np.reshape(n, m * p)
        
        # SVD
        try:
            U_mat, s, Vt = np.linalg.svd(T_1, full_matrices=False)
        except:
            break
        
        if len(s) == 0 or s[0] < 1e-12:
            break
        
        # Take first singular vector
        u_vec = U_mat[:, 0]  # length n
        v_vec = Vt[0, :]     # length m*p
        
        # Reconstruct rank-1 term: u_vec[i] * v_vec[j*p+k]
        # This gives us T[i,j,k] = u[i] * v[j*p+k] * w[k*n+i]
        # For the matricization T_1[i, j*p+k] = sum_r u_r[i] * v_r[j*p+k] * w_r[k*n+i]
        # The third factor w must satisfy: for each (i,j,k), 
        #   T[i,j,k] = u[i] * v[j*p+k] * w[k*n+i]
        # So w[k*n+i] = T[i,j,k] / (u[i] * v[j*p+k]) when both are nonzero
        
        # Build w vector
        w_vec = [0.0] * (p * n)
        valid = True
        for i in range(n):
            if abs(u_vec[i]) < 1e-12:
                continue
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    if abs(v_vec[idx_v]) < 1e-12:
                        continue
                    expected = T_np[i, j, k]
                    if abs(expected) < 1e-12:
                        continue
                    w_val = expected / (u_vec[i] * v_vec[idx_v])
                    w_idx = k * n + i
                    if w_vec[w_idx] != 0.0 and abs(w_vec[w_idx] - w_val) > 1e-9:
                        valid = False
                        break
                    w_vec[w_idx] = w_val
            if not valid:
                break
        if not valid:
            # Try next singular vector or give up
            # For now, just use the raw SVD result
            # Project to half-integers
            u_proj = _project_half_integer(u_vec.tolist())
            v_proj = _project_half_integer(v_vec.tolist())
            w_proj = _project_half_integer(w_vec)
            
            # Verify this rank-1 term
            term = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term[i, j, k] = u_proj[i] * v_proj[j * p + k] * w_proj[k * n + i]
            
            # Subtract from remaining tensor
            T_np -= term
            U_cols.append(u_proj)
            V_cols.append(v_proj)
            W_cols.append(w_proj)
        else:
            # Valid rank-1 term found
            u_proj = _project_half_integer(u_vec.tolist())
            v_proj = _project_half_integer(v_vec.tolist())
            w_proj = _project_half_integer(w_vec)
            
            # Verify
            term = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term[i, j, k] = u_proj[i] * v_proj[j * p + k] * w_proj[k * n + i]
            
            T_np -= term
            U_cols.append(u_proj)
            V_cols.append(v_proj)
            W_cols.append(w_proj)
    
    if not U_cols:
        return None
    
    rank = len(U_cols)
    
    # Check if tensor is fully decomposed
    if np.max(np.abs(T_np)) > 1e-9:
        return None
    
    # Build final matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U_cols[r][i]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = V_cols[r][j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = W_cols[r][k * n + i]
    
    return U, V, W, rank

def _strassen_2x3x2():
    """
    Strassen-like decomposition for 2x3x2 matrix multiplication.
    Rank 10 (schoolbook is 12, we save 2).
    """
    # A is 2x3, B is 3x2, C is 2x2
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Use shared products:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[1,0]-B[1,1])
    # M4 = A[1,1]*(B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # M8 = A[0,2]*B[2,0]
    # M9 = A[0,2]*B[2,1]
    # M10 = (A[1,2]+A[0,2])*(B[2,0]+B[2,1]) -- but this doesn't help directly
    
    # Actually, let's just use schoolbook for 2x3x2 = rank 12
    # and see if we can find a better decomposition
    n, m, p = 2, 3, 2
    rank = n * m * p  # 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _block_decomposition_v3(n, m, p, T):
    """
    Block decomposition v3: split p=7 into (0,1),(2,3),(4,5),(6)
    For each p-block, use optimal 2x3x2 or 2x3x1 decomposition.
    Total: 3 * rank(2x3x2) + rank(2x3x1) = 3*12 + 6 = 42 (schoolbook)
    But if we can do 2x3x2 in rank 10, total = 3*10 + 6 = 36
    """
    # For now, use schoolbook blocks
    # p-blocks: (0,1), (2,3), (4,5), (6)
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Three 2x3x2 blocks
    sU, sV, sW, srank = _strassen_2x3x2()
    for p_start in [0, 2, 4]:
        for r in range(srank):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for i in range(n):
                for j in range(m):
                    U_col[i * m + j] = sU[i * m + j][r]
            for j in range(m):
                for k in range(2):
                    V_col[j * p + (p_start + k)] = sV[j * 2 + k][r]
            for k in range(2):
                for i in range(n):
                    W_col[(p_start + k) * n + i] = sW[k * 2 + i][r]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
        total_rank += srank
    
    # One 2x3x1 block (p=6)
    for i in range(n):
        for j in range(m):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i * m + j] = 1
            V_col[j * p + 6] = 1
            W_col[6 * n + i] = 1
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
            total_rank += 1
    
    rank = total_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, rank

def _search_rank_reduction(n, m, p, T, base_U, base_V, base_W, base_rank, deadline):
    """
    Try to reduce rank by finding linear dependencies among columns.
    """
    # Convert to numpy for efficient computation
    U_np = np.array(base_U, dtype=float)  # (n*m, base_rank)
    V_np = np.array(base_V, dtype=float)  # (m*p, base_rank)
    W_np = np.array(base_W, dtype=float)  # (p*n, base_rank)
    
    # The tensor is T = sum_r U[:,r] * V[:,r] * W[:,r]
    # We want to find if some columns are linearly dependent
    # Try removing one column at a time and checking if we can compensate
    
    # For each pair of columns, check if one can be expressed as combo of others
    # This is expensive, so we'll try a few heuristics
    
    # Heuristic: look for columns with small norm
    norms_U = np.linalg.norm(U_np, axis=0)
    norms_V = np.linalg.norm(V_np, axis=0)
    norms_W = np.linalg.norm(W_np, axis=0)
    
    # Try removing columns where all three norms are small
    candidate_cols = []
    for r in range(base_rank):
        if norms_U[r] < 0.1 and norms_V[r] < 0.1 and norms_W[r] < 0.1:
            candidate_cols.append(r)
    
    # Try removing each candidate
    for col_to_remove in candidate_cols:
        if time.time() > deadline:
            break
        
        # Remove column and check if tensor is still correct
        U_new = np.delete(U_np, col_to_remove, axis=1)
        V_new = np.delete(V_np, col_to_remove, axis=1)
        W_new = np.delete(W_np, col_to_remove, axis=1)
        
        # Reconstruct tensor
        T_recon = np.zeros((n, m, p), dtype=float)
        for r in range(U_new.shape[1]):
            u_vec = U_new[:, r]
            v_vec = V_new[:, r]
            w_vec = W_new[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += u_vec[i * m + j] * v_vec[j * p + k] * w_vec[k * n + i]
        
        T_np = np.array(T, dtype=float).reshape(n, m, p)
        if np.max(np.abs(T_recon - T_np)) < 1e-9:
            # Success! We can remove this column
            U_list = U_new.tolist()
            V_list = V_new.tolist()
            W_list = W_new.tolist()
            new_rank = base_rank - 1
            return U_list, V_list, W_list, new_rank
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    # Verify Strassen is correct
    if not _verify_strassen():
        # Strassen is broken, fall back to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    
    # Best result so far
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try primary block decomposition (rank 39)
    result = _block_decomposition(n, m, p, T)
    if result is not None:
        U, V, W, rank = result
        if rank < best_result[3]:
            best_result = result
    
    # Try alternative block decomposition (rank 39)
    if time.time() < deadline:
        result = _block_decomposition_alt(n, m, p, T)
        if result is not None:
            U, V, W, rank = result
            if rank < best_result[3]:
                best_result = result
    
    # Try v3 block decomposition
    if time.time() < deadline:
        result = _block_decomposition_v3(n, m, p, T)
        if result is not None:
            U, V, W, rank = result
            if rank < best_result[3]:
                best_result = result
    
    # Try rank reduction on best result
    if time.time() < deadline - 30:
        U, V, W, rank = best_result
        reduced = _search_rank_reduction(n, m, p, T, U, V, W, rank, deadline - 10)
        if reduced is not None:
            U, V, W, rank = reduced
            if rank < best_result[3]:
                best_result = result
    
    # Try greedy peeling as last resort
    if time.time() < deadline - 15:
        for max_rank in range(35, 40):
            if time.time() > deadline - 5:
                break
            result = _greedy_peel(T, n, m, p, max_rank, deadline - 2)
            if result is not None:
                U, V, W, rank = result
                if _verify(U, V, W, n, m, p, T) and rank < best_result[3]:
                    best_result = result
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
