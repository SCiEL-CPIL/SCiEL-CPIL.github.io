"""Rank-minimal tensor decomposition for 3x4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k"""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_222():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) where:
      U: (4, 7) - coefficients for A entries
      V: (4, 7) - coefficients for B entries
      W: (4, 7) - coefficients for C entries, W[k*2+i, c] for C[i,k]
    """
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0
    W[3][0] = 1.0
    
    # M2 = (A10+A11)B00
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0
    
    # M3 = A00(B01-B11)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0
    W[3][2] = 1.0
    
    # M4 = A11(B10-B00)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0
    W[2][3] = 1.0
    
    # M5 = (A00+A01)B11
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0
    W[1][4] = 1.0
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W

def _schoolbook_block(n_block, m_block, p_block):
    """Schoolbook decomposition for an n_block x m_block x p_block subproblem."""
    rank = n_block * m_block * p_block
    U = [[0.0]*rank for _ in range(n_block * m_block)]
    V = [[0.0]*rank for _ in range(m_block * p_block)]
    W = [[0.0]*rank for _ in range(p_block * n_block)]
    col = 0
    for i in range(n_block):
        for j in range(m_block):
            for k in range(p_block):
                U[i*m_block+j][col] = 1.0
                V[j*p_block+k][col] = 1.0
                W[k*n_block+i][col] = 1.0
                col += 1
    return U, V, W, rank

def _combine_blocks(block_decomps, n, m, p, n_splits, m_splits, p_splits):
    """
    Combine block decompositions into a full 3x4x7 decomposition.
    """
    total_rank = sum(
        len(block_decomps[i][j][k][0][0])
        for i in range(len(n_splits))
        for j in range(len(m_splits))
        for k in range(len(p_splits))
    )
    
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                U_blk, V_blk, W_blk = block_decomps[i][j][k]
                blk_rank = len(U_blk[0])
                
                n_offset = sum(n_splits[:i])
                m_offset = sum(m_splits[:j])
                p_offset = sum(p_splits[:k])
                
                n_b = n_splits[i]
                m_b = m_splits[j]
                p_b = p_splits[k]
                
                for r in range(n_b * m_b):
                    local_i = r // m_b
                    local_j = r % m_b
                    full_i = n_offset + local_i
                    full_j = m_offset + local_j
                    full_row = full_i * m + full_j
                    for c in range(blk_rank):
                        U[full_row][rank_offset + c] = U_blk[r][c]
                
                for r in range(m_b * p_b):
                    local_j = r // p_b
                    local_k = r % p_b
                    full_j = m_offset + local_j
                    full_k = p_offset + local_k
                    full_row = full_j * p + full_k
                    for c in range(blk_rank):
                        V[full_row][rank_offset + c] = V_blk[r][c]
                
                for r in range(p_b * n_b):
                    local_k = r // n_b
                    local_i = r % n_b
                    full_k = p_offset + local_k
                    full_i = n_offset + local_i
                    full_row = full_k * n + local_i
                    for c in range(blk_rank):
                        W[full_row][rank_offset + c] = W_blk[r][c]
                
                rank_offset += blk_rank
    
    return U, V, W, total_rank

def _construct_block_decomposition():
    """
    Construct block decomposition for 3x4x7.
    Split: n=[2,1], m=[2,2], p=[2,2,2,1]
    """
    n_splits = [2, 1]
    m_splits = [2, 2]
    p_splits = [2, 2, 2, 1]
    
    block_decomps = [[[None]*len(p_splits) for _ in range(len(m_splits))] for _ in range(len(n_splits))]
    
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                n_b = n_splits[i]
                m_b = m_splits[j]
                p_b = p_splits[k]
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    U_blk, V_blk, W_blk = _strassen_222()
                else:
                    U_blk, V_blk, W_blk, _ = _schoolbook_block(n_b, m_b, p_b)
                
                block_decomps[i][j][k] = (U_blk, V_blk, W_blk)
    
    n, m, p = 3, 4, 7
    U, V, W, total_rank = _combine_blocks(block_decomps, n, m, p, n_splits, m_splits, p_splits)
    
    return U, V, W, total_rank

def _try_remove_columns(U, V, W, n, m, p, max_time=60):
    """
    Try to remove linearly dependent columns from the factor matrices.
    A column c can be removed if there exist coefficients alpha such that
    for all entries, the column c equals the linear combination of other columns.
    """
    start_time = time.time()
    best_rank = len(U[0])
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    
    while time.time() - start_time < max_time:
        r_cur = len(current_U[0])
        if r_cur <= 1:
            break
        
        found_removal = False
        for col_to_remove in range(r_cur):
            if time.time() - start_time > max_time:
                break
            
            other_cols = [c for c in range(r_cur) if c != col_to_remove]
            
            # Build the system: for each entry (i,j,k), we need
            # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # If we remove column col_to_remove, we need to adjust other columns
            # such that the decomposition still holds.
            
            # The constraint is that column col_to_remove must be expressible
            # as a linear combination of the other columns in all three matrices
            # simultaneously, but this is not quite right because the product
            # is bilinear in U and V and linear in W.
            
            # Actually, a column c can be removed if there exist vectors
            # u_c in R^(n*m), v_c in R^(m*p), w_c in R^(p*n) such that
            # the column c of U equals u_c, etc., and the tensor
            # sum_{c'} u_{c'} ⊗ v_{c'} ⊗ w_{c'} (excluding c)
            # still equals the matrix multiplication tensor.
            
            # This is equivalent to saying that the vector
            # u_c ⊗ v_c ⊗ w_c is in the span of the other vectors
            # u_{c'} ⊗ v_{c'} ⊗ w_{c'} in the tensor product space.
            
            # A simpler approach: try to find alpha such that
            # for all (i,j,k): sum_{c' != c} U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
            #   = 1 - U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # This is a system of n*m*p equations in the variables
            # being the adjusted columns. But this is nonlinear.
            
            # Let's use a different approach: check if column c is
            # linearly dependent on the others in a specific sense.
            
            # Build a matrix where each row corresponds to an (i,j,k) triple
            # and each column corresponds to a remaining column c'.
            # Entry (ijk, c') = U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
            # We want to check if the vector b where b[ijk] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # is in the column space of this matrix.
            
            n_eqs = n * m * p
            n_unknowns = len(other_cols)
            
            A_mat = np.zeros((n_eqs, n_unknowns))
            b_vec = np.zeros(n_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ci, c in enumerate(other_cols):
                            A_mat[idx, ci] = current_U[i*m+j][c] * current_V[j*p+k][c] * current_W[k*n+i][c]
                        b_vec[idx] = current_U[i*m+j][col_to_remove] * current_V[j*p+k][col_to_remove] * current_W[k*n+i][col_to_remove]
                        idx += 1
            
            try:
                alpha, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            residual = np.linalg.norm(A_mat @ alpha - b_vec)
            if residual < 1e-8:
                # Remove this column
                new_U = []
                for i in range(len(current_U)):
                    new_row = [current_U[i][c] for c in other_cols]
                    new_U.append(new_row)
                
                new_V = []
                for j in range(len(current_V)):
                    new_row = [current_V[j][c] for c in other_cols]
                    new_V.append(new_row)
                
                new_W = []
                for k in range(len(current_W)):
                    new_row = [current_W[k][c] for c in other_cols]
                    new_W.append(new_row)
                
                if _verify(new_U, new_V, new_W, n, m, p):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    if len(current_U[0]) < best_rank:
                        best_rank = len(current_U[0])
                        best_U = [row[:] for row in current_U]
                        best_V = [row[:] for row in current_V]
                        best_W = [row[:] for row in current_W]
                    found_removal = True
                    break
        
        if not found_removal:
            break
    
    return best_U, best_V, best_W, best_rank

def _random_perturbation_search(U, V, W, n, m, p, max_time=30):
    """
    Try random perturbations of the factor matrices to find lower rank decompositions.
    """
    start_time = time.time()
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    best_rank = len(U[0])
    
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() - start_time > max_time:
            break
        
        # Try to reduce rank by 1 by finding a valid perturbation
        r_cur = len(best_U[0])
        if r_cur <= 1:
            break
        
        # Create a perturbed version with rank r_cur - 1
        # by dropping one random column and adjusting
        col_to_try = np.random.randint(r_cur)
        
        other_cols = [c for c in range(r_cur) if c != col_to_try]
        
        new_U = [[best_U[i][c] for c in other_cols] for i in range(len(best_U))]
        new_V = [[best_V[j][c] for c in other_cols] for j in range(len(best_V))]
        new_W = [[best_W[k][c] for c in other_cols] for k in range(len(best_W))]
        
        # Add small random perturbation
        for i in range(len(new_U)):
            for c in range(len(new_U[0])):
                new_U[i][c] += np.random.normal(0, 0.01)
        for j in range(len(new_V)):
            for c in range(len(new_V[0])):
                new_V[j][c] += np.random.normal(0, 0.01)
        for k in range(len(new_W)):
            for c in range(len(new_W[0])):
                new_W[k][c] += np.random.normal(0, 0.01)
        
        if _verify(new_U, new_V, new_W, n, m, p):
            best_U = new_U
            best_V = new_V
            best_W = new_W
            best_rank = r_cur - 1
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115  # Safety margin before 120s limit
    
    # Build block decomposition
    U_block, V_block, W_block, rank_block = _construct_block_decomposition()
    
    # Verify block decomposition
    if not _verify(U_block, V_block, W_block, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by removing dependent columns
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 10:
        try:
            U_best, V_best, W_best, rank_best = _try_remove_columns(
                U_block, V_block, W_block, n, m, p,
                max_time=min(60, remaining_time - 5)
            )
        except Exception:
            U_best, V_best, W_best = U_block, V_block, W_block
    
    # Final verification
    if not _verify(U_best, V_best, W_best, n, m, p):
        U_best, V_best, W_best = U_block, V_block, W_block
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
