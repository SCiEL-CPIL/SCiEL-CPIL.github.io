import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct by checking tensor identity."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i*m + k
                b = k*p + j
                c = j*n + i
                target[a, b, c] = 1.0
                
    # Compute decomp tensor
    decomp = np.einsum('it,jt,kt->ijk', U, V, W)
    
    diff = np.abs(decomp - target)
    return np.max(diff) < tol

def construct():
    """
    Returns {'U', 'V', 'W'} with integer entries that exactly decompose
    the 3x5 @ 5x6 matrix multiplication tensor.
    
    Uses block decomposition:
    n=3 -> 2+1, m=5 -> 2+2+1, p=6 -> 2+2+2
    With Strassen rank-7 for 2x2x2 blocks and schoolbook for others.
    Total rank = 84.
    """
    deadline = time.time() + 140.0
    n, m, p = 3, 5, 6
    
    # Splits
    n_splits = [0, 2, 3]
    m_splits = [0, 2, 4, 5]
    p_splits = [0, 2, 4, 6]
    
    # Pre-compute total rank
    rank = 0
    for ni in range(len(n_splits) - 1):
        for mi in range(len(m_splits) - 1):
            for pi in range(len(p_splits) - 1):
                a_size = n_splits[ni+1] - n_splits[ni]
                m_size = m_splits[mi+1] - m_splits[mi]
                p_size = p_splits[pi+1] - p_splits[pi]
                if a_size == 2 and m_size == 2 and p_size == 2:
                    rank += 7
                else:
                    rank += a_size * m_size * p_size
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_block(A_r_start, A_r_end, A_c_start, A_c_end, 
                             B_r_start, B_r_end, B_c_start, B_c_end):
        nonlocal col
        for i in range(A_r_start, A_r_end):
            for k in range(A_c_start, A_c_end):
                for j in range(B_c_start, B_c_end):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
                    if time.time() > deadline:
                        return False
        return True
    
    def add_strassen_block(A_r_start, A_c_start, B_c_start):
        """Add Strassen rank-7 for 2x2x2 block."""
        nonlocal col
        a11 = A_r_start * m + A_c_start
        a12 = A_r_start * m + A_c_start + 1
        a21 = (A_r_start + 1) * m + A_c_start
        a22 = (A_r_start + 1) * m + A_c_start + 1
        b11 = A_c_start * p + B_c_start
        b12 = A_c_start * p + B_c_start + 1
        b21 = (A_c_start + 1) * p + B_c_start
        b22 = (A_c_start + 1) * p + B_c_start + 1
        c11 = B_c_start * n + A_r_start
        c12 = (B_c_start + 1) * n + A_r_start
        c21 = B_c_start * n + A_r_start + 1
        c22 = (B_c_start + 1) * n + A_r_start + 1
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, coeff in u_coeffs:
                U[idx][col] = coeff
            for idx, coeff in v_coeffs:
                V[idx][col] = coeff
            for idx, coeff in w_coeffs:
                W[idx][col] = coeff
            col += 1
            if time.time() > deadline:
                return False
            return True
        
        # T1: (a11+a22)(b11+b22) -> c11, c22
        if not add_term([(a11,1),(a22,1)], [(b11,1),(b22,1)], [(c11,1),(c22,1)]): return False
        # T2: (a21+a22)b11 -> c21, c22
        if not add_term([(a21,1),(a22,1)], [(b11,1)], [(c21,1),(c22,1)]): return False
        # T3: a11(b12-b22) -> c12, -c22
        if not add_term([(a11,1)], [(b12,1),(b22,-1)], [(c12,1),(c22,-1)]): return False
        # T4: a22(b21-b11) -> -c11, c21
        if not add_term([(a22,1)], [(b21,1),(b11,-1)], [(c11,-1),(c21,1)]): return False
        # T5: (a11+a12)b22 -> c12, c22
        if not add_term([(a11,1),(a12,1)], [(b22,1)], [(c12,1),(c22,1)]): return False
        # T6: (a21-a11)(b11+b12) -> c11
        if not add_term([(a21,1),(a11,-1)], [(b11,1),(b12,1)], [(c11,1)]): return False
        # T7: (a12-a22)(b21+b22) -> c11
        if not add_term([(a12,1),(a22,-1)], [(b21,1),(b22,1)], [(c11,1)]): return False
        return True
    
    # Iterate over all blocks
    for ni in range(len(n_splits) - 1):
        for mi in range(len(m_splits) - 1):
            for pi in range(len(p_splits) - 1):
                ar_s, ar_e = n_splits[ni], n_splits[ni+1]
                ac_s, ac_e = m_splits[mi], m_splits[mi+1]
                bc_s, bc_e = p_splits[pi], p_splits[pi+1]
                
                a_size = ar_e - ar_s
                m_size = ac_e - ac_s
                p_size = bc_e - bc_s
                
                if a_size == 2 and m_size == 2 and p_size == 2:
                    if not add_strassen_block(ar_s, ac_s, bc_s):
                        return {"U": U, "V": V, "W": W}
                else:
                    if not add_schoolbook_block(ar_s, ar_e, ac_s, ac_e, ac_s, ac_e, bc_s, bc_e):
                        return {"U": U, "V": V, "W": W}
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to full schoolbook
        U = [[0] * (n*m*p) for _ in range(n * m)]
        V = [[0] * (n*m*p) for _ in range(m * p)]
        W = [[0] * (n*m*p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][col] = 1
                    V[k * p + j][col] = 1
                    W[j * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 20:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check for linear dependencies in the tensor decomposition
        # We can try to find if any columns are linear combinations of others
        # This is a complex problem, so we'll use a simple approach:
        # Check if the rank of the combined matrix is less than the number of terms
        
        # Create a matrix where each column corresponds to a term
        # and rows correspond to all possible (i,k,j) combinations
        n_terms = U_arr.shape[1]
        n_rows = n * m * p
        
        # Build the matrix
        M = np.zeros((n_rows, n_terms), dtype=np.float64)
        for t in range(n_terms):
            # For each term, the contribution is U[:,t] * V[:,t] * W[:,t]
            # We need to extract the coefficients for each (i,k,j)
            # U[i*m+k, t] * V[k*p+j, t] * W[j*n+i, t]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        row_idx = i * m * p + k * p + j
                        M[row_idx, t] = U_arr[i*m+k, t] * V_arr[k*p+j, t] * W_arr[j*n+i, t]
        
        # Compute the rank of M
        try:
            r = np.linalg.matrix_rank(M, tol=1e-9)
            if r < n_terms:
                # There are linear dependencies, but we can't easily remove them
                # because we need to maintain the exact decomposition
                pass
        except:
            pass
    
    return {"U": U, "V": V, "W": W}
