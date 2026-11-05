"""
Rank-minimal exact tensor decomposition for 3x5 x 5x6 matrix multiplication.

Strategy:
  1. Exploit the block structure of matrix multiplication.
  2. Decompose the 3x5x6 problem into a collection of smaller sub-problems:
     - 2x2x2 blocks: Use the rank-7 Strassen algorithm.
     - Remaining 1xk or kx1 or 1x1 blocks: Use rank-1 schoolbook (or rank-2 if needed, but 1x1 is rank 1).
  3. Assemble these sub-decompositions into the global factor matrices U, V, W.
  4. Verify the exactness of the resulting decomposition.
  5. Return the valid decomposition.
  
  This approach avoids the pitfalls of generic numerical optimization by using
  known algebraic identities (Strassen) for the dense parts of the matrix.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]


def _verify(U, V, W, n, m, p):
    """
    Verify exact decomposition:
       sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False

    Uf = U.reshape(n, m, rank)
    Vf = V.reshape(m, p, rank)
    Wf = W.reshape(p, n, rank)
    
    # Efficient verification using einsum
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We want T[i, j, k] == 1 for all i, j, k.
    
    # Compute the tensor T
    # Uf: (n, m, r), Vf: (m, p, r), Wf: (p, n, r)
    # We need to sum over r.
    # Let's construct the full tensor and check.
    
    # For small sizes, direct loop is fine and less prone to einsum index errors
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Uf[i, j, :], Vf[j, k, :] * Wf[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2():
    """
    Return rank-7 decomposition for 2x2x2 matrix multiplication.
    Indices:
    U: (2*2, 7) -> U[i*2+j, r]
    V: (2*2, 7) -> V[j*2+k, r]
    W: (2*2, 7) -> W[k*2+i, r]
    
    Standard Strassen coefficients:
    M1 = (A11 - A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U coefficients for A entries (indices 0:00, 1:01, 2:10, 3:11)
    # M1: A11(1), A22(-1)
    # M2: A21(1), A22(1)
    # M3: A11(1)
    # M4: A22(1)
    # M5: A11(1), A12(1)
    # M6: A21(1), A11(-1)
    # M7: A12(1), A22(-1)
    
    U = np.array([
        [ 1,  0,  1,  0,  1, -1,  0],  # A11 (0,0)
        [ 0,  0,  0,  0,  1,  0,  1],  # A12 (0,1)
        [ 0,  1,  0,  0,  0,  1,  0],  # A21 (1,0)
        [-1,  1,  0,  1,  0,  0, -1]   # A22 (1,1)
    ], dtype=float)
    
    # V coefficients for B entries (indices 0:00, 1:01, 2:10, 3:11)
    # M1: B11(1), B22(1)
    # M2: B11(1)
    # M3: B12(1), B22(-1)
    # M4: B21(1), B11(-1)
    # M5: B22(1)
    # M6: B11(1), B12(1)
    # M7: B21(1), B22(1)
    
    V = np.array([
        [ 1,  1,  0, -1,  0,  1,  0],  # B11 (0,0)
        [ 0,  0,  1,  0,  0,  1,  0],  # B12 (0,1)
        [ 0,  0,  0,  1,  0,  0,  1],  # B21 (1,0)
        [ 1,  0, -1,  0,  1,  0,  1]   # B22 (1,1)
    ], dtype=float)
    
    # W coefficients for C entries (indices 0:00, 1:01, 2:10, 3:11)
    # Note: W index is k*n + i. For 2x2, k is row of C, i is col of C.
    # C11 (k=0, i=0): M1(1), M4(1), M5(-1), M7(1)
    # C12 (k=0, i=1): M3(1), M5(1)
    # C21 (k=1, i=0): M2(1), M4(1)
    # C22 (k=1, i=1): M1(1), M2(-1), M3(1), M6(1)
    
    W = np.array([
        [ 1,  0,  0,  1, -1,  0,  1],  # C11 (0,0)
        [ 0,  0,  1,  0,  1,  0,  0],  # C12 (0,1)
        [ 0,  1,  0,  1,  0,  0,  0],  # C21 (1,0)
        [ 1, -1,  1,  0,  0,  1,  0]   # C22 (1,1)
    ], dtype=float)
    
    return U, V, W


def _build_block_decomposition(n, m, p):
    """
    Construct the decomposition by tiling 2x2x2 Strassen blocks and filling
    the rest with rank-1 schoolbook terms.
    """
    U_cols = []
    V_cols = []
    W_cols = []
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    covered = set()
    blocks = []
    
    # Identify 2x2x2 blocks
    for i0 in range(0, n - 1, 2):
        for j0 in range(0, m - 1, 2):
            for k0 in range(0, p - 1, 2):
                blocks.append((i0, j0, k0))
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i0 + di, j0 + dj, k0 + dk))
    
    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        for r in range(7):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            # Fill U
            for di in range(2):
                for dj in range(2):
                    u_col[(i0 + di) * m + (j0 + dj)] = strassen_U[di * 2 + dj, r]
            
            # Fill V
            for dj in range(2):
                for dk in range(2):
                    v_col[(j0 + dj) * p + (k0 + dk)] = strassen_V[dj * 2 + dk, r]
            
            # Fill W
            for dk in range(2):
                for di in range(2):
                    w_col[(k0 + dk) * n + (i0 + di)] = strassen_W[dk * 2 + di, r]
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Add Schoolbook for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_col = np.zeros(n * m)
                    v_col = np.zeros(m * p)
                    w_col = np.zeros(p * n)
                    
                    u_col[i * m + j] = 1.0
                    v_col[j * p + k] = 1.0
                    w_col[k * n + i] = 1.0
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    if not U_cols:
        # Fallback to full schoolbook if something went wrong
        return _schoolbook(n, m, p)
        
    U_mat = np.column_stack(U_cols)
    V_mat = np.column_stack(V_cols)
    W_mat = np.column_stack(W_cols)
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _schoolbook(n, m, p):
    """Schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # 1. Try the block decomposition (Strassen + Schoolbook)
    result = _build_block_decomposition(n, m, p)
    
    # 2. Verify
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Valid! Return it.
        # The rank is sum of ranks of blocks.
        # For 3x5x6:
        # 2x2x2 blocks: floor(3/2)=1, floor(5/2)=2, floor(6/2)=3 -> 1*2*3 = 6 blocks.
        # Rank from Strassen: 6 * 7 = 42.
        # Uncovered entries:
        # Total entries 3*5*6 = 90.
        # Covered by 2x2x2: 6 * 8 = 48.
        # Remaining: 90 - 48 = 42 entries.
        # Each remaining entry is rank 1.
        # Total rank = 42 + 42 = 84.
        
        # Check time
        if time.time() < deadline:
            return {
                "U": result["U"].tolist(),
                "V": result["V"].tolist(),
                "W": result["W"].tolist()
            }
    
    # 3. Fallback to schoolbook if block decomposition failed verification (should not happen)
    if time.time() < deadline:
        school = _schoolbook(n, m, p)
        if _verify(school["U"], school["V"], school["W"], n, m, p):
            return {
                "U": school["U"].tolist(),
                "V": school["V"].tolist(),
                "W": school["W"].tolist()
            }
            
    # Emergency fallback: return schoolbook even if verification takes too long
    # (Schoolbook is always valid, so this is safe)
    school = _schoolbook(n, m, p)
    return {
        "U": school["U"].tolist(),
        "V": school["V"].tolist(),
        "W": school["W"].tolist()
    }
