import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
N, M, P = 3, 5, 7

def _verify(U, V, W, n=N, m=M, p=P):
    """
    Verify rank decomposition for n x m times m x p matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: for each c, (U_c @ V_c)_{i,k} = W_c[k, i]
    where U_c = U[:, c].reshape(n, m), V_c = V[:, c].reshape(m, p), W_c = W[:, c].reshape(p, n)
    """
    r = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, r)
    V_arr = np.array(V, dtype=float).reshape(m, p, r)
    W_arr = np.array(W, dtype=float).reshape(p, n, r)
    for c in range(r):
        u_mat = U_arr[:, :, c]
        v_mat = V_arr[:, :, c]
        prod = u_mat @ v_mat
        w_mat = W_arr[:, :, c].T
        if not np.allclose(prod, w_mat, atol=1e-9):
            return False
    return True

def _construct_schoolbook():
    """Standard n*m*p rank decomposition."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries.
    """
    # Strassen's 7 products for multiplying A (2x2) * B (2x2) = C
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # The 7 rank-1 terms:
    # m1 = (a21 - a11) * (b11 + b12)   -> contributes to C
    # m2 = (a11 + a12) * b21            -> contributes to C
    # m3 = a11 * (b22 - b21)            -> contributes to C
    # m4 = a22 * (b11 - b12)            -> contributes to C
    # m5 = (a11 + a22) * (b11 + b22)    -> contributes to C
    # m6 = (a12 - a22) * (b21 + b22)    -> contributes to C
    # m7 = (a11 - a21) * (b11 + b12)    -> contributes to C
    #
    # U rows: [a11, a12, a21, a22] indexed as [0,1,2,3]
    # V rows: [b11, b12, b21, b22] indexed as [0,1,2,3]
    # W rows: [c11, c12, c21, c22] indexed as [0,1,2,3]
    #
    # C11 = m1 + m3 - m4 + m5
    # C12 = m2 + m4
    # C21 = m6 + m7
    # C22 = -m1 + m3 + m5 + m6
    
    # U: 4x7, row i corresponds to a_{i} in order [a11, a12, a21, a22]
    # Column c corresponds to product c
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a21 - a11) * (b11 + b12)
    # a21 is index 2, a11 is index 0; b11 is index 0, b12 is index 1
    U[0, 0] = -1  # a11
    U[2, 0] = 1   # a21
    V[0, 0] = 1   # b11
    V[1, 0] = 1   # b12
    W[0, 0] = 1   # c11
    W[3, 0] = -1  # c22
    
    # m2 = (a11 + a12) * b21
    # a11 is index 0, a12 is index 1; b21 is index 2
    U[0, 1] = 1   # a11
    U[1, 1] = 1   # a12
    V[2, 1] = 1   # b21
    W[1, 1] = 1   # c12
    
    # m3 = a11 * (b22 - b21)
    # a11 is index 0; b22 is index 3, b21 is index 2
    U[0, 2] = 1   # a11
    V[2, 2] = -1  # b21
    V[3, 2] = 1   # b22
    W[0, 2] = 1   # c11
    W[3, 2] = 1   # c22
    
    # m4 = a22 * (b11 - b12)
    # a22 is index 3; b11 is index 0, b12 is index 1
    U[3, 3] = 1   # a22
    V[0, 3] = 1   # b11
    V[1, 3] = -1  # b12
    W[0, 3] = -1  # c11
    W[1, 3] = 1   # c12
    
    # m5 = (a11 + a22) * (b11 + b22)
    # a11 is index 0, a22 is index 3; b11 is index 0, b22 is index 3
    U[0, 4] = 1   # a11
    U[3, 4] = 1   # a22
    V[0, 4] = 1   # b11
    V[3, 4] = 1   # b22
    W[0, 4] = 1   # c11
    W[3, 4] = 1   # c22
    
    # m6 = (a12 - a22) * (b21 + b22)
    # a12 is index 1, a22 is index 3; b21 is index 2, b22 is index 3
    U[1, 5] = 1   # a12
    U[3, 5] = -1  # a22
    V[2, 5] = 1   # b21
    V[3, 5] = 1   # b22
    W[2, 5] = 1   # c21
    W[3, 5] = 1   # c22
    
    # m7 = (a11 - a21) * (b11 + b12)
    # a11 is index 0, a21 is index 2; b11 is index 0, b12 is index 1
    U[0, 6] = 1   # a11
    U[2, 6] = -1  # a21
    V[0, 6] = 1   # b11
    V[1, 6] = 1   # b12
    W[2, 6] = 1   # c21
    
    return U.tolist(), V.tolist(), W.tolist()

def _construct_hybrid_strassen():
    """
    Hybrid approach: Use Strassen blocks where possible, schoolbook for the rest.
    
    For 3x5 x 5x7:
    - Split the 3 rows into 2+1
    - Split the 5 cols of A (and rows of B) into 2+3
    - Split the 7 cols of B into 3+4
    
    Block structure:
    A = [[A11 (2x2), A12 (2x3)],
         [A21 (1x2), A22 (1x3)]]
    B = [[B11 (2x3), B12 (2x4)],
         [B21 (3x3), B22 (3x4)]]
    
    C = [[C11 (2x3), C12 (2x4)],
         [C21 (1x3), C22 (1x3)]]  -- wait, C is 3x7
    
    Actually let me think more carefully about what blocks we can apply Strassen to.
    Strassen applies to square blocks. For non-square, we need rectangular generalizations.
    
    Let me try a different approach: partition into overlapping 2x2 subproblems.
    
    Actually, let's try a simpler structural approach:
    - Use the fact that 3 = 2+1, 5 = 2+2+1, 7 = 3+4 or similar
    - For each 2x2 x 2x2 subblock pair, use Strassen (rank 7 instead of 8)
    - For remaining, use schoolbook
    
    Let me try: partition A's columns as 2+2+1 and B's columns as 3+4 (so B rows as 2+2+1)
    A is 3x5: rows 3, cols 5 = 2+2+1
    B is 5x7: rows 5 = 2+2+1, cols 7 = 3+4
    
    Sub-blocks of A (row_block, col_block):
    (0,0): 2x2, (0,1): 2x2, (0,2): 2x1
    (1,0): 1x2, (1,1): 1x2, (1,2): 1x1
    
    Sub-blocks of B (row_block, col_block):
    (0,0): 2x3, (0,1): 2x4
    (1,0): 2x3, (1,1): 2x4
    (2,0): 1x3, (2,1): 1x4
    
    C = A @ B, C is 3x7
    C(row_block, col_block) = sum over B's row blocks: A(row_block, mid) @ B(mid, col_block)
    
    C(0,0) (2x3) = A(0,0)(2x2)@B(0,0)(2x3) + A(0,1)(2x2)@B(1,0)(2x3) + A(0,2)(2x1)@B(2,0)(1x3)
    C(0,1) (2x4) = A(0,0)(2x2)@B(0,1)(2x4) + A(0,1)(2x2)@B(1,1)(2x4) + A(0,2)(2x1)@B(2,1)(1x4)
    C(1,0) (1x3) = A(1,0)(1x2)@B(0,0)(2x3) + A(1,1)(1x2)@B(1,0)(2x3) + A(1,2)(1x1)@B(2,0)(1x3)
    C(1,1) (1x4) = A(1,0)(1x2)@B(0,1)(2x4) + A(1,1)(1x2)@B(1,1)(2x4) + A(1,2)(1x1)@B(2,1)(1x4)
    
    This is getting complex. Let me just try a direct rank reduction approach.
    
    Alternative: Try to find a good low-rank decomposition using numerical optimization
    but with a smarter initialization based on Kronecker products of smaller blocks.
    
    For now, let me just return the schoolbook as a baseline and try numerical search.
    """
    return None

def _num_search(r, deadline, num_restarts=10):
    """
    Numerical search for rank-r decomposition using ALS.
    """
    n, m, p = N, M, P
    
    def _als(U, V, W, iters=30):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        for it in range(iters):
            if time.time() > deadline - 2:
                break
            # Update W: W_c = (U_c @ V_c).T
            for c in range(r):
                prod = U[:, c].reshape(n, m) @ V[:, c].reshape(m, p)
                W[:, c] = prod.T.flatten()
            # Update U
            for c in range(r):
                V_c = V[:, c].reshape(m, p)
                W_cT = W[:, c].reshape(p, n).T
                A_mat = np.kron(V_c.T, np.eye(n))
                b_vec = W_cT.flatten()
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[:, c] = sol
            # Update V
            for c in range(r):
                U_c = U[:, c].reshape(n, m)
                W_cT = W[:, c].reshape(p, n).T
                A_mat = np.kron(np.eye(p), U_c)
                b_vec = W_cT.flatten()
                sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[:, c] = sol
        return U, V, W
    
    def _try_round(U, V, W):
        # Try half-integer rounding
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
        # Try integer rounding
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
        return None
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        rng = np.random.RandomState(restart * 1000 + 42)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        U, V, W = _als(U, V, W, iters=20)
        result = _try_round(U, V, W)
        if result is not None:
            U_r, V_r, W_r = result
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return None

def construct():
    deadline = time.time() + 110
    best_result = None
    best_rank = 105
    
    # Schoolbook
    schoolbook = _construct_schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        best_result = schoolbook
        best_rank = 105
    
    # Try numerical search for various ranks
    if time.time() < deadline - 10:
        for r in [55, 60, 65, 70, 75, 80, 85, 90, 95]:
            if time.time() > deadline - 8:
                break
            result = _num_search(r, deadline, num_restarts=3)
            if result is not None and _verify(result["U"], result["V"], result["W"]):
                if r < best_rank:
                    best_rank = r
                    best_result = result
                break  # Found the lowest in this scan
    
    if best_result is None:
        return schoolbook
    return best_result
