import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    rank = U_np.shape[1]
    
    # Vectorized check
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We need T[i, j, k] == 1 for all i, j, k.
    
    # Construct index arrays
    i_idx, j_idx, k_idx = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    
    # Flatten indices for U, V, W
    u_idx = (i_idx * m + j_idx).flatten()
    v_idx = (j_idx * p + k_idx).flatten()
    w_idx = (k_idx * n + i_idx).flatten()
    
    # Get rows
    U_rows = U_np[u_idx, :]
    V_rows = V_np[v_idx, :]
    W_rows = W_np[w_idx, :]
    
    # Compute sum of products
    products = U_rows * V_rows * W_rows
    totals = np.sum(products, axis=1)
    
    if not np.allclose(totals, 1.0, atol=tol):
        return False
        
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification. Only used if fast verification passes or as final check."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(U, V, W, col_offset, b_col_start):
    """
    Add a Strassen 2x2*2x2 block for columns b_col_start and b_col_start+1 of B.
    """
    n, m, p = 2, 3, 7
    
    # A indices in U: A[i,j] -> U[i*m + j]
    A00 = 0 * m + 0  # U[0]
    A01 = 0 * m + 1  # U[1]
    A10 = 1 * m + 0  # U[3]
    A11 = 1 * m + 1  # U[4]
    
    # B indices in V: B[j,k] -> V[j*p + k]
    b00 = 0 * p + b_col_start      # V[b_col_start]
    b01 = 0 * p + (b_col_start + 1)  # V[b_col_start + 1]
    b10 = 1 * p + b_col_start      # V[7 + b_col_start]
    b11 = 1 * p + (b_col_start + 1)  # V[7 + b_col_start + 1]
    
    # C indices in W: C[i,k] -> W[k*n + i]
    C00 = b_col_start * n + 0       # W[b_col_start * 2]
    C01 = (b_col_start + 1) * n + 0  # W[(b_col_start + 1) * 2]
    C10 = b_col_start * n + 1       # W[b_col_start * 2 + 1]
    C11 = (b_col_start + 1) * n + 1  # W[(b_col_start + 1) * 2 + 1]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[A00][col_offset] += 1
    U[A11][col_offset] += 1
    V[b00][col_offset] += 1
    V[b11][col_offset] += 1
    W[C00][col_offset] += 1
    W[C11][col_offset] += 1
    
    # M2 = (A10 + A11)B00
    U[A10][col_offset + 1] += 1
    U[A11][col_offset + 1] += 1
    V[b00][col_offset + 1] += 1
    W[C10][col_offset + 1] += 1
    W[C11][col_offset + 1] -= 1
    
    # M3 = A00(B01 - B11)
    U[A00][col_offset + 2] += 1
    V[b01][col_offset + 2] += 1
    V[b11][col_offset + 2] -= 1
    W[C01][col_offset + 2] += 1
    W[C11][col_offset + 2] += 1
    
    # M4 = A11(B10 - B00)
    U[A11][col_offset + 3] += 1
    V[b10][col_offset + 3] += 1
    V[b00][col_offset + 3] -= 1
    W[C00][col_offset + 3] += 1
    W[C10][col_offset + 3] += 1
    
    # M5 = (A00 + A01)B11
    U[A00][col_offset + 4] += 1
    U[A01][col_offset + 4] += 1
    V[b11][col_offset + 4] += 1
    W[C00][col_offset + 4] -= 1
    W[C01][col_offset + 4] += 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[A10][col_offset + 5] += 1
    U[A00][col_offset + 5] -= 1
    V[b00][col_offset + 5] += 1
    V[b01][col_offset + 5] += 1
    W[C11][col_offset + 5] += 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[A01][col_offset + 6] += 1
    U[A11][col_offset + 6] -= 1
    V[b10][col_offset + 6] += 1
    V[b11][col_offset + 6] += 1
    W[C00][col_offset + 6] += 1
    W[C11][col_offset + 6] += 1
    
    return col_offset + 7

def _schoolbook_2x2_2x1(U, V, W, col_offset, b_col):
    """
    Add a 2x2 * 2x1 schoolbook block (4 multiplications) for column b_col of B.
    """
    n, m, p = 2, 3, 7
    
    A00 = 0 * m + 0
    A01 = 0 * m + 1
    A10 = 1 * m + 0
    A11 = 1 * m + 1
    
    b0 = 0 * p + b_col
    b1 = 1 * p + b_col
    
    C00 = b_col * n + 0
    C10 = b_col * n + 1
    
    # A00 * B[0, b_col]
    U[A00][col_offset] = 1
    V[b0][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A01 * B[1, b_col]
    U[A01][col_offset] = 1
    V[b1][col_offset] = 1
    W[C00][col_offset] = 1
    col_offset += 1
    
    # A10 * B[0, b_col]
    U[A10][col_offset] = 1
    V[b0][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    # A11 * B[1, b_col]
    U[A11][col_offset] = 1
    V[b1][col_offset] = 1
    W[C10][col_offset] = 1
    col_offset += 1
    
    return col_offset

def _schoolbook_2x1_1x7(U, V, W, col_offset, a_col, b_row):
    """
    Add a 2x1 * 1x7 schoolbook block.
    """
    n, m, p = 2, 3, 7
    
    for i in range(n):
        for k in range(p):
            U[i * m + a_col][col_offset] = 1
            V[b_row * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    
    return col_offset

def _construct_rank_39():
    """
    Construct the rank-39 decomposition.
    """
    n, m, p = 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n
    rank = 39
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    # Part 1: A_left * B_top
    col = _strassen_2x2_block(U, V, W, col, 0)   # columns 0,1
    col = _strassen_2x2_block(U, V, W, col, 2)   # columns 2,3
    col = _strassen_2x2_block(U, V, W, col, 4)   # columns 4,5
    col = _schoolbook_2x2_2x1(U, V, W, col, 6)   # column 6
    
    # Part 2: A_right * B_bottom
    col = _schoolbook_2x1_1x7(U, V, W, col, 2, 2)
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try the rank-39 construction first
    result = _construct_rank_39()
    
    # Verify it's correct
    if _verify_decomposition_fast(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
        # Optional: Exact verification if time permits
        if time.time() < deadline - 5:
            if _verify_decomposition_exact(result["U"], result["V"], result["W"], n, m, p):
                return result
        else:
            return result

    # Fallback to schoolbook
    return _schoolbook()
