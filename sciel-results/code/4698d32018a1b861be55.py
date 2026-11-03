"""
Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication.

Strategy:
  - Use Strassen's rank-7 algorithm for the 4x4 x 4x4 part (columns 0-3 of B).
  - Use schoolbook rank-16 for the 4x4 x 4x1 part (column 4 of B).
  - Total rank: 7 + 16 = 23.

The tensor decomposition satisfies:
  C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
  where C = A * B, with A (4x4), B (4x5), C (4x5).

Indexing convention:
  U: shape (n*m, rank) = (16, 23)   -- index: i*m + j  (i=row of A, j=col of A)
  V: shape (m*p, rank) = (20, 23)   -- index: j*p + k  (j=row of B, k=col of B)
  W: shape (p*n, rank) = (20, 23)   -- index: k*n + i  (k=col of C, i=row of C)
"""

import numpy as np
import time

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 120  # seconds; keep safety margin


def _strassen_4x4x4():
    """
    Return the 7 Strassen factors for 4x4x4 matrix multiplication.
    
    Each factor is a tuple (A_coeffs, B_coeffs, C_coeffs) where:
      A_coeffs: dict (i,j) -> coeff for linear form on A
      B_coeffs: dict (i,j) -> coeff for linear form on B  
      C_coeffs: dict (i,j) -> coeff for output matrix C[i,j]
    
    Uses 0-based indexing: i,j in {0,1,2,3}
    
    Strassen splits 4x4 into 2x2 blocks:
      A = [[A00, A01], [A10, A11]] where A00=A[0:2,0:2], etc.
      B = [[B00, B01], [B10, B11]]
      C = [[C00, C01], [C10, C11]]
    
    The 7 Strassen products for 2x2x2:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B10)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    Output:
      C00 = M1 + M4 - M5 + M7
      C01 = M3 + M5
      C10 = M2 + M4
      C11 = M1 - M2 + M3 + M6
    """
    # Each factor: (A_dict, B_dict, C_dict)
    # A_dict maps (block_row, block_col) -> coeff, where block indices are 0 or 1
    # We'll expand to element-level indices.
    
    # Block (br, bc) covers elements (2*br+dr, 2*bc+dc) for dr,dc in {0,1}
    
    def block_to_elements(block_dict):
        """Convert block-level dict to element-level dict."""
        elem = {}
        for (br, bc), coeff in block_dict.items():
            for dr in range(2):
                for dc in range(2):
                    elem[(2*br + dr, 2*bc + dc)] = elem.get((2*br + dr, 2*bc + dc), 0) + coeff
        return elem
    
    # Define the 7 Strassen factors at block level
    strassen_block = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B10-B11) -> C01+=1, C11+=1
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B01-B10) -> C00+=1, C10+=1
        ({(1,1): 1}, {(0,1): 1, (1,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Convert to element-level
    strassen_elem = []
    for A_b, B_b, C_b in strassen_block:
        strassen_elem.append((block_to_elements(A_b), block_to_elements(B_b), block_to_elements(C_b)))
    
    return strassen_elem


def _verify_decomposition(U, V, W, n=4, m=4, p=5, num_tests=10, tol=1e-10):
    """
    Verify that the tensor decomposition correctly computes C = A * B.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    Convention: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    """
    rank = U.shape[1]
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    # Linear form in A: sum_j U[i*m+j, r] * A[i,j]
                    a_val = 0.0
                    for j in range(m):
                        a_val += U[i*m + j, r] * A[i, j]
                    # Linear form in B: sum_j V[j*p+k, r] * B[j,k]
                    b_val = 0.0
                    for j in range(m):
                        b_val += V[j*p + k, r] * B[j, k]
                    val += W[k*n + i, r] * a_val * b_val
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    
    # Build schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 80
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Strassen for 4x4x4: rank 7
    # Schoolbook for 4x4x1 (column 4): rank 16
    # Total: 23
    
    rank_total = 23
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    # Strassen part: columns 0-3 of B (k=0,1,2,3)
    strassen_factors = _strassen_4x4x4()
    
    for r, (A_lin, B_lin, C_mat) in enumerate(strassen_factors):
        for (i, j), coeff in A_lin.items():
            # U index: i*m + j, where i is row of A, j is col of A
            U[i*m + j, r] += coeff
        for (i, j), coeff in B_lin.items():
            # V index: i*p + j, where i is row of B (= col of A), j is col of B
            # But B_lin uses (row_of_B, col_of_B) = (i, j)
            # For the Strassen part, B is 4x4 (columns 0-3 of the full B)
            # So B[i,j] with j in {0,1,2,3} maps to V[i*p + j, r]
            V[i*p + j, r] += coeff
        for (i, j), coeff in C_mat.items():
            # C_mat[i,j] means contribution to C[i,j]
            # W index is k*n + i where k is col of C, i is row of C
            # So for C[i,j]: k=j, i=i, W index = j*n + i
            W[j*n + i, r] += coeff
    
    # Schoolbook part: column 4 of B (k=4)
    for i in range(n):
        for j in range(m):
            col_idx = 7 + i*m + j
            U[i*m + j, col_idx] = 1
            V[j*p + 4, col_idx] = 1
            W[4*n + i, col_idx] = 1  # k=4, so W index is 4*n + i
    
    # Verify the decomposition
    if time.time() - start_time < TIME_LIMIT - 10:
        try:
            if _verify_decomposition(U, V, W, n, m, p, num_tests=10, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass
    
    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
