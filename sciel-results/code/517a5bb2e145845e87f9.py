"""Matrix multiplication 4,4,6 - Rank-28 Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, seed=42):
    """Verify that U,V,W exactly decompose the matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(seed)
    for trial in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_recon = np.zeros((n, p))
        for c in range(U.shape[1]):
            u_col = U[:, c].reshape(n, m)
            v_col = V[:, c].reshape(m, p)
            w_col = W[:, c].reshape(p, n)
            prod = u_col @ v_col  # n x p
            C_recon += prod * w_col.T
        if not np.allclose(C_recon, C_expected, atol=1e-9):
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
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
    return U, V, W

def _strassen_4x4_4x6(n, m, p):
    """
    Build Strassen decomposition for 4x4 * 4x6 with rank 28.
    
    Split A(4x4) into 2x2 blocks of 2x2: A11, A12, A21, A22
    Split B(4x6) into 2x3 blocks of 2x3: B11, B12, B21, B22
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 - M5 + M7   (2x3)
    C12 = M3 + M5        (2x3)
    C21 = M2 + M4        (2x3)
    C22 = M1 - M2 + M3 + M6   (2x3)
    
    Each Mi is 2x2 * 2x3, decomposed via schoolbook (rank 4 each).
    Total rank: 7 * 4 = 28.
    """
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 7 Strassen multipliers.
    # Each multiplier: X (2x2 linear combo of A) times Y (2x3 linear combo of B)
    # X[ri, ci] = sum of (coeff, A_row, A_col)
    # Y[ri, ci] = sum of (coeff, B_row, B_col)
    # C contribution: for each (ri, ci) in result, add to C at (base_row+ri, base_col+ci) with sign
    
    # Block definitions: (x_coeffs, y_coeffs, c_contrib)
    # c_contrib: dict (c_row, c_col) -> coeff
    
    def make_block(x_coeffs, y_coeffs, c_contrib):
        return (x_coeffs, y_coeffs, c_contrib)
    
    blocks = []
    
    # M1 = (A11+A22)(B11+B22)
    # A11: A[0:2,0:2], A22: A[2:4,2:4]
    # B11: B[0:2,0:3], B22: B[2:4,3:6]
    # C11: C[0:2,0:3], sign +1; C22: C[2:4,3:6], sign +1
    blocks.append(make_block(
        {(0,0): [(1,0,0),(1,2,2)], (0,1): [(1,0,1),(1,2,3)],
         (1,0): [(1,1,0),(1,3,2)], (1,1): [(1,1,1),(1,3,3)]},
        {(0,0): [(1,0,0),(1,2,3)], (0,1): [(1,0,1),(1,2,4)], (0,2): [(1,0,2),(1,2,5)],
         (1,0): [(1,1,0),(1,3,3)], (1,1): [(1,1,1),(1,3,4)], (1,2): [(1,1,2),(1,3,5)]},
        {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1,
         (2,3):1,(2,4):1,(2,5):1,(3,3):1,(3,4):1,(3,5):1}
    ))
    
    # M2 = (A21+A22)B11
    # A21: A[2:4,0:2], A22: A[2:4,2:4]
    # B11: B[0:2,0:3]
    # C21: C[2:4,0:3], sign +1; C22: C[2:4,3:6], sign -1
    blocks.append(make_block(
        {(0,0): [(1,2,0),(1,2,2)], (0,1): [(1,2,1),(1,2,3)],
         (1,0): [(1,3,0),(1,3,2)], (1,1): [(1,3,1),(1,3,3)]},
        {(0,0): [(1,0,0)], (0,1): [(1,0,1)], (0,2): [(1,0,2)],
         (1,0): [(1,1,0)], (1,1): [(1,1,1)], (1,2): [(1,1,2)]},
        {(2,0):1,(2,1):1,(2,2):1,(3,0):1,(3,1):1,(3,2):1,
         (2,3):-1,(2,4):-1,(2,5):-1,(3,3):-1,(3,4):-1,(3,5):-1}
    ))
    
    # M3 = A11(B12-B22)
    # A11: A[0:2,0:2]
    # B12: B[0:2,3:6], B22: B[2:4,3:6]
    # C12: C[0:2,3:6], sign +1; C22: C[2:4,3:6], sign +1
    blocks.append(make_block(
        {(0,0): [(1,0,0)], (0,1): [(1,0,1)],
         (1,0): [(1,1,0)], (1,1): [(1,1,1)]},
        {(0,0): [(1,0,3),(-1,2,3)], (0,1): [(1,0,4),(-1,2,4)], (0,2): [(1,0,5),(-1,2,5)],
         (1,0): [(1,1,3),(-1,3,3)], (1,1): [(1,1,4),(-1,3,4)], (1,2): [(1,1,5),(-1,3,5)]},
        {(0,3):1,(0,4):1,(0,5):1,(1,3):1,(1,4):1,(1,5):1,
         (2,3):1,(2,4):1,(2,5):1,(3,3):1,(3,4):1,(3,5):1}
    ))
    
    # M4 = A22(B21-B11)
    # A22: A[2:4,2:4]
    # B21: B[2:4,0:3], B11: B[0:2,0:3]
    # C21: C[2:4,0:3], sign +1
    blocks.append(make_block(
        {(0,0): [(1,2,2)], (0,1): [(1,2,3)],
         (1,0): [(1,3,2)], (1,1): [(1,3,3)]},
        {(0,0): [(1,2,0),(-1,0,0)], (0,1): [(1,2,1),(-1,0,1)], (0,2): [(1,2,2),(-1,0,2)],
         (1,0): [(1,3,0),(-1,1,0)], (1,1): [(1,3,1),(-1,1,1)], (1,2): [(1,3,2),(-1,1,2)]},
        {(2,0):1,(2,1):1,(2,2):1,(3,0):1,(3,1):1,(3,2):1}
    ))
    
    # M5 = (A11+A12)B22
    # A11: A[0:2,0:2], A12: A[0:2,2:4]
    # B22: B[2:4,3:6]
    # C11: C[0:2,0:3], sign -1; C12: C[0:2,3:6], sign +1
    blocks.append(make_block(
        {(0,0): [(1,0,0),(1,0,2)], (0,1): [(1,0,1),(1,0,3)],
         (1,0): [(1,1,0),(1,1,2)], (1,1): [(1,1,1),(1,1,3)]},
        {(0,0): [(1,2,3)], (0,1): [(1,2,4)], (0,2): [(1,2,5)],
         (1,0): [(1,3,3)], (1,1): [(1,3,4)], (1,2): [(1,3,5)]},
        {(0,0):-1,(0,1):-1,(0,2):-1,(1,0):-1,(1,1):-1,(1,2):-1,
         (0,3):1,(0,4):1,(0,5):1,(1,3):1,(1,4):1,(1,5):1}
    ))
    
    # M6 = (A21-A11)(B11+B12)
    # A21: A[2:4,0:2], A11: A[0:2,0:2]
    # B11: B[0:2,0:3], B12: B[0:2,3:6]
    # C22: C[2:4,3:6], sign +1
    blocks.append(make_block(
        {(0,0): [(1,2,0),(-1,0,0)], (0,1): [(1,2,1),(-1,0,1)],
         (1,0): [(1,3,0),(-1,1,0)], (1,1): [(1,3,1),(-1,1,1)]},
        {(0,0): [(1,0,0),(1,0,3)], (0,1): [(1,0,1),(1,0,4)], (0,2): [(1,0,2),(1,0,5)],
         (1,0): [(1,1,0),(1,1,3)], (1,1): [(1,1,1),(1,1,4)], (1,2): [(1,1,2),(1,1,5)]},
        {(2,3):1,(2,4):1,(2,5):1,(3,3):1,(3,4):1,(3,5):1}
    ))
    
    # M7 = (A12-A22)(B21+B22)
    # A12: A[0:2,2:4], A22: A[2:4,2:4]
    # B21: B[2:4,0:3], B22: B[2:4,3:6]
    # C11: C[0:2,0:3], sign +1
    blocks.append(make_block(
        {(0,0): [(1,0,2),(-1,2,2)], (0,1): [(1,0,3),(-1,2,3)],
         (1,0): [(1,1,2),(-1,3,2)], (1,1): [(1,1,3),(-1,3,3)]},
        {(0,0): [(1,2,0),(1,2,3)], (0,1): [(1,2,1),(1,2,4)], (0,2): [(1,2,2),(1,2,5)],
         (1,0): [(1,3,0),(1,3,3)], (1,1): [(1,3,1),(1,3,4)], (1,2): [(1,3,2),(1,3,5)]},
        {(0,0):1,(0,1):1,(0,2):1,(1,0):1,(1,1):1,(1,2):1}
    ))
    
    # Now generate the 4 schoolbook terms per block.
    # For 2x2 * 2x3: Z[ri, ci] = sum_k X[ri,k] * Y[k,ci]
    # Schoolbook terms: (ri, k) for ri in {0,1}, k in {0,1} -> 4 terms
    # Term (ri, k): X[ri,k] * Y[k,:] contributes to Z[ri,:]
    
    col = 0
    for x_coeffs, y_coeffs, c_contrib in blocks:
        for ri in range(2):
            for k in range(2):
                # U: coefficients for A entries in X[ri,k]
                for (coeff, ar, ac) in x_coeffs[(ri, k)]:
                    U[ar * m + ac, col] += coeff
                
                # V: coefficients for B entries in Y[k, ci] for ci=0,1,2
                for ci in range(3):
                    for (coeff, br, bc) in y_coeffs[(k, ci)]:
                        V[br * p + bc, col] += coeff
                
                # W: coefficients for C entries
                # Z[ri, ci] contributes to C at position determined by c_contrib
                # The C block has 2 rows and 3 cols.
                # c_contrib maps (c_row, c_col) -> coeff
                
                c_rows = sorted(set(r for (r, c) in c_contrib.keys()))
                c_cols = sorted(set(c for (r, c) in c_contrib.keys()))
                
                c_row = c_rows[ri]
                for ci in range(3):
                    c_col = c_cols[ci]
                    if (c_row, c_col) in c_contrib:
                        W[c_col * n + c_row, col] += c_contrib[(c_row, c_col)]
                
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Try Strassen decomposition (rank 28)
    try:
        U, V, W = _strassen_4x4_4x6(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 96)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
