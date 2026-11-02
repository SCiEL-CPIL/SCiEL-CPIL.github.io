import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _make_zero_matrices(n, m, p, rank):
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    return U, V, W

def _add_strassen_block(U, V, W, col,
                        i1, i2, j1, j2, k1, k2,
                        n, m, p):
    # Strassen factors for a 2x2x2 block:
    # A = [[a11, a12], [a21, a22]]  indexed by (i in {i1,i2}, j in {j1,j2})
    # B = [[b11, b12], [b21, b22]]  indexed by (j in {j1,j2}, k in {k1,k2})
    # C = [[c11, c12], [c21, c22]]  indexed by (k in {k1,k2}, i in {i1,i2})
    #
    # 7 products:
    # m0 = (A00+A11)*(B00+B11) = (a11+a22)*(b11+b22)
    # m1 = (A10+A11)*B00      = (a21+a22)*b11
    # m2 = A00*(B01-B11)      = a11*(b12-b22)
    # m3 = A11*(B10-B00)      = a22*(b21-b11)
    # m4 = (A00+A01)*B11      = (a11+a12)*b22
    # m5 = (A10-A00)*(B00+B01)= (a21-a11)*(b11+b12)
    # m6 = (A01-A11)*(B10+B11)= (a12-a22)*(b21+b22)
    #
    # C00 = m0 + m3 - m4 - m6
    # C01 = m2 + m4
    # C10 = m1 + m5
    # C11 = m0 - m1 + m2 + m5

    # For each factor r=0..6, set U[i*m+j, col+r], V[j*p+k, col+r], W[k*n+i, col+r]
    # We need to add signed contributions.

    # factor 0: (a11+a22)*(b11+b22)
    # A part: a11 + a22 -> U at (i1,j1) + (i2,j2)
    # B part: b11 + b22 -> V at (j1,k1) + (j2,k2)
    # C part: c00 - c01 - c10 - c11 -> W at (k1,i1) - (k1,i2) - (k2,i1) - (k2,i2)
    U[i1*m+j1][col+0] += 1
    U[i2*m+j2][col+0] += 1
    V[j1*p+k1][col+0] += 1
    V[j2*p+k2][col+0] += 1
    W[k1*n+i1][col+0] += 1
    W[k1*n+i2][col+0] += -1
    W[k2*n+i1][col+0] += -1
    W[k2*n+i2][col+0] += -1

    # factor 1: (a21+a22)*b11
    # A part: a21 + a22 -> U at (i2,j1) + (i2,j2)
    # B part: b11 -> V at (j1,k1)
    # C part: c10 + c11 -> W at (k2,i1) + (k2,i2)
    U[i2*m+j1][col+1] += 1
    U[i2*m+j2][col+1] += 1
    V[j1*p+k1][col+1] += 1
    W[k2*n+i1][col+1] += 1
    W[k2*n+i2][col+1] += 1

    # factor 2: a11*(b12-b22)
    # A part: a11 -> U at (i1,j1)
    # B part: b12 - b22 -> V at (j1,k2) - (j2,k2)
    # C part: c01 + c11 -> W at (k1,i2) + (k2,i2)
    U[i1*m+j1][col+2] += 1
    V[j1*p+k2][col+2] += 1
    V[j2*p+k2][col+2] += -1
    W[k1*n+i2][col+2] += 1
    W[k2*n+i2][col+2] += 1

    # factor 3: a22*(b21-b11)
    # A part: a22 -> U at (i2,j2)
    # B part: b21 - b11 -> V at (j2,k1) - (j1,k1)
    # C part: c00 -> W at (k1,i1)
    U[i2*m+j2][col+3] += 1
    V[j2*p+k1][col+3] += 1
    V[j1*p+k1][col+3] += -1
    W[k1*n+i1][col+3] += 1

    # factor 4: (a11+a12)*b22
    # A part: a11 + a12 -> U at (i1,j1) + (i1,j2)
    # B part: b22 -> V at (j2,k2)
    # C part: -c00 - c01 -> -W at (k1,i1) - (k1,i2)
    U[i1*m+j1][col+4] += 1
    U[i1*m+j2][col+4] += 1
    V[j2*p+k2][col+4] += 1
    W[k1*n+i1][col+4] += -1
    W[k1*n+i2][col+4] += -1

    # factor 5: (a21-a11)*(b11+b12)
    # A part: a21 - a11 -> U at (i2,j1) - (i1,j1)
    # B part: b11 + b12 -> V at (j1,k1) + (j1,k2)
    # C part: c10 + c11 -> W at (k2,i1) + (k2,i2)
    U[i2*m+j1][col+5] += 1
    U[i1*m+j1][col+5] += -1
    V[j1*p+k1][col+5] += 1
    V[j1*p+k2][col+5] += 1
    W[k2*n+i1][col+5] += 1
    W[k2*n+i2][col+5] += 1

    # factor 6: (a12-a22)*(b21+b22)
    # A part: a12 - a22 -> U at (i1,j2) - (i2,j2)
    # B part: b21 + b22 -> V at (j2,k1) + (j2,k2)
    # C part: -c00 - c10 -> -W at (k1,i1) - (k2,i1)
    U[i1*m+j2][col+6] += 1
    U[i2*m+j2][col+6] += -1
    V[j2*p+k1][col+6] += 1
    V[j2*p+k2][col+6] += 1
    W[k1*n+i1][col+6] += -1
    W[k2*n+i1][col+6] += -1

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify U,V,W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    nm, mp, pn = n*m, m*p, p*n
    rank = U_arr.shape[1]
    
    # Build target tensor T[a,b,c] = 1 iff consistent
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute estimated tensor
    T_est = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    
    max_err = float(np.max(np.abs(T_est - T)))
    return max_err < tol, max_err

def _verify_half_integer(U, V, W, tol=1e-12):
    """Check if all entries are half-integers (integer or half-integer)."""
    for arr_name, arr in [('U', U), ('V', V), ('W', W)]:
        arr_f = np.array(arr, dtype=float)
        doubled = arr_f * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U, V, W = _make_zero_matrices(n, m, p, rank)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _build_rank88(n, m, p):
    """
    Build a rank-88 decomposition using two disjoint Strassen 2x2x2 blocks.
    
    Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 7 factors
    Block 2: i in {0,2}, j in {2,3}, k in {2,3} -> 7 factors
    
    Remaining terms (90 - 4 - 4 = 82) use schoolbook.
    Total: 7 + 7 + 82 = 96? No wait.
    
    Actually: 
    - Block 1 covers (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}: 2*2*2 = 8 terms, replaced by 7 factors.
    - Block 2 covers (i,j,k) where i in {0,2}, j in {2,3}, k in {2,3}: 2*2*2 = 8 terms, replaced by 7 factors.
    - Total terms covered by Strassen: 16, using 14 factors.
    - Remaining terms: 90 - 16 = 74, using 74 factors.
    - Total rank: 14 + 74 = 88.
    """
    rank = 88
    U, V, W = _make_zero_matrices(n, m, p, rank)
    
    # Apply Strassen block 1: i in {0,1}, j in {0,1}, k in {0,1}
    _add_strassen_block(U, V, W, 0,
                        i1=0, i2=1, j1=0, j2=1, k1=0, k2=1,
                        n=n, m=m, p=p)
    
    # Apply Strassen block 2: i in {0,2}, j in {2,3}, k in {2,3}
    _add_strassen_block(U, V, W, 7,
                        i1=0, i2=2, j1=2, j2=3, k1=2, k2=3,
                        n=n, m=m, p=p)
    
    # Remaining terms: all (i,j,k) NOT in block 1 or block 2
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in block 1
                in_block1 = (i in (0,1)) and (j in (0,1)) and (k in (0,1))
                # Check if in block 2
                in_block2 = (i in (0,2)) and (j in (2,3)) and (k in (2,3))
                
                if not in_block1 and not in_block2:
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    assert col == rank, f"Expected col={rank}, got col={col}"
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try the structured rank-88 decomposition
    U88, V88, W88, rank88 = _build_rank88(n, m, p)
    
    # Verify numerically
    valid88, max_err88 = _verify_decomposition(U88, V88, W88, n, m, p)
    
    if valid88:
        # Check if entries are half-integers
        if _verify_half_integer(U88, V88, W88):
            return {"U": U88, "V": V88, "W": W88}
    
    # If verification failed or not half-integer, fall back to schoolbook
    U_sb, V_sb, W_sb, rank_sb = _build_schoolbook(n, m, p)
    valid_sb, _ = _verify_decomposition(U_sb, V_sb, W_sb, n, m, p)
    
    if valid_sb:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # This should never happen, but return schoolbook anyway
    return {"U": U_sb, "V": V_sb, "W": W_sb}
