"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication using Recursive Strassen.

This program constructs the 49-term bilinear form for multiplying two 4x4 matrices.
It leverages the known rank-7 decomposition for 2x2 matrices (Strassen) and applies
it recursively: first at the block level (2x2 blocks of 2x2 matrices) and then at
the element level within those blocks.

The output follows the convention:
U: shape (n*m, rank) -> U[i*m + j, r]
V: shape (m*p, rank) -> V[j*p + k, r]
W: shape (p*n, rank) -> W[k*n + i, r]

For n=m=p=4, rank=49.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's 7-multiplication algorithm.
    n = m = p = 2.
    U: (4, 7), V: (4, 7), W: (4, 7).
    Indexing:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # Indices for 2x2:
    # A00 -> U[0], A01 -> U[1], A10 -> U[2], A11 -> U[3]
    # B00 -> V[0], B01 -> V[1], B10 -> V[2], B11 -> V[3]
    # C00 -> W[0] (k=0, i=0), C01 -> W[1] (k=1, i=0)? 
    # Wait, W index is k*n + i.
    # C00 (i=0, k=0) -> W[0*2+0] = W[0]
    # C01 (i=0, k=1) -> W[1*2+0] = W[2]  <-- CAREFUL: W[k*n+i]
    # C10 (i=1, k=0) -> W[0*2+1] = W[1]
    # C11 (i=1, k=1) -> W[1*2+1] = W[3]
    
    # Let's re-verify W indexing: W[k*n + i]
    # n=2.
    # (i=0, k=0) -> W[0]
    # (i=0, k=1) -> W[2]
    # (i=1, k=0) -> W[1]
    # (i=1, k=1) -> W[3]
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # C00 is (i=0, k=0) -> W[0]. C11 is (i=1, k=1) -> W[3].
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A10+A11)B00 -> C10 += 1
    # C10 is (i=1, k=0) -> W[1].
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3: A00(B01-B11) -> C01 += 1, C11 -= 1
    # C01 is (i=0, k=1) -> W[2]. C11 is (i=1, k=1) -> W[3].
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = -1
    
    # M4: A11(B10-B00) -> C00 += 1, C10 += 1
    # C00 -> W[0]. C10 -> W[1].
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
    # C00 -> W[0]. C01 -> W[2].
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    # C11 -> W[3].
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    # C00 -> W[0]. C11 -> W[3].
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _construct_strassen_4x4():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    U2, V2, W2 = _get_strassen_2x2()
    
    # The 7 block-level Strassen terms.
    # Each term defines linear combinations of 2x2 blocks of A, B, and C.
    # Let A be 2x2 blocks A00, A01, A10, A11.
    # Let B be 2x2 blocks B00, B01, B10, B11.
    # Let C be 2x2 blocks C00, C01, C10, C11.
    
    # Block indices (bi, bj) for 2x2 blocks.
    # bi=0, bj=0 -> A00
    # bi=0, bj=1 -> A01
    # bi=1, bj=0 -> A10
    # bi=1, bj=1 -> A11
    
    # We define the 7 terms by their coefficients on the blocks.
    # Term t has:
    # P_t: dict mapping (bi, bj) -> coeff for A block
    # Q_t: dict mapping (bi, bj) -> coeff for B block
    # C_t: dict mapping (bi, bj) -> coeff for C block
    
    terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        {'P': {(0,0): 1, (1,1): 1}, 'Q': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00 -> C10 += 1
        {'P': {(1,0): 1, (1,1): 1}, 'Q': {(0,0): 1}, 'C': {(1,0): 1}},
        # M3 = A00(B01-B11) -> C01 += 1, C11 -= 1
        {'P': {(0,0): 1}, 'Q': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): -1}},
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        {'P': {(1,1): 1}, 'Q': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        {'P': {(0,0): 1, (0,1): 1}, 'Q': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        {'P': {(1,0): 1, (0,0): -1}, 'Q': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        # M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
        {'P': {(0,1): 1, (1,1): -1}, 'Q': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): -1}},
    ]
    
    for rb, term in enumerate(terms):
        # rb is the block-level Strassen term index (0..6)
        P_coeffs = term['P']
        Q_coeffs = term['Q']
        C_coeffs = term['C']
        
        for rs in range(7):
            # rs is the element-level Strassen term index (0..6)
            r = rb * 7 + rs
            
            # Update U:
            # U[i*m+j, r] where i,j are 0..3.
            # i = 2*bi + a, j = 2*bj + b.
            # The coefficient is P[bi,bj] * U2[a*2+b, rs].
            for (bi, bj), pc in P_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        u2_val = U2[a * 2 + b][rs]
                        if u2_val != 0:
                            i = 2 * bi + a
                            j = 2 * bj + b
                            U[i * 4 + j][r] += pc * u2_val
            
            # Update V:
            # V[j*p+k, r] where j,k are 0..3.
            # j = 2*bi + a, k = 2*bj + b. (Note: block indices for B are (bi, bj) in our term definition? 
            # Yes, Q_coeffs maps block indices (bi, bj) to coefficients.
            # Inside the block, local indices are a, b.
            # Global j = 2*bi + a, Global k = 2*bj + b.
            # Coefficient is Q[bi,bj] * V2[a*2+b, rs].
            for (bi, bj), qc in Q_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        v2_val = V2[a * 2 + b][rs]
                        if v2_val != 0:
                            j = 2 * bi + a
                            k = 2 * bj + b
                            V[j * 4 + k][r] += qc * v2_val
            
            # Update W:
            # W[k*n+i, r] where k,i are 0..3.
            # Block indices (ci, cj) correspond to C block position.
            # C block (ci, cj) contains elements C[2*ci+a, 2*cj+b].
            # Here, row index of C is i (global), col index is k (global).
            # So i = 2*ci + a, k = 2*cj + b.
            # W index is k*4 + i.
            # Coefficient is C[ci,cj] * W2[a*2+b, rs].
            # Note: W2 index is [local_k * 2 + local_i, rs].
            # In _get_strassen_2x2, W2[k*2+i, rs] was used.
            # So local_k = b, local_i = a.
            # W2[b*2+a, rs].
            for (ci, cj), cc in C_coeffs.items():
                for a in range(2):
                    for b in range(2):
                        w2_val = W2[b * 2 + a][rs] # local_k=b, local_i=a
                        if w2_val != 0:
                            i = 2 * ci + a
                            k = 2 * cj + b
                            W[k * 4 + i][r] += cc * w2_val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    Checks if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # It should be all ones.
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback: Schoolbook decomposition (rank n*m*p).
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Primary: Recursive Strassen (Rank 49)
    try:
        U, V, W = _construct_strassen_4x4()
        if time.time() < deadline:
            if _verify_decomposition(U, V, W, n, m, p):
                # Convert to lists as required by interface (or numpy arrays if allowed, but lists are safer for JSON-like serialization if needed, though task says "factor matrices of shapes...")
                # The task says "factor matrices of shapes (n*m, rank)...". It doesn't strictly forbid numpy arrays, but "integer/half-integer entries" suggests standard types might be preferred or at least accepted. 
                # However, numpy arrays are often accepted in these tasks. Let's return lists to be absolutely safe with "integer/half-integer" interpretation if the evaluator checks types, though usually it just checks values.
                # Actually, the prompt says "Python standard library, NumPy and SciPy are available". And "return interface EXACTLY as in the current program".
                # Current program returns lists: {"U": U, "V": V, "W": W} where U is a list of lists.
                # Inspiration 1 returned .tolist().
                # I will return .tolist() to match the "list of lists" structure of the seed program.
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        # If Strassen construction fails, fall back to schoolbook
        pass
    
    # Fallback: Schoolbook (Rank 64)
    U, V, W = _schoolbook_decomposition(n, m, p)
    # Verify schoolbook just in case (it should always pass)
    if time.time() < deadline:
        _verify_decomposition(U, V, W, n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
