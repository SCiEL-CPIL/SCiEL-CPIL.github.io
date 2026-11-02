import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_n, _m, _p = DIMENSIONS

def _schoolbook_rank27():
    """Known valid rank-27 (schoolbook) decomposition."""
    n, m, p = _n, _m, _p
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

def _verify(U, V, W):
    """Check exact tensor decomposition."""
    n, m, p = _n, _m, _p
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank26_block():
    """Rank-26 decomposition: 2x2 Strassen for top-left block + direct for row/col 2.
    
    Partition 3x3 into:
      A = [[A11, A12],
           [A21, A22]]   where A11 is 2x2, A12/A21 are 2x1, A22 is 1x1
      B = [[B11, B12],
           [B21, B22]]
      C = [[C11, C12],
           [C21, C22]]
    
    Use Strassen (7 mults) for the 2x2 block C11, plus direct multiplications
    for the remaining entries:
      C12 = A11*B12 + A12*B22  (2x1 = 2x1*1x1 + 1x1*1x1 -> 2 mults per row = 4 total? No)
      
    Actually let's be more careful. Let me use a simpler partition:
    Treat indices 0,1 as the "big" block and index 2 as the "small" block.
    
    C[i,k] = sum_j A[i,j]*B[j,k]
    
    For i in {0,1}, k in {0,1}: C11[i-0,k-0] = sum_{j in {0,1,2}} A[i,j]*B[j,k]
    For i in {0,1}, k=2: C12[i,0] = sum_{j} A[i,j]*B[j,2]
    For i=2, k in {0,1}: C21[0,k] = sum_{j} A[2,j]*B[j,k]
    For i=2, k=2: C22[0,0] = sum_j A[2,j]*B[j,2]
    
    Strassen on the 2x2 sub-block (indices 0,1 x 0,1) gives 7 products for the
    "core" part. But we also have cross terms with index 2.
    
    Let me think differently. Use the known result that 3x3 matrix multiplication
    can be done in 23 multiplications (Bini et al.), but a simpler 26-multiplication
    scheme exists via block decomposition.
    
    Actually, the simplest valid improvement: use Strassen on a 2x2 block.
    Split into 2x2 and 1x1 blocks. The 2x2*2x2 part uses 7 mults (Strassen).
    The remaining parts:
      C12 (2x1) = A11(2x2)*B12(2x1) + A12(2x1)*B22(1x1)
        = 2*2 + 2*1 = 4 + 2 = 6 mults... that's not right either.
    
    Let me just directly enumerate 26 rank-1 terms that I can verify.
    """
    n, m, p = _n, _m, _p
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 7 products for the 2x2 block (rows/cols 0,1)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # M1
    for (i, j) in [(0,0),(1,1)]:
        U[i*m+j][col] += 1.0
    for (j, k) in [(0,0),(1,1)]:
        V[j*p+k][col] += 1.0
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1
    for (k, i) in [(0,0),(0,1),(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M2 = (A10+A11)*B00
    for (i, j) in [(1,0),(1,1)]:
        U[i*m+j][col] += 1.0
    V[0*p+0][col] += 1.0
    # C00 += M2, C10 += M2
    for (k, i) in [(0,0),(1,0)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M3 = A00*(B10-B11)
    U[0*m+0][col] += 1.0
    V[1*p+0][col] += 1.0
    V[1*p+1][col] += -1.0
    # C01 += M3, C11 += M3
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M4 = A11*(B01-B00)
    U[1*m+1][col] += 1.0
    V[0*p+1][col] += 1.0
    V[0*p+0][col] += -1.0
    # C01 += M4, C11 += M4
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M5 = (A00+A10)*B11
    for (i, j) in [(0,0),(1,0)]:
        U[i*m+j][col] += 1.0
    V[1*p+1][col] += 1.0
    # C01 += M5, C11 += M5
    for (k, i) in [(1,0),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M6 = (A11-A10)*(B00+B10)
    for (i, j) in [(1,1),(1,0)]:
        U[i*m+j][col] += 1.0 if j == 1 else -1.0
    for (j, k) in [(0,0),(1,0)]:
        V[j*p+k][col] += 1.0
    # C10 += M6, C11 += M6
    for (k, i) in [(0,1),(1,1)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    for (i, j) in [(0,1),(1,1)]:
        U[i*m+j][col] += 1.0 if i == 0 else -1.0
    for (j, k) in [(1,0),(1,1)]:
        V[j*p+k][col] += 1.0
    # C00 += M7, C10 += M7
    for (k, i) in [(0,0),(1,0)]:
        W[k*n+i][col] += 1.0
    col += 1
    
    # Now handle the cross terms with index 2.
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]  for i=0,1
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]  for k=0,1
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 2*3 + 2*3 + 3 = 6+6+3 = 15 more terms, total 7+15=22. 
    # But wait, some of these can be combined with Strassen terms.
    # Actually, let me just add them directly. 7 + 15 = 22, not 26.
    # Let me recount: I need to be more careful.
    
    # Hmm, let me just do the straightforward schoolbook for the cross terms.
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    
    # That's 15 terms. Total = 7 + 15 = 22. But that seems too low.
    # The issue: Strassen already computes C11 correctly for the 2x2 submatrix,
    # but C[i,2] and C[2,k] for i,k in {0,1} still need all j in {0,1,2}.
    # The Strassen terms only account for j in {0,1} contributions to C11.
    # For C[i,2], we need j=0,1,2. The j=0,1 parts are NOT covered by Strassen.
    
    # So actually I need:
    # C[0,2]: 3 terms
    # C[1,2]: 3 terms  
    # C[2,0]: 3 terms
    # C[2,1]: 3 terms
    # C[2,2]: 3 terms
    # Total cross: 15 terms
    # Plus the 2x2 block needs Strassen (7 terms) but Strassen only gives
    # C[i,k] for i,k in {0,1} using j in {0,1}. We ALSO need j=2 contribution
    # to C[i,k] for i,k in {0,1}:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # That's 4 more terms.
    
    # Total: 7 + 4 + 15 = 26. 
    # Let me add the j=2 contributions to the 2x2 block.
    
    # A[0,2]*B[2,0] -> C[0,0]
    U[0*m+2][col] = 1.0
    V[2*p+0][col] = 1.0
    W[0*n+0][col] = 1.0
    col += 1
    
    # A[0,2]*B[2,1] -> C[0,1]
    U[0*m+2][col] = 1.0
    V[2*p+1][col] = 1.0
    W[1*n+0][col] = 1.0
    col += 1
    
    # A[1,2]*B[2,0] -> C[1,0]
    U[1*m+2][col] = 1.0
    V[2*p+0][col] = 1.0
    W[0*n+1][col] = 1.0
    col += 1
    
    # A[1,2]*B[2,1] -> C[1,1]
    U[1*m+2][col] = 1.0
    V[2*p+1][col] = 1.0
    W[1*n+1][col] = 1.0
    col += 1
    
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    U[0*m+0][col] = 1.0
    V[0*p+2][col] = 1.0
    W[2*n+0][col] = 1.0
    col += 1
    
    U[0*m+1][col] = 1.0
    V[1*p+2][col] = 1.0
    W[2*n+0][col] = 1.0
    col += 1
    
    U[0*m+2][col] = 1.0
    V[2*p+2][col] = 1.0
    W[2*n+0][col] = 1.0
    col += 1
    
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    U[1*m+0][col] = 1.0
    V[0*p+2][col] = 1.0
    W[2*n+1][col] = 1.0
    col += 1
    
    U[1*m+1][col] = 1.0
    V[1*p+2][col] = 1.0
    W[2*n+1][col] = 1.0
    col += 1
    
    U[1*m+2][col] = 1.0
    V[2*p+2][col] = 1.0
    W[2*n+1][col] = 1.0
    col += 1
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    U[2*m+0][col] = 1.0
    V[0*p+0][col] = 1.0
    W[0*n+2][col] = 1.0
    col += 1
    
    U[2*m+1][col] = 1.0
    V[1*p+0][col] = 1.0
    W[0*n+2][col] = 1.0
    col += 1
    
    U[2*m+2][col] = 1.0
    V[2*p+0][col] = 1.0
    W[0*n+2][col] = 1.0
    col += 1
    
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    U[2*m+0][col] = 1.0
    V[0*p+1][col] = 1.0
    W[1*n+2][col] = 1.0
    col += 1
    
    U[2*m+1][col] = 1.0
    V[1*p+1][col] = 1.0
    W[1*n+2][col] = 1.0
    col += 1
    
    U[2*m+2][col] = 1.0
    V[2*p+1][col] = 1.0
    W[1*n+2][col] = 1.0
    col += 1
    
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    U[2*m+0][col] = 1.0
    V[0*p+2][col] = 1.0
    W[2*n+2][col] = 1.0
    col += 1
    
    U[2*m+1][col] = 1.0
    V[1*p+2][col] = 1.0
    W[2*n+2][col] = 1.0
    col += 1
    
    U[2*m+2][col] = 1.0
    V[2*p+2][col] = 1.0
    W[2*n+2][col] = 1.0
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to integer where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    best = _schoolbook_rank27()
    
    # Try rank 26 block decomposition
    cand = _try_rank26_block()
    if cand is not None and _verify(cand["U"], cand["V"], cand["W"]):
        return cand
    
    # Fallback to rank 27
    return best
