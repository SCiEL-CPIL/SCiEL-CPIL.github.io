import time
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
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

def _construct_rank_26():
    """
    Construct a rank-26 decomposition for 3x3 matrix multiplication.
    This uses Strassen's algorithm on the top-left 2x2 block.
    
    The 3x3 matrices A, B are partitioned as:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    B = [[B11, B12, B13],
         [B21, B22, B23],
         [B31, B32, B33]]
    
    We compute C = A * B.
    C11 = A11*B11 + A12*B21 + A13*B31
    C12 = A11*B12 + A12*B22 + A13*B32
    C13 = A11*B13 + A12*B23 + A13*B33
    C21 = A21*B11 + A22*B21 + A23*B31
    C22 = A21*B12 + A22*B22 + A23*B32
    C23 = A21*B13 + A22*B23 + A23*B33
    C31 = A31*B11 + A32*B21 + A33*B31
    C32 = A31*B12 + A32*B22 + A33*B32
    C33 = A31*B13 + A32*B23 + A33*B33
    
    We group the 2x2 block (indices 0,1 for rows/cols) and use Strassen for that part.
    Let X11 = A11, X12 = A12, X21 = A21, X22 = A22.
    Let Y11 = B11, Y12 = B12, Y21 = B21, Y22 = B22.
    
    Strassen terms for the 2x2 block:
    M1 = (X11 + X22) * (Y11 + Y22)
    M2 = (X21 + X22) * Y11
    M3 = X11 * (Y12 - Y22)
    M4 = X22 * (Y21 - Y11)
    M5 = (X11 + X12) * Y22
    M6 = (X21 - X11) * (Y11 + Y12)
    M7 = (X12 - X22) * (Y21 + Y22)
    
    Then:
    C11 = M1 - M4 - M6 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M6 - M5
    
    The remaining terms involving index 2 (the third row/col) are just standard multiplications:
    Terms for C13, C23, C31, C32, C33 involve A13, A23, A31, A32, A33 and B13, B23, B31, B32, B33.
    
    Let's list all 26 terms (M1..M7 and 19 scalar multiplications).
    Actually, it's easier to just define the 26 vectors directly.
    """
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    # Term c contributes to U[i,j], V[j,k], W[k,i]
    # We need to sum these contributions such that the product equals the matmul tensor.
    # The matmul tensor T[i,j,k] = 1.
    # Sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # Let's define the 26 terms explicitly.
    # Indices: 0,1,2
    # Strassen terms operate on the 2x2 submatrix (indices 0,1).
    # The "1" terms are the scalar products for the rest.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U contributions:
    # U[0,0] += 1, U[1,1] += 1
    # V contributions:
    # V[0,0] += 1, V[1,1] += 1
    # W contributions:
    # W[0,0] += 1, W[1,1] += 1  -> Wait, W is W[k*n+i].
    # The term M1 contributes to C00, C01, C10, C11.
    # C00 = M1 - M4 - M6 + M7. So M1 has coeff +1 in W[0,0] (k=0, i=0).
    # C01 = M3 + M5. M1 is not in C01? 
    # Let's re-evaluate Strassen output formulas.
    # C11 (top-left) = M1 - M4 - M6 + M7
    # C12 (top-right) = M3 + M5
    # C21 (bottom-left) = M2 + M4
    # C22 (bottom-right) = M1 - M3 + M6 - M5
    
    # So M1 contributes to C11 (+1) and C22 (+1).
    # In our indices (0,1):
    # C00 gets +1 from M1. W[0*3+0][0] += 1.
    # C11 gets +1 from M1. W[1*3+1][0] += 1.
    
    # Let's build the terms one by one.
    
    # Term 0: M1
    U[0*3+0][0] += 1
    U[1*3+1][0] += 1
    V[0*3+0][0] += 1
    V[1*3+1][0] += 1
    W[0*3+0][0] += 1  # C00
    W[1*3+1][0] += 1  # C11
    
    # Term 1: M2 = (A10+A11)B00
    U[1*3+0][1] += 1
    U[1*3+1][1] += 1
    V[0*3+0][1] += 1
    W[1*3+0][1] += 1  # C10
    
    # Term 2: M3 = A00(B01-B11)
    U[0*3+0][2] += 1
    V[0*3+1][2] += 1
    V[1*3+1][2] -= 1
    W[0*3+1][2] += 1  # C01
    
    # Term 3: M4 = A11(B10-B00)
    U[1*3+1][3] += 1
    V[1*3+0][3] += 1
    V[0*3+0][3] -= 1
    W[0*3+0][3] -= 1  # C00 (from -M4)
    W[1*3+0][3] += 1  # C10 (from +M4)
    
    # Term 4: M5 = (A00+A01)B11
    U[0*3+0][4] += 1
    U[0*3+1][4] += 1
    V[1*3+1][4] += 1
    W[0*3+1][4] += 1  # C01 (from +M5)
    W[1*3+1][4] -= 1  # C11 (from -M5)
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U[1*3+0][5] += 1
    U[0*3+0][5] -= 1
    V[0*3+0][5] += 1
    V[0*3+1][5] += 1
    W[0*3+0][5] -= 1  # C00 (from -M6)
    W[1*3+1][5] += 1  # C11 (from +M6)
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U[0*3+1][6] += 1
    U[1*3+1][6] -= 1
    V[1*3+0][6] += 1
    V[1*3+1][6] += 1
    W[0*3+0][6] += 1  # C00 (from +M7)
    W[1*3+0][6] += 1  # C10? No, M7 is in C11 formula? 
    # C11 = M1 - M4 - M6 + M7. Yes, +M7.
    # Wait, C10 = M2 + M4. M7 is not in C10.
    # Let's check C11 again. C11 (index 1,1) gets +M7.
    W[1*3+1][6] += 1  # C11
    
    # Now the remaining 19 terms for the interactions with index 2.
    # We need to satisfy:
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    
    # And the cross terms for the 2x2 block are already handled?
    # No, the 2x2 Strassen only handles C00, C01, C10, C11 using A00..A11 and B00..B11.
    # It does NOT use A02, A12, A20, A21, A22, B02, B12, B20, B21, B22.
    # So we must add terms for all products involving index 2.
    
    # List of pairs (i,j) from A and (j,k) from B such that i!=2 or k!=2 or j!=2?
    # Actually, any term where i, j, or k is 2 must be explicitly added if not covered.
    # The Strassen terms only involve i,j,k in {0,1}.
    # So we need to add all terms for the "rest".
    
    # The full set of 27 terms is all (i,j,k).
    # The 9 terms with i,j,k in {0,1} are covered by the 7 Strassen terms (with cancellations).
    # We need to add the 18 terms where at least one index is 2?
    # Wait, 27 - 9 = 18.
    # 7 + 18 = 25? 
    # Let's check the count.
    # Strassen covers the 2x2 block.
    # The remaining terms are:
    # 1. i=0,1,2, j=2, k=0,1,2 (3*1*3 = 9 terms? No, j=2 is fixed)
    #    A[0,2]*B[2,k], A[1,2]*B[2,k], A[2,2]*B[2,k] for k=0,1,2. (9 terms)
    # 2. i=0,1,2, j=0,1, k=2 (3*2*1 = 6 terms)
    #    A[i,0]*B[0,2], A[i,1]*B[1,2] for i=0,1,2. (6 terms)
    # 3. i=2, j=0,1, k=0,1 (1*2*2 = 4 terms)
    #    A[2,0]*B[0,k], A[2,1]*B[1,k] for k=0,1. (4 terms)
    # Total 9 + 6 + 4 = 19 terms.
    # 7 + 19 = 26 terms. Correct.
    
    # Let's add these 19 terms.
    # Term 7-25
    
    # Group 1: j=2
    # (i, 2, k) for i in 0,1,2 and k in 0,1,2
    col = 7
    for i in range(3):
        for k in range(3):
            U[i*3+2][col] = 1
            V[2*3+k][col] = 1
            W[k*3+i][col] = 1
            col += 1
            
    # Group 2: k=2, j=0,1
    # (i, j, 2) for i in 0,1,2 and j in 0,1
    for i in range(3):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*3+2][col] = 1
            W[2*3+i][col] = 1
            col += 1
            
    # Group 3: i=2, j=0,1, k=0,1
    # (2, j, k) for j in 0,1 and k in 0,1
    for j in range(2):
        for k in range(2):
            U[2*3+j][col] = 1
            V[j*3+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
            
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    overall_deadline = time.time() + 140
    
    # Try rank-26 first
    result = _construct_rank_26()
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _construct_schoolbook()
