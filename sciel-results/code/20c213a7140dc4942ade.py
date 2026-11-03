"""Tensor decomposition for 4x5x5 matrix multiplication using Strassen embedding."""

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7).
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    U: 4x7, V: 4x7, W: 4x7
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2, C is 2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    # where:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # U[i*m+j, c] where m=2, so row = i*2+j
    # U maps A elements: U[a_row, c] is coefficient of A[a_row] in M_c
    # A rows: 0->a00, 1->a01, 2->a10, 3->a11
    
    # M1: (a00+a11)*(b00+b11)
    # U contributions: a00->1, a11->1
    U[0][0] = 1  # a00 in M1
    U[3][0] = 1  # a11 in M1
    # V[j*p+k, c] where p=2, so row = j*2+k
    # B rows: 0->b00, 1->b01, 2->b10, 3->b11
    # M1: (b00+b11)
    V[0][0] = 1  # b00 in M1
    V[3][0] = 1  # b11 in M1
    # W[k*n+i, c] where n=2, so row = k*2+i
    # C rows: 0->C00, 1->C01, 2->C10, 3->C11
    # M1 contributes +1 to C00 and C11
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # M2: (a10+a11)*b00
    U[2][1] = 1  # a10 in M2
    U[3][1] = 1  # a11 in M2
    V[0][1] = 1  # b00 in M2
    # M2 contributes +1 to C10
    W[2][1] = 1  # C10 += M2
    
    # M3: a00*(b01-b11)
    U[0][2] = 1  # a00 in M3
    V[1][2] = 1  # b01 in M3
    V[3][2] = -1 # b11 in M3
    # M3 contributes +1 to C01, -1 to C11
    W[1][2] = 1  # C01 += M3
    W[3][2] = -1 # C11 -= M3
    
    # M4: a11*(b10-b00)
    U[3][3] = 1  # a11 in M4
    V[2][3] = 1  # b10 in M4
    V[0][3] = -1 # b00 in M4
    # M4 contributes -1 to C00, +1 to C10
    W[0][3] = -1 # C00 -= M4
    W[2][3] = 1  # C10 += M4
    
    # M5: (a00+a01)*b11
    U[0][4] = 1  # a00 in M5
    U[1][4] = 1  # a01 in M5
    V[3][4] = 1  # b11 in M5
    # M5 contributes +1 to C00, +1 to C01
    W[0][4] = 1  # C00 += M5
    W[1][4] = 1  # C01 += M5
    
    # M6: (a10-a00)*(b00+b01)
    U[2][5] = 1  # a10 in M6
    U[0][5] = -1 # a00 in M6
    V[0][5] = 1  # b00 in M6
    V[1][5] = 1  # b01 in M6
    # M6 contributes +1 to C11
    W[3][5] = 1  # C11 += M6
    
    # M7: (a01-a11)*(b10+b11)
    U[1][6] = 1  # a01 in M7
    U[3][6] = -1 # a11 in M7
    V[2][6] = 1  # b10 in M7
    V[3][6] = 1  # b11 in M7
    # M7 contributes +1 to C00, +1 to C11
    W[0][6] = 1  # C00 += M7
    W[3][6] = 1  # C11 += M7
    
    return U, V, W

def _kron(U1, V1, W1, n1, m1, p1, n2, m2, p2):
    """Kronecker product of two tensor decompositions.
    Given decompositions for n1xm1xp1 and n2xm2xp2,
    produces decomposition for (n1*n2)x(m1*m2)x(p2*p1) ... 
    Actually, for matrix multiplication: (n1*n2) x (m1*m2) x (p1*p2)
    where the first matrix is n1*n2 x m1*m2, second is m1*m2 x p1*p2, result n1*n2 x p1*p2.
    
    Wait, the Kronecker product of matrix multiplications:
    If A1 is n1x m1, B1 is m1 x p1, C1 = A1*B1 is n1 x p1
    If A2 is n2x m2, B2 is m2 x p2, C2 = A2*B2 is n2 x p2
    Then A1⊗A2 is (n1*n2) x (m1*m2), B2⊗B1 is (m2*m1) x (p2*p1), 
    and (A1⊗A2)*(B2⊗B1) = (A1*B2')⊗(A2*B1') ... 
    
    Let me think more carefully. For block matrix multiplication:
    A is (n1*n2) x (m1*m2) viewed as n1 x n2 blocks each of size n2 x m1... 
    
    Actually the standard way: If we have multiplications of sizes (n1,m1,p1) and (n2,m2,p2),
    the Kronecker product gives a multiplication of size (n1*n2, m1*m2, p1*p2).
    
    For our case, we want to use the 2x2x2 Strassen to build 4x4x4.
    So n1=m1=p1=2, n2=m2=p2=2, giving 4x4x4.
    """
    r1 = len(V1[0]) if V1 else 0
    r2 = len(V2[0]) if W2 else 0
    # r1 = rank of first, r2 = rank of second
    r1 = len(W1[0])
    r2 = len(W2[0])
    rank = r1 * r2
    
    U = [[0]*rank for _ in range(n1*n2*m1*m2)]
    V = [[0]*rank for _ in range(m1*m2*p1*p2)]
    W = [[0]*rank for _ in range(p1*p2*n1*n2)]
    
    # U[i*m+j, c] where i in 0..n1*n2-1, j in 0..m1*m2-1, m = m1*m2
    # i = i1*n2 + i2, j = j1*m2 + j2
    # c = c1*r2 + c2
    # U[i*m+j, c] = U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    m = m1 * m2
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1 * n2 + i2
            for j1 in range(m1):
                for j2 in range(m2):
                    j = j1 * m2 + j2
                    row = i * m + j
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            U[row][c] = U1[i1*m1+j1][c1] * U2[i2*m2+j2][c2]
    
    # V[j*p+k, c] where j in 0..m1*m2-1, k in 0..p1*p2-1, p = p1*p2
    # j = j1*p2 + j2, k = k1*m2 + k2  -- wait, need to be careful
    # Actually V[j*p+k, c] where p = p1*p2
    # j = j1*p2 + j2, k = k1*m2 + k2? No...
    # Let me re-examine. In the Kronecker product for matrix mult:
    # First matrix A is (n1*n2) x (m1*m2). A[i,j] where i=i1*n2+i2, j=j1*m2+j2
    # Second matrix B is (m1*m2) x (p1*p2). B[j,k] where j=j1*p2+j2, k=k1*m2+k2?
    # Hmm, this is getting confusing. Let me use the standard formulation.
    
    # For the Kronecker product of two matrix multiplications:
    # A (n1*n2 x m1*m2), B (m1*m2 x p1*p2), C = A*B (n1*n2 x p1*p2)
    # A[i1*n2+i2, j1*m2+j2] = A1[i1,j1]*A2[i2,j2]  -- no, that's not right either
    # Actually A is just an arbitrary (n1*n2) x (m1*m2) matrix. The decomposition
    # works on the tensor level.
    
    # Let me just directly construct using the tensor structure.
    # The matrix multiplication tensor T[i,j,k] for n x m x p has T[i,j,k]=1 if they
    # are valid indices. The Kronecker product of two tensors gives the tensor for
    # the product-sized multiplication.
    
    # U1[i1*m1+j1, c1], V1[j1*p1+k1, c1], W1[k1*n1+i1, c1]
    # U2[i2*m2+j2, c2], V2[j2*p2+k2, c2], W2[k2*n2+i2, c2]
    
    # For the combined (n1*n2) x (m1*m2) x (p1*p2) multiplication:
    # i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # c = c1*r2 + c2
    
    # U[i*m+j, c] where m = m1*m2
    # U[i1*n2+i2, j1*m2+j2 combined] -> row = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    # = (i1*m1+j1)*... hmm let me just index directly
    
    # U_combined[row, col] where row = i*(m1*m2) + j, col = c1*r2+c2
    # row = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    # This should equal U1[i1*m1+j1, c1] * U2[i2*m2+j2, c2]
    
    # Let's verify: i1*m1+j1 is the row in U1, i2*m2+j2 is the row in U2.
    # The mapping from (i1,i2,j1,j2) to global row (i1*n2+i2)*(m1*m2)+(j1*m2+j2)
    # And U1 row i1*m1+j1, U2 row i2*m2+j2.
    # This is consistent.
    
    # Similarly for V:
    # V_combined[j*(p1*p2)+k, c] where j=j1*p2+j2, k=k1*m2+k2? 
    # Wait, no. j is the column index of A = row index of B.
    # In the combined problem, j ranges over m1*m2, k ranges over p1*p2.
    # j = j1*p2 + j2, k = k1*m2 + k2
    # V_combined[j*(p1*p2)+k, c] = V1[j1*p1+k1, c1] * V2[j2*p2+k2, c2]
    
    # And W:
    # W_combined[k*(n1*n2)+i, c] where k=k1*p2+k2, i=i1*n2+i2
    # W_combined[k*(n1*n2)+i, c] = W1[k1*n1+i1, c1] * W2[k2*n2+i2, c2]
    
    # Let me rebuild from scratch more carefully.
    U = [[0]*rank for _ in range((n1*n2)*(m1*m2))]
    V = [[0]*rank for _ in range((m1*m2)*(p1*p2))]
    W = [[0]*rank for _ in range((p1*p2)*(n1*n2))]
    
    for c1 in range(r1):
        for c2 in range(r2):
            c = c1 * r2 + c2
            for i1 in range(n1):
                for j1 in range(m1):
                    u1_val = U1[i1*m1+j1][c1]
                    if u1_val == 0:
                        continue
                    for k1 in range(p1):
                        v1_val = V1[j1*p1+k1][c1]
                        if v1_val == 0:
                            continue
                        w1_val = W1[k1*n1+i1][c1]
                        if w1_val == 0:
                            continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                u2_val = U2[i2*m2+j2][c2]
                                if u2_val == 0:
                                    continue
                                for k2 in range(p2):
                                    v2_val = V2[j2*p2+k2][c2]
                                    if v2_val == 0:
                                        continue
                                    w2_val = W2[k2*n2+i2][c2]
                                    if w2_val == 0:
                                        continue
                                    
                                    # U contribution
                                    u_row = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
                                    U[u_row][c] += u1_val * u2_val
                                    
                                    # V contribution
                                    v_row = (j1*p2+j2)*(p1*p2) + (k1*m2+k2)
                                    V[v_row][c] += v1_val * v2_val
                                    
                                    # W contribution
                                    w_row = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
                                    W[w_row][c] += w1_val * w2_val
    
    return U, V, W

def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition using Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    U4, V4, W4 = _kron(U2, V2, W2, 2, 2, 2, 2, 2, 2)
    return U4, V4, W4

def _verify(U, V, W, n, m, p, verbose=False):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all valid (i,j,k)."""
    rank = len(U[0])
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if total != 1:
                    errors += 1
                    if verbose and errors <= 10:
                        print(f"  Error at ({i},{j},{k}): got {total}, expected 1")
    return errors == 0

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _build_4x4x4_strassen()
    rank4 = len(U4[0])  # should be 49
    
    # Verify 4x4x4
    if not _verify(U4, V4, W4, 4, 4, 4):
        # Fallback to schoolbook if Strassen verification fails
        pass
    
    # Now embed into 4x5x5
    # Core: i in 0..3, j in 0..3, k in 0..3 (all < 4)
    # Remaining: terms where k==4 or j==4
    # Total schoolbook terms: 4*5*5 = 100
    # Core terms: 4*4*4 = 64
    # Remaining: 100 - 64 = 36
    
    total_rank = rank4 + 36
    
    U = [[0]*total_rank for _ in range(n*m)]      # 20 x total_rank
    V = [[0]*total_rank for _ in range(m*p)]      # 25 x total_rank
    W = [[0]*total_rank for _ in range(p*n)]      # 20 x total_rank
    
    # Fill first rank4 columns with Strassen 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = i * m + j    # i*5+j
                v_row = j * p + k    # j*5+k
                w_row = k * n + i    # k*4+i
                for c in range(rank4):
                    u_val = U4[i*4+j][c]
                    v_val = V4[j*4+k][c]
                    w_val = W4[k*4+i][c]
                    U[u_row][c] += u_val
                    V[v_row][c] += v_val
                    W[w_row][c] += w_val
    
    # Fill remaining columns with schoolbook for boundary terms
    col = rank4
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this is in the core (already handled by Strassen)
                if k < 4 and j < 4:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    # Verify the full decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback: pure schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
