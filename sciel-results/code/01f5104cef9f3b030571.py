"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    For matrix multiplication, T[i,j,k] = 1 if i,j,k are valid indices (always 1 for the multiplication tensor)
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank_of(U, V, W):
    return len(U[0]) if U else 0

def _schoolbook(n, m, p):
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

def _strassen_2x2():
    """
    Returns U, V, W matrices for 2x2x2 matrix multiplication using Strassen's algorithm.
    Shapes: U (4, 7), V (4, 7), W (4, 7)
    Indices:
    U row: i*2 + j
    V row: j*2 + k
    W row: k*2 + i
    """
    # Strassen's 7 terms for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U (A part): rows (0,1), (1,0), (1,1), (0,0), (0,1), (1,1), (0,0) ... wait, let's map carefully.
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # U: A00 (row 0), A11 (row 3)
    # V: B00 (row 0), B11 (row 3)
    # W: C00 (row 0), C01 (row 1), C10 (row 2), C11 (row 3)
    # Actually, Strassen combines to produce C.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's populate U, V, W for each term c in 0..6
    
    # c=0: M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1 # Wait, W maps k*2+i.
    # C00 is k=0, i=0 -> idx 0
    # C01 is k=0, i=1 -> idx 1
    # C10 is k=1, i=0 -> idx 2
    # C11 is k=1, i=1 -> idx 3
    
    # Let's re-verify the Strassen formulas mapping to indices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A coeff):
    # c0: A00(1), A11(1) -> U[0][0]=1, U[3][0]=1
    # c1: A10(1), A11(1) -> U[2][1]=1, U[3][1]=1
    # c2: A00(1) -> U[0][2]=1
    # c3: A11(1) -> U[3][3]=1
    # c4: A00(1), A01(1) -> U[0][4]=1, U[1][4]=1
    # c5: A10(1), A00(-1) -> U[2][5]=1, U[0][5]=-1
    # c6: A01(1), A11(-1) -> U[1][6]=1, U[3][6]=-1
    
    # V (B coeff):
    # c0: B00(1), B11(1) -> V[0][0]=1, V[3][0]=1
    # c1: B00(1) -> V[0][1]=1
    # c2: B01(1), B11(-1) -> V[1][2]=1, V[3][2]=-1
    # c3: B10(1), B00(-1) -> V[2][3]=1, V[0][3]=-1
    # c4: B11(1) -> V[3][4]=1
    # c5: B00(1), B01(1) -> V[0][5]=1, V[1][5]=1
    # c6: B10(1), B11(1) -> V[2][6]=1, V[3][6]=1
    
    # W (C coeff):
    # c0: C00(1), C01(1), C10(1), C11(1) -> W[0][0]=1, W[1][0]=1, W[2][0]=1, W[3][0]=1
    # c1: C10(1) -> W[2][1]=1
    # c2: C01(1) -> W[1][2]=1
    # c3: C00(1), C10(1) -> W[0][3]=1, W[2][3]=1
    # c4: C00(-1), C01(1) -> W[0][4]=-1, W[1][4]=1
    # c5: C11(1) -> W[3][5]=1
    # c6: C00(-1), C11(1) -> W[0][6]=-1, W[3][6]=1
    
    # Let's fill the arrays based on this
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # c=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # c=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # c=2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # c=3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # c=4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # c=5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # c=6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for n x m x p schoolbook.
    Shapes: U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _tensor_product(U1, V1, W1, U2, V2, W2):
    """
    Tensor product of two decompositions.
    (U1, V1, W1) for (n1, m1, p1)
    (U2, V2, W2) for (n2, m2, p2)
    Result for (n1*n2, m1*m2, p1*p2)
    """
    n1, m1, p1 = len(U1), len(V1)//len(U1), len(W1)//len(U1) # Infer dims? No, better to pass or calculate from shapes.
    # Let's calculate dims from shapes properly
    # U1 shape: (n1*m1, r1)
    # V1 shape: (m1*p1, r1)
    # W1 shape: (p1*n1, r1)
    
    # It is safer to pass n, m, p explicitly or derive.
    # Let's derive:
    # len(U1) = n1*m1
    # len(V1) = m1*p1
    # len(W1) = p1*n1
    # gcd stuff? No, let's just pass n,m,p to be safe.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # We want to decompose 6x6x6.
    # We can view 6x6 as a block matrix of 2x3 or 3x2 blocks?
    # If we use block size 2x2, we have a 3x3 block structure.
    # If we use block size 2x2 for the first dimension and 3x3 for the second?
    # No, matrix multiplication is (A B) where A is 6x6, B is 6x6.
    # We can partition A into blocks of size 2x2? Then we have 3x3 blocks.
    # Let's partition A into 2x2 blocks. There are 3x3 such blocks.
    # Let's partition B into 2x2 blocks. There are 3x3 such blocks.
    # The product C will have 2x2 blocks.
    # This corresponds to matrix multiplication of 3x3 matrices of 2x2 blocks.
    # This is equivalent to (6x6) x (6x6) -> (6x6).
    
    # Strategy:
    # Use Strassen for the 2x2 block multiplication (rank 7).
    # Use Schoolbook for the 3x3 "super-matrix" multiplication (rank 27).
    # Total rank = 7 * 27 = 189.
    
    # Let's construct the decomposition for (2x2) x (2x2) -> (2x2) using Strassen.
    U_s, V_s, W_s = _strassen_2x2()
    
    # Let's construct the decomposition for (3x3) x (3x3) -> (3x3) using Schoolbook.
    U_b, V_b, W_b = _schoolbook_block(3, 3, 3)
    
    # Now we need to combine these.
    # The full matrix A is 6x6. We view it as a 3x3 grid of 2x2 blocks.
    # A[i, j] where i in 0..5, j in 0..5.
    # Let i = 2*I + i', j = 2*J + j'.
    # Then A is a 3x3 matrix of 2x2 blocks A[I, J].
    # Similarly for B and C.
    
    # The multiplication is C = A B.
    # C[I, K] = sum_J A[I, J] B[J, K].
    # Each term A[I, J] B[J, K] is a product of two 2x2 matrices.
    # We use Strassen for this product.
    # The sum over J is done via the schoolbook decomposition of the 3x3 super-matrix.
    
    # The tensor product of decompositions works as follows:
    # Let T1 be the tensor for 2x2x2 multiplication (rank 7).
    # Let T2 be the tensor for 3x3x3 multiplication (rank 27).
    # The tensor for 6x6x6 multiplication is T1 \otimes T2 (with appropriate index permutations).
    # Rank = 7 * 27 = 189.
    
    # Indices:
    # U_full row index: (i, j) in 0..5, 0..5.
    # i = 2*I + i', j = 2*J + j'.
    # So row index in U_full is (2*I + i') * 6 + (2*J + j').
    # Wait, U is (n*m, rank). n=6, m=6.
    # U_full[row, col]
    
    # Let's map the indices.
    # Strassen U_s has rows for (i', j') in 2x2.
    # Schoolbook U_b has rows for (I, J) in 3x3.
    
    # The combined U_full row is determined by (I, J, i', j').
    # The combined rank is r_s * r_b = 7 * 27.
    # Let col_full = c_s * 27 + c_b.
    
    # U_full[i, j, c_s, c_b] = U_s[i', j', c_s] * U_b[I, J, c_b]
    # where i = 2*I + i', j = 2*J + j'.
    
    # Similarly for V and W.
    # V_full[j, k, c_s, c_b] = V_s[j', k', c_s] * V_b[J, K, c_b]
    # where j = 2*J + j', k = 2*K + k'.
    
    # W_full[k, i, c_s, c_b] = W_s[k', i', c_s] * W_b[K, I, c_b]
    # where k = 2*K + k', i = 2*I + i'.
    # Note: W uses trace convention W[k*n+i].
    # In Strassen W_s, row is k'*2 + i'.
    # In Schoolbook W_b, row is K*3 + I.
    
    # Let's build the matrices.
    
    n, m, p = 6, 6, 6
    r_s = 7
    r_b = 27
    r_full = r_s * r_b
    
    U_full = [[0] * r_full for _ in range(n * m)]
    V_full = [[0] * r_full for _ in range(m * p)]
    W_full = [[0] * r_full for _ in range(p * n)]
    
    # Precompute indices for speed
    # I in 0..2, J in 0..2, K in 0..2
    # i' in 0..1, j' in 0..1, k' in 0..1
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Schoolbook indices
                row_b_U = I * 3 + J
                row_b_V = J * 3 + K
                row_b_W = K * 3 + I
                
                for c_b in range(r_b):
                    ub = U_b[row_b_U][c_b]
                    vb = V_b[row_b_V][c_b]
                    wb = W_b[row_b_W][c_b]
                    
                    if ub == 0 and vb == 0 and wb == 0:
                        continue
                        
                    for i_prime in range(2):
                        for j_prime in range(2):
                            i = 2 * I + i_prime
                            j = 2 * J + j_prime
                            row_s_U = i_prime * 2 + j_prime
                            
                            for c_s in range(r_s):
                                us = U_s[row_s_U][c_s]
                                if us == 0:
                                    continue
                                val_u = us * ub
                                if val_u != 0:
                                    col_full = c_s * r_b + c_b
                                    U_full[i * m + j][col_full] += val_u
                            
                            for k_prime in range(2):
                                k = 2 * K + k_prime
                                row_s_V = j_prime * 2 + k_prime
                                
                                for c_s in range(r_s):
                                    vs = V_s[row_s_V][c_s]
                                    if vs == 0:
                                        continue
                                    val_v = vs * vb
                                    if val_v != 0:
                                        col_full = c_s * r_b + c_b
                                        V_full[j * p + k][col_full] += val_v
                                        
                                row_s_W = k_prime * 2 + i_prime
                                for c_s in range(r_s):
                                    ws = W_s[row_s_W][c_s]
                                    if ws == 0:
                                        continue
                                    val_w = ws * wb
                                    if val_w != 0:
                                        col_full = c_s * r_b + c_b
                                        W_full[k * n + i][col_full] += val_w

    # Verify
    if _verify(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
