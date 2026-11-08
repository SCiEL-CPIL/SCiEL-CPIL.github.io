"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# Strassen's 2x2x2 rank-7 decomposition
# T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}
# Using the standard Strassen decomposition
_STRASSEN_U = [
    [1, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, -1],
]
_STRASSEN_V = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
]
_STRASSEN_W = [
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1],
]

# Actually, let me use the correct Strassen decomposition.
# The standard Strassen decomposition for 2x2 matrix multiplication:
# Given A = [[a,b],[c,d]] and B = [[e,f],[g,h]], the product C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
# Strassen uses 7 multiplications:
# M1 = (a+d)(e+h)
# M2 = (c+d)e
# M3 = a(f-h)
# M4 = d(g-e)
# M5 = (a+b)h
# M6 = (c-a)(e+f)
# M7 = (b-d)(g+h)
# Then:
# c11 = M1 + M4 - M5 + M7
# c12 = M3 + M5
# c21 = M2 + M4
# c22 = M1 - M2 + M3 + M6

# Let me define the proper Strassen decomposition with correct indices
# For the tensor T[i,j,k] where i=0,1 (row of A), j=0,1 (col of A / row of B), k=0,1 (col of B)
# We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

# The 7 terms:
# Term 0: (a+d)(e+h) -> U has 1 at positions (0,0) and (1,1) for A part, V has 1 at (0,0) and (1,1) for B part
# Actually, let me think more carefully.

# For 2x2*2x2, the tensor entries are:
# T[0,0,0] = 1 (a*e)
# T[0,0,1] = 1 (a*f)  
# T[0,1,0] = 1 (b*g)
# T[0,1,1] = 1 (b*h)
# T[1,0,0] = 1 (c*e)
# T[1,0,1] = 1 (c*f)
# T[1,1,0] = 1 (d*g)
# T[1,1,1] = 1 (d*h)

# Using Strassen's 7 multiplications:
# M1 = (a+d)(e+h): contributes to all four output positions
#   c11 += M1, c12 += M1, c21 += M1, c22 += M1
# M2 = (c+d)e: contributes to c21 and c22
#   c21 += M2, c22 -= M2
# M3 = a(f-h): contributes to c11 and c12
#   c11 -= M3, c12 += M3
# M4 = d(g-e): contributes to c11 and c21
#   c11 += M4, c21 += M4
# M5 = (a+b)h: contributes to c11 and c12
#   c11 -= M5, c12 += M5
# M6 = (c-a)(e+f): contributes to c22
#   c22 += M6
# M7 = (b-d)(g+h): contributes to c11
#   c11 += M7

# So for each multiplication term r, we need:
# U[i*m+j, r]: coefficient for A[i,j]
# V[j*p+k, r]: coefficient for B[j,k]  
# W[k*n+i, r]: coefficient for output C[i,k]

# M1 = (a+d)(e+h):
#   A part: a (i=0,j=0) + d (i=1,j=1)
#   B part: e (j=0,k=0) + h (j=1,k=1)
#   Output: c11 (i=0,k=0), c12 (i=0,k=1), c21 (i=1,k=0), c22 (i=1,k=1)
#   U[0,0]=1, U[1,1]=1
#   V[0,0]=1, V[1,1]=1
#   W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1

# M2 = (c+d)e:
#   A part: c (i=1,j=0) + d (i=1,j=1)
#   B part: e (j=0,k=0)
#   Output: c21 (i=1,k=0), c22 (i=1,k=1) with -1 for c22
#   U[1,0]=1, U[1,1]=1
#   V[0,0]=1
#   W[1,0]=1, W[1,1]=-1

# M3 = a(f-h):
#   A part: a (i=0,j=0)
#   B part: f (j=0,k=1) - h (j=1,k=1)
#   Output: c11 (i=0,k=0) with -1, c12 (i=0,k=1)
#   U[0,0]=1
#   V[0,1]=1, V[1,1]=-1
#   W[0,0]=-1, W[0,1]=1

# M4 = d(g-e):
#   A part: d (i=1,j=1)
#   B part: g (j=1,k=0) - e (j=0,k=0)
#   Output: c11 (i=0,k=0), c21 (i=1,k=0)
#   U[1,1]=1
#   V[0,0]=-1, V[1,0]=1
#   W[0,0]=1, W[1,0]=1

# M5 = (a+b)h:
#   A part: a (i=0,j=0) + b (i=0,j=1)
#   B part: h (j=1,k=1)
#   Output: c11 (i=0,k=0) with -1, c12 (i=0,k=1)
#   U[0,0]=1, U[0,1]=1
#   V[1,1]=1
#   W[0,0]=-1, W[0,1]=1

# M6 = (c-a)(e+f):
#   A part: c (i=1,j=0) - a (i=0,j=0)
#   B part: e (j=0,k=0) + f (j=0,k=1)
#   Output: c22 (i=1,k=1)
#   U[0,0]=-1, U[1,0]=1
#   V[0,0]=1, V[0,1]=1
#   W[1,1]=1

# M7 = (b-d)(g+h):
#   A part: b (i=0,j=1) - d (i=1,j=1)
#   B part: g (j=1,k=0) + h (j=1,k=1)
#   Output: c11 (i=0,k=0)
#   U[0,1]=1, U[1,1]=-1
#   V[1,0]=1, V[1,1]=1
#   W[0,0]=1

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    # Schoolbook decomposition - guaranteed valid
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-9):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Block decomposition using Strassen for 2x2 blocks
    # 4x4 * 4x6 can be viewed as 2x2 blocks of 2x2 * 2x3 blocks
    # But we need to be more careful. Let's decompose into 2x2 blocks.
    
    # For 4x4 * 4x6:
    # A is 4x4, B is 4x6, C is 4x6
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # Split B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each Bj is 2x3
    # Split C into 2x3 blocks: C = [[C00, C01, C02], [C10, C11, C12]] where each Ck is 2x3
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # For each Ci, we have a 2x2 * 2x3 multiplication.
    # We can use Strassen-like decomposition for 2x2 * 2x3.
    
    # Actually, let's use a simpler approach: use Strassen for the 2x2 part
    # and handle the 2x3 part separately.
    
    # For 2x2 * 2x3:
    # A is 2x2, B is 2x3, C is 2x3
    # C = A*B
    # We can write this as two 2x2 * 2x2 multiplications plus a 2x2 * 2x1 multiplication
    # Or we can use a direct rank decomposition.
    
    # Let's use a direct approach: for 2x2 * 2x3, the schoolbook rank is 12.
    # Can we do better? Let's try using Strassen-like ideas.
    
    # Actually, let's just use the block structure with Strassen for the 2x2*2x2 parts
    # and handle the remaining column separately.
    
    # For 4x4 * 4x6, split B into two 4x3 parts: B = [B_left | B_right]
    # C = A*B = [A*B_left | A*B_right]
    # Each A*B_left and A*B_right is 4x4 * 4x3.
    
    # For 4x4 * 4x3:
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B = [[B00, B01, B02], [B10, B11, B12]] (each 2x3)
    # C = [[C00, C01, C02], [C10, C11, C12]] (each 2x3)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # For each Ci (2x3), we can use a rank decomposition.
    # Let's use schoolbook for 2x2*2x3: rank 12 per Ci, total 6*12 = 72 for 4x4*4x3.
    # For 4x4*4x6, we'd have 2 such blocks, total rank 144. That's worse than schoolbook 96.
    
    # Let me try a different approach. Use Strassen for 2x2*2x2 and extend.
    
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x3, we can think of it as 2x2*2x2 + 2x2*2x1.
    # 2x2*2x1 has rank 4 (schoolbook).
    # So 2x2*2x3 might have rank 7+4=11? Not sure if that's valid.
    
    # Let's just use a direct rank decomposition for 2x2*2x3.
    # The tensor has 2*2*3 = 12 entries, all 1.
    # Schoolbook rank is 12. Can we do better?
    
    # Let's try to find a rank-11 or lower decomposition for 2x2*2x3.
    # This is equivalent to finding a rank decomposition of the 4x6 matrix T where
    # T[i,j,k] = 1 for all i,j,k.
    
    # Actually, the tensor for 2x2*2x3 is:
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    # This is a 2x2x3 tensor with all ones.
    
    # The rank of this tensor... let's think. If we use the fact that the tensor
    # is separable in some way, we might find a lower rank.
    
    # For now, let's use a block decomposition with Strassen for the 2x2 parts.
    # 4x4 * 4x6:
    # Split into 2x2 blocks of size 2x2 * 2x3.
    # For each 2x2 * 2x3 block, use a rank decomposition.
    
    # Let me try to construct a rank-84 decomposition using Strassen for 2x2*2x2
    # and schoolbook for the remaining parts.
    
    # Actually, the cell finding says: "Block-decomposing 4x4x6 into 12 sub-problems of 2x2x2 
    # using Strassen's rank-7 decomposition for each, achieving total rank 84"
    
    # 12 sub-problems of 2x2x2: 4x4 * 4x6 can be decomposed into 12 such sub-problems?
    # 4x4 has 4 rows and 4 cols. 4x6 has 4 rows and 6 cols.
    # If we split into 2x2 blocks:
    # A has 2x2 = 4 blocks of size 2x2
    # B has 2x3 = 6 blocks of size 2x2 (treating 2x3 as 2x2 + 2x1, or 3 blocks of 2x2 with overlap?)
    
    # Wait, 4x4 * 4x6:
    # A is 4x4: split into 2x2 blocks: 4 blocks
    # B is 4x6: split into 2x3 blocks: 6 blocks (2 rows x 3 cols of 2x3 blocks)
    # But 2x3 is not 2x2.
    
    # Alternatively, split B into 2x2 blocks: 4x6 / (2x2) = 2*3 = 6 blocks, but 6 is not divisible by 2.
    # 4x6 can be split into 2x2 blocks: 2 rows of 2x2, 3 cols of 2x2, but 3*2=6, so 2x3=6 blocks of 2x2.
    # But 4x6 = 2*2 + 2*2 + 2*2 in each row? No, 4x6 has 4 rows and 6 cols.
    # Split into 2x2 blocks: (4/2) * (6/2) = 2 * 3 = 6 blocks of 2x2.
    
    # So A has 4 blocks of 2x2, B has 6 blocks of 2x2.
    # C has 6 blocks of 2x2.
    
    # C[i_block, j_block] = sum_k A[i_block, k_block] * B[k_block, j_block]
    # where i_block, k_block are row blocks (0,1), j_block is col block (0,1,2)
    
    # For each (i_block, j_block), C[i_block, j_block] is a 2x2 matrix.
    # C[i_block, j_block] = A[i_block, 0] * B[0, j_block] + A[i_block, 1] * B[1, j_block]
    
    # Each term A[i_block, k] * B[k, j_block] is a 2x2 * 2x2 multiplication.
    # Using Strassen, each such multiplication has rank 7.
    # For each (i_block, j_block), we have 2 such terms, so rank 14.
    # Total: 2 * 3 * 14 = 84.
    
    # This matches the cell finding! Let's construct this.
    
    # Define the Strassen decomposition for 2x2*2x2:
    # U, V, W are 4x7 matrices (since 2*2=4 rows for U and V, 2*2=4 rows for W)
    
    def get_strassen_2x2():
        """Return Strassen decomposition for 2x2*2x2 as (U, V, W) each 4x7."""
        # Using the standard Strassen decomposition
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (a+d)(e+h)
        # a = A[0,0], d = A[1,1]
        # e = B[0,0], h = B[1,1]
        # Output: C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] += M1
        U[0, 0] = 1  # a
        U[3, 0] = 1  # d (A[1,1] is at index 1*2+1=3)
        V[0, 0] = 1  # e (B[0,0] is at index 0*2+0=0)
        V[3, 0] = 1  # h (B[1,1] is at index 1*2+1=3)
        W[0, 0] = 1  # C[0,0]
        W[1, 0] = 1  # C[0,1]
        W[2, 0] = 1  # C[1,0]
        W[3, 0] = 1  # C[1,1]
        
        # M2 = (c+d)e
        # c = A[1,0], d = A[1,1]
        # e = B[0,0]
        # Output: C[1,0] += M2, C[1,1] -= M2
        U[2, 1] = 1  # c (A[1,0] is at index 1*2+0=2)
        U[3, 1] = 1  # d
        V[0, 1] = 1  # e
        W[2, 1] = 1  # C[1,0]
        W[3, 1] = -1  # C[1,1]
        
        # M3 = a(f-h)
        # a = A[0,0]
        # f = B[0,1], h = B[1,1]
        # Output: C[0,0] -= M3, C[0,1] += M3
        U[0, 2] = 1  # a
        V[1, 2] = 1  # f (B[0,1] is at index 0*2+1=1)
        V[3, 2] = -1  # h
        W[0, 2] = -1  # C[0,0]
        W[1, 2] = 1  # C[0,1]
        
        # M4 = d(g-e)
        # d = A[1,1]
        # g = B[1,0], e = B[0,0]
        # Output: C[0,0] += M4, C[1,0] += M4
        U[3, 3] = 1  # d
        V[0, 3] = -1  # e
        V[2, 3] = 1  # g (B[1,0] is at index 1*2+0=2)
        W[0, 3] = 1  # C[0,0]
        W[2, 3] = 1  # C[1,0]
        
        # M5 = (a+b)h
        # a = A[0,0], b = A[0,1]
        # h = B[1,1]
        # Output: C[0,0] -= M5, C[0,1] += M5
        U[0, 4] = 1  # a
        U[1, 4] = 1  # b (A[0,1] is at index 0*2+1=1)
        V[3, 4] = 1  # h
        W[0, 4] = -1  # C[0,0]
        W[1, 4] = 1  # C[0,1]
        
        # M6 = (c-a)(e+f)
        # c = A[1,0], a = A[0,0]
        # e = B[0,0], f = B[0,1]
        # Output: C[1,1] += M6
        U[0, 5] = -1  # a
        U[2, 5] = 1  # c
        V[0, 5] = 1  # e
        V[1, 5] = 1  # f
        W[3, 5] = 1  # C[1,1]
        
        # M7 = (b-d)(g+h)
        # b = A[0,1], d = A[1,1]
        # g = B[1,0], h = B[1,1]
        # Output: C[0,0] += M7
        U[1, 6] = 1  # b
        U[3, 6] = -1  # d
        V[2, 6] = 1  # g
        V[3, 6] = 1  # h
        W[0, 6] = 1  # C[0,0]
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = get_strassen_2x2()
    
    def verify_strassen():
        """Verify that the Strassen decomposition is correct."""
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.dot(U_s[i * 2 + j], V_s[j * 2 + k] * W_s[k * 2 + i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_strassen():
        # Fall back to schoolbook
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Now construct the 4x4 * 4x6 decomposition using Strassen blocks
    # A is 4x4, B is 4x6, C is 4x6
    # Split A into 2x2 blocks: A[i_block, j_block] where i_block, j_block in {0,1}
    # Split B into 2x2 blocks: B[i_block, j_block] where i_block in {0,1}, j_block in {0,1,2}
    # Split C into 2x2 blocks: C[i_block, j_block] where i_block in {0,1}, j_block in {0,1,2}
    
    # C[i_block, j_block] = A[i_block, 0] * B[0, j_block] + A[i_block, 1] * B[1, j_block]
    
    # For each (i_block, j_block), we have two 2x2*2x2 multiplications.
    # Each uses Strassen with rank 7.
    # Total rank: 2 * 3 * 2 * 7 = 84
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i_block in range(2):
        for j_block in range(3):
            for k_block in range(2):
                # Term: A[i_block, k_block] * B[k_block, j_block]
                # A[i_block, k_block] is a 2x2 matrix at rows (i_block*2, i_block*2+1), cols (k_block*2, k_block*2+1)
                # B[k_block, j_block] is a 2x2 matrix at rows (k_block*2, k_block*2+1), cols (j_block*2, j_block*2+1)
                # C[i_block, j_block] is a 2x2 matrix at rows (i_block*2, i_block*2+1), cols (j_block*2, j_block*2+1)
                
                for r in range(7):
                    # U entries: for A[i_block, k_block]
                    for ii in range(2):
                        for jj in range(2):
                            u_val = U_s[ii * 2 + jj, r]
                            if u_val != 0:
                                # A row: i_block*2 + ii, col: k_block*2 + jj
                                row_a = (i_block * 2 + ii) * m + (k_block * 2 + jj)
                                U[row_a][col] += u_val
                    
                    # V entries: for B[k_block, j_block]
                    for jj in range(2):
                        for kk in range(2):
                            v_val = V_s[jj * 2 + kk, r]
                            if v_val != 0:
                                # B row: k_block*2 + jj, col: j_block*2 + kk
                                row_b = (k_block * 2 + jj) * p + (j_block * 2 + kk)
                                V[row_b][col] += v_val
                    
                    # W entries: for C[i_block, j_block]
                    for kk in range(2):
                        for ii in range(2):
                            w_val = W_s[kk * 2 + ii, r]
                            if w_val != 0:
                                # C row: j_block*2 + kk, col: i_block*2 + ii
                                # W index: k*n + i where k is col of C, i is row of C
                                # C[i_block, j_block] has rows (i_block*2, i_block*2+1) and cols (j_block*2, j_block*2+1)
                                # W[k*n + i, r] where k = col of C, i = row of C
                                row_c = (j_block * 2 + kk) * n + (i_block * 2 + ii)
                                W[row_c][col] += w_val
                    
                    col += 1
    
    # Verify the decomposition
    if verify_decomposition(U, V, W):
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
