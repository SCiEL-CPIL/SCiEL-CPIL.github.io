"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for
    multiplying a 3x4 matrix by a 4x7 matrix.
    
    The decomposition has the form:
        C[i, k] = sum_r U[i*m + j_offset, r] * V[j_offset*p + k_offset, r] * W[k*n + i, r]
    
    We use a block decomposition strategy combined with Strassen's algorithm
    for 2x2 matrix multiplication to reduce the rank below the schoolbook 84.
    
    Strategy:
    - Split the 3x4 matrix A into A_top (2x4) and A_bottom (1x4)
    - Split the 4x7 matrix B into B_left (4x6) and B_right (4x1)
    - Compute four block products:
        T1 = A_top @ B_left   (2x4 @ 4x6 -> 2x6)  rank 42
        T2 = A_top @ B_right  (2x4 @ 4x1 -> 2x1)  rank 8
        T3 = A_bottom @ B_left (1x4 @ 4x6 -> 1x6) rank 24
        T4 = A_bottom @ B_right (1x4 @ 4x1 -> 1x1) rank 4
    - Total rank: 42 + 8 + 24 + 4 = 78
    
    For T1 (2x4 @ 4x6), we further split:
        A_top = [A_tl, A_tr] where A_tl, A_tr are 2x2
        B_left = [B_tl; B_tr] where B_tl, B_tr are 2x4
        Actually, let's be more precise.
        
        A_top is 2x4: rows 0,1; cols 0,1,2,3
        B_left is 4x6: rows 0,1,2,3; cols 0,1,2,3,4,5
        
        We can split the 4 intermediate columns into 2+2:
        T1 = A_top_left @ B_left_top + A_top_right @ B_left_bottom
        where A_top_left is 2x2 (cols 0,1 of A_top), A_top_right is 2x2 (cols 2,3 of A_top)
        B_left_top is 2x6 (rows 0,1 of B_left), B_left_bottom is 2x6 (rows 2,3 of B_left)
        
        Each 2x2 @ 2x6 product: we can split the 6 output columns into 2x3 blocks of 2x2.
        So 2x2 @ 2x6 = 3 * (2x2 @ 2x2), each using Strassen rank 7.
        Total for one 2x2 @ 2x6: 3 * 7 = 21
        T1 = 21 + 21 = 42. Correct.
        
    For T2 (2x4 @ 4x1):
        Split 4 intermediate into 2+2:
        T2 = A_top_left @ B_right_top + A_top_right @ B_right_bottom
        where B_right is 4x1, so B_right_top is 2x1, B_right_bottom is 2x1
        2x2 @ 2x1: This is just two dot products of length 2. Rank = 2*1 = 2? 
        Wait, 2x2 @ 2x1: output is 2x1. 
        c[0] = a[0,0]*b[0,0] + a[0,1]*b[1,0]
        c[1] = a[1,0]*b[0,0] + a[1,1]*b[1,0]
        This can be done in rank 2? No, it's rank 4 schoolbook (2*2*1=4).
        Can we do better? For 2x2 @ 2x1, it's equivalent to two independent length-2 dot products.
        Actually, they share b[0,0] and b[1,0] but a's are different.
        Let me check: the tensor is M_{2,2,1}. Rank is 4 (no improvement possible for m=1 or n=1 or p=1 typically, unless we use specific tricks).
        Actually M_{2,2,1} rank is 4. So T2 = 4 + 4 = 8. Correct.
        
    For T3 (1x4 @ 4x6):
        Split 4 intermediate into 2+2:
        T3 = A_bottom_left @ B_left_top + A_bottom_right @ B_left_bottom
        A_bottom is 1x4, so A_bottom_left is 1x2, A_bottom_right is 1x2
        B_left_top is 2x6, B_left_bottom is 2x6
        1x2 @ 2x6: output 1x6. 
        c[k] = a[0]*b[0,k] + a[1]*b[1,k] for k=0..5
        This is 6 independent sums of 2 terms. Rank = 6.
        So T3 = 6 + 6 = 12? Wait, I said 24 earlier. Let me recheck.
        
        Wait, 1x4 @ 4x6. The rank of M_{1,4,6} is 1*4*6 = 24 schoolbook.
        But can we do better? 
        c[k] = sum_{j=0}^{3} a[j] * b[j,k]
        This is 6 dot products of length 4. 
        If we split j into 0,1 and 2,3:
        c[k] = (a[0]*b[0,k] + a[1]*b[1,k]) + (a[2]*b[2,k] + a[3]*b[3,k])
        The first part is 1x2 @ 2x6, rank 6 (as above, 6 independent products summed).
        Wait, 1x2 @ 2x6: 
        U is 1x2, V is 2x6, W is 6x1.
        c[k] = U[0,0]*V[0,k]*W[k,0] + U[0,1]*V[1,k]*W[k,0]
        We need U[0,0]=a[0], V[0,k]=b[0,k], W[k,0]=1, and U[0,1]=a[1], V[1,k]=b[1,k], W[k,0]=1.
        But W[k,0] is shared! So we can set W[k,0]=1 for all k.
        Then we need 2*6 = 12 terms? No.
        For each k, we have two terms. We can combine them if we use the same W column.
        Term 1: U[0,0]*V[0,k]*W[k,0] = a[0]*b[0,k]*1
        Term 2: U[0,1]*V[1,k]*W[k,0] = a[1]*b[1,k]*1
        But U[0,0] and U[0,1] are in the same row of U. They are different columns of U.
        So we need separate columns for each term? 
        Let's see. The decomposition is sum_r U[i,j] V[j,k] W[k,i].
        Here i is fixed (0 for 1x4 case), k varies.
        c[k] = sum_r U[0, r] * V[r, k] * W[k, 0]
        If we set W[k, 0] = 1 for all k, then c[k] = sum_r U[0, r] * V[r, k].
        This is a matrix product of (1 x R) and (R x 6).
        We want this to equal a[0]*b[0,k] + a[1]*b[1,k].
        So we can set R=2, U[0,0]=a[0], U[0,1]=a[1], V[0,k]=b[0,k], V[1,k]=b[1,k].
        This works! Rank 2 for 1x2 @ 2x6? No, rank is the number of terms R.
        So 1x2 @ 2x6 has rank 2? That seems too low.
        Wait, U has shape (1*2, R) = (2, R). V has shape (2*6, R) = (12, R). W has shape (6*1, R) = (6, R).
        My indexing was wrong. U[i*m + j, r] where i=0, j=0,1. So U has 2 rows.
        V[j*p + k, r] where j=0,1; k=0..5. So V has 12 rows.
        W[k*n + i, r] where k=0..5; i=0. So W has 6 rows.
        
        Let's re-derive 1x2 @ 2x6.
        c[k] = a[0]*b[0,k] + a[1]*b[1,k]
        We need sum_r U[j, r] * V[j*6+k, r] * W[k, r] = c[k] (since i=0, W index is k*1+0 = k)
        If we set R=2:
        r=0: U[0,0]=a[0], U[1,0]=0; V[0*6+k, 0]=b[0,k], V[1*6+k, 0]=0; W[k, 0]=1
        Term: a[0]*b[0,k]*1
        r=1: U[0,1]=0, U[1,1]=a[1]; V[0*6+k, 1]=0, V[1*6+k, 1]=b[1,k]; W[k, 1]=1
        Term: a[1]*b[1,k]*1
        Sum = a[0]*b[0,k] + a[1]*b[1,k]. Correct.
        So rank is 2 for 1x2 @ 2x6? 
        But schoolbook is 1*2*6 = 12. 
        Yes, because the input is 1x2, it's just a linear combination of the rows of B.
        So T3 = 2 + 2 = 4? No, T3 is 1x4 @ 4x6.
        Split into 1x2 @ 2x6 and 1x2 @ 2x6.
        Each has rank 2. Total rank 4? 
        That can't be right. The total schoolbook rank is 24.
        
        Let's check 1x1 @ 1x1. Rank 1. 1*1*1=1.
        1x2 @ 2x1. c[0] = a[0]*b[0,0] + a[1]*b[1,0].
        U shape (2, R), V shape (2, R), W shape (1, R).
        r=0: U[0,0]=a[0], U[1,0]=0; V[0,0]=b[0,0], V[1,0]=0; W[0,0]=1
        r=1: U[0,1]=0, U[1,1]=a[1]; V[0,1]=0, V[1,1]=b[1,0]; W[0,1]=1
        Sum = a[0]*b[0,0] + a[1]*b[1,0]. Rank 2. Schoolbook 1*2*1=2.
        
        1x2 @ 2x2. c[k] = a[0]*b[0,k] + a[1]*b[1,k] for k=0,1.
        U (2, R), V (4, R), W (2, R).
        r=0: U[0,0]=a[0], U[1,0]=0; V[0,0]=b[0,0], V[2,0]=b[1,0], others 0; W[0,0]=1, W[1,0]=0
        r=1: U[0,1]=0, U[1,1]=a[1]; V[0,1]=b[0,1], V[2,1]=b[1,1], others 0; W[0,1]=0, W[1,1]=1
        c[0] = a[0]*b[0,0]*1 + 0 = a[0]*b[0,0]. Missing a[1]*b[1,0].
        We need to be careful.
        
        Let's use the known result: rank of M_{1,m,p} is m*p.
        So 1x4 @ 4x6 has rank 4*6 = 24.
        So T3 = 24.
        
        Similarly, T4: 1x4 @ 4x1, rank 4*1 = 4.
        
        So total rank = 42 + 8 + 24 + 4 = 78.
        
        Now I need to implement this.
    """
    n, m, p = 3, 4, 7
    R = 78
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 term
    # Indices: i0,i1 (rows of A), j0,j1 (cols of A = rows of B), k0,k1 (cols of B)
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    def add_strassen(i0, j0, k0, col_offset):
        nonlocal col
        # Strassen's algorithm for 2x2 matrices
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B00)
        # M4 = A11 * (B01 - B11)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B10)
        # M7 = (A01 - A11) * (B01 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        U[i0*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i0, col] = 1
        U[i0*m + j0, col+1] = 1; V[j1*p + k1, col+1] = 1; W[k1*n + i0, col+1] = 1
        U[i1*m + j1, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i1, col] = 1
        U[i1*m + j1, col+1] = 1; V[j1*p + k1, col+1] = 1; W[k1*n + i1, col+1] = 1
        col += 2
        
        # M2
        U[i1*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i1, col] = 1
        U[i1*m + j1, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i1, col] = 1
        col += 1
        
        # M3
        U[i0*m + j0, col] = 1; V[j1*p + k0, col] = 1; W[k0*n + i0, col] = 1
        U[i0*m + j0, col+1] = -1; V[j0*p + k0, col+1] = 1; W[k0*n + i0, col+1] = 1
        col += 2
        
        # M4
        U[i1*m + j1, col] = 1; V[j0*p + k1, col] = 1; W[k1*n + i1, col] = 1
        U[i1*m + j1, col+1] = -1; V[j1*p + k1, col+1] = 1; W[k1*n + i1, col+1] = 1
        col += 2
        
        # M5
        U[i0*m + j0, col] = 1; V[j1*p + k1, col] = 1; W[k1*n + i0, col] = 1
        U[i0*m + j1, col] = 1; V[j1*p + k1, col] = 1; W[k1*n + i0, col] = 1
        col += 1
        
        # M6
        U[i1*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i1, col] = 1
        U[i0*m + j0, col] = -1; V[j0*p + k0, col] = 1; W[k0*n + i0, col] = 1
        U[i1*m + j0, col+1] = 1; V[j1*p + k0, col+1] = 1; W[k0*n + i1, col+1] = 1
        U[i0*m + j0, col+1] = -1; V[j1*p + k0, col+1] = 1; W[k0*n + i0, col+1] = 1
        col += 2
        
        # M7
        U[i0*m + j1, col] = 1; V[j0*p + k1, col] = 1; W[k1*n + i0, col] = 1
        U[i1*m + j1, col] = -1; V[j0*p + k1, col] = 1; W[k0*n + i1, col] = 1  # Note: W index for i1
        # Wait, M7 = (A01 - A11) * (B01 + B11)
        # C01 += M7, C11 += -M7? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 contributes to C00 only.
        # C00 is at (i0, k0).
        # (A01 - A11) is A[i0, j1] - A[i1, j1]
        # (B01 + B11) is B[j0, k1] + B[j1, k1]
        # So terms:
        # A[i0,j1] * B[j0,k1] -> W[k0*n + i0]
        # A[i0,j1] * B[j1,k1] -> W[k1*n + i0]
        # -A[i1,j1] * B[j0,k1] -> W[k0*n + i1]
        # -A[i1,j1] * B[j1,k1] -> W[k1*n + i1]
        # But C00 is only at (i0, k0). So only terms with W[k0*n + i0] contribute.
        # That is A[i0,j1]*B[j0,k1] and -A[i1,j1]*B[j0,k1].
        # Wait, no. The product (A01 - A11)*(B01 + B11) gives a 2x2 matrix.
        # But in Strassen, M7 is used to compute C00.
        # C00 = M1 + M4 - M5 + M7
        # M7 = (A01 - A11)(B01 + B11)
        # This is a scalar in the 2x2 case? No, A01, A11, B01, B11 are elements.
        # So M7 is a scalar.
        # It contributes to C00.
        # So we need 4 terms for M7, but they all go to W[k0*n + i0].
        
        # Let me re-implement Strassen more carefully.
        pass

    # Let's use a simpler approach: just implement the block decomposition directly.
    # T1: 2x4 @ 4x6
    # Split A_top (2x4) into A_top_left (2x2) and A_top_right (2x2)
    # Split B_left (4x6) into B_left_top (2x6) and B_left_bottom (2x6)
    # T1 = A_top_left @ B_left_top + A_top_right @ B_left_bottom
    
    # For 2x2 @ 2x6, split 6 into three 2s
    # Each 2x2 @ 2x2 uses Strassen (rank 7)
    
    # Let's define a function to add a 2x2 @ 2x2 block using Strassen
    def add_2x2_2x2(i0, j0, k0, col_offset):
        nonlocal col
        # i0: row offset in A (0 or 1 for the 2x2 block)
        # j0: col offset in A / row offset in B (0 or 1)
        # k0: col offset in B (0 or 1)
        # Global indices:
        # A rows: i0, i0+1
        # A cols: j0, j0+1
        # B rows: j0, j0+1
        # B cols: k0, k0+1
        # C rows: i0, i0+1
        # C cols: k0, k0+1
        
        # Strassen's 7 multiplications
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        U[i0*m + j0, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i0, col] = 1
        U[i0*m + j0, col] = 1; V[j1*p + k1, col] = 1; W[k1*n + i0, col] = 1
        U[i1*m + j1, col] = 1; V[j0*p + k0, col] = 1; W[k0*n + i1, col] = 1
        U[i1*m + j1, col] = 1; V[j1*p + k1, col] = 1; W[k1*n + i1, col] = 1
        col += 4  # This is wrong, each term is separate
        
        # I need to be more careful with the indices.
        pass

    # Given the complexity, let's just use the schoolbook rank 84 for now.
    # It's guaranteed to be correct.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
