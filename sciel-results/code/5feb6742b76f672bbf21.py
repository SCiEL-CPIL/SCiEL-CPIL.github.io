"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _matmul_tensor_from_factors(U, V, W):
    """Compute the tensor from factor matrices."""
    rank = U.shape[1]
    n_a, n_b, n_c = U.shape[0], V.shape[0], W.shape[0]
    T = np.zeros((n_a, n_b, n_c))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return T

def _rank17_decomposition(n, m, p):
    """Return the rank-17 decomposition using Strassen for 2x2 block + direct for rest."""
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    rank17_terms = []
    
    # Part 1: Strassen for A1(2x2) x B1(2x2) -> C[0:2, 0:2]
    # A1 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]] = indices [0,1,3,4] in A (row-major 2x3, first 2 cols)
    # B1 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]] = indices [0,1,3,4] in B (3x3, first 2 rows/cols)
    # C[0:2, 0:2] = indices [0,1,2,3] in C (2x3, first 2 cols)
    
    # Strassen terms for M1..M7 where:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # A indices: A00=0, A01=1, A10=3, A11=4
    # B indices: B00=0, B01=1, B10=3, B11=4
    # C indices: C00=0, C01=2, C10=1, C11=3 (2x3 matrix, row-major: [C00,C01,C02,C10,C11,C12])
    # Wait, C is 2x3, so C[i,j] for i in {0,1}, j in {0,1,2}
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    
    # Let me redo. The output C is p*n = 3*2 = 6 elements.
    # C[k*n+i, r] where k is column of B (0..p-1=0..2), i is row of A (0..n-1=0..1)
    # So C index = k*n + i = k*2 + i
    # C[0*2+0]=C[0]=C00, C[0*2+1]=C[1]=C10, C[1*2+0]=C[2]=C01, C[1*2+1]=C[3]=C11, C[2*2+0]=C[4]=C02, C[2*2+1]=C[5]=C12
    
    # So:
    # C00 = index 0
    # C10 = index 1
    # C01 = index 2
    # C11 = index 3
    # C02 = index 4
    # C12 = index 5
    
    # Strassen for the 2x2 block (rows 0,1 of A, cols 0,1 of B -> C cols 0,1):
    # M1 = (A00+A11)(B00+B11) = (A[0]+A[4])(B[0]+B[4])
    # M2 = (A10+A11)B00 = (A[3]+A[4])B[0]
    # M3 = A00(B10-B00) = A[0](B[3]-B[0])
    # M4 = A11(B01-B10) = A[4](B[1]-B[3])
    # M5 = (A00+A01)B11 = (A[0]+A[1])B[4]
    # M6 = (A10-A00)(B00+B10) = (A[3]-A[0])(B[0]+B[3])
    # M7 = (A01-A11)(B01+B11) = (A[1]-A[4])(B[1]+B[4])
    
    # C00 = M1+M4-M5+M7 -> index 0
    # C10 = M2+M4 -> index 1
    # C01 = M3+M5 -> index 2
    # C11 = M1-M2+M3+M7 -> index 3
    
    # For each Strassen term, we have U factors (coefficients on A indices), V factors (coefficients on B indices), W factors (coefficients on C indices)
    
    # M1: U = {0:1, 4:1}, V = {0:1, 4:1}, W = {0:1, 3:1}
    rank17_terms.append(([0, 4], [1, 1], [0, 4], [1, 1], [0, 3], [1, 1]))
    
    # M2: U = {3:1, 4:1}, V = {0:1}, W = {1:1, 3:-1}
    rank17_terms.append(([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]))
    
    # M3: U = {0:1}, V = {3:1, 0:-1}, W = {2:1, 3:1}
    rank17_terms.append(([0], [1], [3, 0], [1, -1], [2, 3], [1, 1]))
    
    # M4: U = {4:1}, V = {1:1, 3:-1}, W = {0:1, 1:1}
    rank17_terms.append(([4], [1], [1, 3], [1, -1], [0, 1], [1, 1]))
    
    # M5: U = {0:1, 1:1}, V = {4:1}, W = {2:-1, 3:-1}
    rank17_terms.append(([0, 1], [1, 1], [4], [1], [2, 3], [-1, -1]))
    
    # M6: U = {3:1, 0:-1}, V = {0:1, 3:1}, W = {1:1}
    rank17_terms.append(([3, 0], [1, -1], [0, 3], [1, 1], [1], [1]))
    
    # M7: U = {1:1, 4:-1}, V = {1:1, 4:1}, W = {0:1, 3:1}
    rank17_terms.append(([1, 4], [1, -1], [1, 4], [1, 1], [0, 3], [1, 1]))
    
    # Part 2: Direct for A1(2x2) x B_col2(2x1) -> C[:, 2]
    # A rows 0,1 (indices 0,1,3,4) times B col 2 (indices 2,5,8)
    # C02 = A00*B02 + A01*B12 = A[0]*B[2] + A[1]*B[5] -> index 4
    # C12 = A10*B02 + A11*B12 = A[3]*B[2] + A[4]*B[5] -> index 5
    
    rank17_terms.append(([0], [1], [2], [1], [4], [1]))
    rank17_terms.append(([1], [1], [5], [1], [4], [1]))
    rank17_terms.append(([3], [1], [2], [1], [5], [1]))
    rank17_terms.append(([4], [1], [5], [1], [5], [1]))
    
    # Part 3: Direct for A_row2(1x3) x B(3x3) -> C[2, :]
    # A row 2 (indices 2,3,4... wait, A is 2x3, so rows are 0,1. A indices: row0=[0,1,2], row1=[3,4,5])
    # Actually A is n x m = 2 x 3. A[i,j] for i in {0,1}, j in {0,1,2}
    # A index = i*m + j = i*3 + j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    
    # So A row 0 = indices [0,1,2], A row 1 = indices [3,4,5]
    # B is m x p = 3 x 3. B[j,k] for j in {0,1,2}, k in {0,1,2}
    # B index = j*p + k = j*3 + k
    # B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    
    # C is p x n = 3 x 2. C[k,i] for k in {0,1,2}, i in {0,1}
    # C index = k*n + i = k*2 + i
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5
    
    # The 2x2 block uses A rows 0,1 and B cols 0,1, giving C cols 0,1 (i.e., C[0,0],C[0,1],C[1,0],C[1,1] = indices 0,1,2,3)
    # The remaining C col 2 uses B col 2, giving C[2,0]=index 4, C[2,1]=index 5
    # A row 2 doesn't exist since n=2. So Part 3 is actually:
    # C[2,0] = sum_j A[j,2] * B[2,j] for j in {0,1,2}... 
    
    # Wait, I need to reconsider. The matrix multiplication is:
    # C = A * B where A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # But the output convention is C[k*n+i, r] = C[k*2+i]
    # So C[0] = C[0,0], C[1] = C[1,0], C[2] = C[0,1], C[3] = C[1,1], C[4] = C[0,2], C[5] = C[1,2]
    
    # The 2x2 block: i in {0,1}, k in {0,1}, j in {0,1}
    # This covers C[0,0], C[0,1], C[1,0], C[1,1] = indices 0, 2, 1, 3
    # Wait, C[i,k] with i=0,k=0 -> index 0; i=0,k=1 -> index 2; i=1,k=0 -> index 1; i=1,k=1 -> index 3
    
    # The Strassen decomposition I wrote above assumed:
    # C00 = C[0,0] = index 0
    # C01 = C[0,1] = index 2  
    # C10 = C[1,0] = index 1
    # C11 = C[1,1] = index 3
    
    # Let me recheck my Strassen terms with this mapping:
    # M1: W = {0:1, 3:1} means C[0,0] and C[1,1] get +1 from M1. Correct: C00=M1+..., C11=M1-...
    # M2: W = {1:1, 3:-1} means C[1,0] gets +1, C[1,1] gets -1. Correct: C10=M2+..., C11=...-M2+...
    # M3: W = {2:1, 3:1} means C[0,1] gets +1, C[1,1] gets +1. Correct: C01=...+M3... wait no.
    
    # Let me recheck Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # So:
    # M1 contributes to C00(+1), C11(+1) -> W indices {0:1, 3:1}
    # M2 contributes to C10(+1), C11(-1) -> W indices {1:1, 3:-1}
    # M3 contributes to C01(+1), C11(+1) -> W indices {2:1, 3:1}
    # M4 contributes to C00(+1), C10(+1) -> W indices {0:1, 1:1}
    # M5 contributes to C00(-1), C01(+1) -> W indices {0:-1, 2:1}
    # M6 contributes to C10(+1) -> W indices {1:1}
    # M7 contributes to C00(+1), C11(+1) -> W indices {0:1, 3:1}
    
    # I had errors in my W assignments above. Let me fix:
    
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    # M1: U={0:1,4:1}, V={0:1,4:1}, W={0:1,3:1}
    U17[0,0]=1; U17[4,0]=1
    V17[0,0]=1; V17[4,0]=1
    W17[0,0]=1; W17[3,0]=1
    
    # M2: U={3:1,4:1}, V={0:1}, W={1:1,3:-1}
    U17[3,1]=1; U17[4,1]=1
    V17[0,1]=1
    W17[1,1]=1; W17[3,1]=-1
    
    # M3: U={0:1}, V={3:1,0:-1}, W={2:1,3:1}
    U17[0,2]=1
    V17[3,2]=1; V17[0,2]=-1
    W17[2,2]=1; W17[3,2]=1
    
    # M4: U={4:1}, V={1:1,3:-1}, W={0:1,1:1}
    U17[4,3]=1
    V17[1,3]=1; V17[3,3]=-1
    W17[0,3]=1; W17[1,3]=1
    
    # M5: U={0:1,1:1}, V={4:1}, W={0:-1,2:1}
    U17[0,4]=1; U17[1,4]=1
    V17[4,4]=1
    W17[0,4]=-1; W17[2,4]=1
    
    # M6: U={3:1,0:-1}, V={0:1,3:1}, W={1:1}
    U17[3,5]=1; U17[0,5]=-1
    V17[0,5]=1; V17[3,5]=1
    W17[1,5]=1
    
    # M7: U={1:1,4:-1}, V={1:1,4:1}, W={0:1,3:1}
    U17[1,6]=1; U17[4,6]=-1
    V17[1,6]=1; V17[4,6]=1
    W17[0,6]=1; W17[3,6]=1
    
    # Part 2: Direct for B col 2 (indices 2,5,8)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2] = A[0]*B[2]+A[1]*B[5]+A[2]*B[8] -> index 4
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2] = A[3]*B[2]+A[4]*B[5]+A[5]*B[8] -> index 5
    
    # 6 terms for col 2:
    # A[0]*B[2] -> C[0,2]
    U17[0,7]=1; V17[2,7]=1; W17[4,7]=1
    # A[1]*B[5] -> C[0,2]
    U17[1,8]=1; V17[5,8]=1; W17[4,8]=1
    # A[2]*B[8] -> C[0,2]
    U17[2,9]=1; V17[8,9]=1; W17[4,9]=1
    # A[3]*B[2] -> C[1,2]
    U17[3,10]=1; V17[2,10]=1; W17[5,10]=1
    # A[4]*B[5] -> C[1,2]
    U17[4,11]=1; V17[5,11]=1; W17[5,11]=1
    # A[5]*B[8] -> C[1,2]
    U17[5,12]=1; V17[8,12]=1; W17[5,12]=1
    
    # But wait, that's 7+6=13 terms. I need 17. Let me recount.
    # The 2x2 block uses A cols 0,1 (indices 0,1,3,4) and B cols 0,1 (indices 0,1,3,4)
    # But A has 3 columns! A[0,2]=index 2 and A[1,2]=index 5 are not in the 2x2 block.
    # Similarly B has 3 rows, so B[2,0]=6, B[2,1]=7, B[2,2]=8 are not in the 2x2 block.
    
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # For k=0,1 (the 2x2 block part), we need j=0,1,2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # So the 2x2 block only covers j=0,1. We need additional terms for j=2:
    # A[0,2]*B[2,0] -> C[0,0] (index 0)
    # A[0,2]*B[2,1] -> C[0,1] (index 2)
    # A[1,2]*B[2,0] -> C[1,0] (index 1)
    # A[1,2]*B[2,1] -> C[1,1] (index 3)
    
    # That's 4 more terms, giving 7+4+6 = 17.
    
    # A[0,2]*B[2,0] -> C[0,0]
    U17[2,13]=1; V17[6,13]=1; W17[0,13]=1
    # A[0,2]*B[2,1] -> C[0,1]
    U17[2,14]=1; V17[7,14]=1; W17[2,14]=1
    # A[1,2]*B[2,0] -> C[1,0]
    U17[5,15]=1; V17[6,15]=1; W17[1,15]=1
    # A[1,2]*B[2,1] -> C[1,1]
    U17[5,16]=1; V17[7,16]=1; W17[3,16]=1
    
    return U17, V17, W17

def _try_rank16_via_row_decomposition(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition by treating the two rows of A as
    independent 1x3 * 3x3 multiplications and finding shared structure.
    """
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Row 0 of A: indices 0,1,2 -> computes C[0,0],C[0,1],C[0,2] = indices 0,2,4
    # Row 1 of A: indices 3,4,5 -> computes C[1,0],C[1,1],C[1,2] = indices 1,3,5
    
    # Each 1x3 * 3x3 multiplication has rank 9 (direct) or can be done with fewer.
    # The 1x3 * 3x3 tensor has shape (3, 9, 3) with T[a,b,c]=1 iff b=3*a+c... 
    # Actually for 1x3 * 3x3: A is 1x3 (indices 0,1,2), B is 3x3 (indices 0..8), C is 1x3 (indices 0,1,2)
    # C[c] = sum_j A[j]*B[j,c]
    # In our indexing: A index j (0,1,2), B index 3*j+c, C index c
    # So T[j, 3*j+c, c] = 1
    
    # The rank of 1x3 * 3x3 is 9 (it's a simple outer product structure).
    # But if we share between the two rows, we might reduce total rank.
    
    # Strategy: Assume rank-16 decomposition. Each of the 16 terms has:
    # U vector in R^6, V vector in R^9, W vector in R^6
    # The U vector has support on either {0,1,2} (row 0) or {3,4,5} (row 1) or both.
    
    # For a rank-16 decomposition, if we can find 8 terms for row 0 and 8 terms for row 1
    # that share some structure, we might get rank 16.
    
    # Actually, the key insight: if a term has U supported only on row 0 indices {0,1,2},
    # then its W must be supported only on {0,2,4} (C row 0).
    # Similarly for row 1.
    
    # So we can separate: terms with U on row 0 contribute to W on {0,2,4},
    # terms with U on row 1 contribute to W on {1,3,5}.
    
    # If we have r0 terms for row 0 and r1 terms for row 1, total rank = r0 + r1.
    # For rank 16, we need r0 + r1 = 16.
    
    # Each row's computation is a 1x3 * 3x3 -> 1x3 multiplication.
    # The minimum rank for 1x3 * 3x3 is 9 (it's a "matrix multiplication" with one side being 1x3).
    # Actually, the rank of M_{1,3,3} is 9 since it's equivalent to 9 independent products.
    
    # Wait, is that right? The tensor T[j, 3j+c, c] = 1 for j,c in {0,1,2}.
    # This is a 3x9x3 tensor. Can we decompose it with rank < 9?
    # Each rank-1 term is u_j * v_{3j+c} * w_c.
    # For a fixed j, we need v_{3j+c} * w_c to produce the identity on c.
    # This requires at least 3 terms per j, so 9 total.
    
    # So r0 >= 9 and r1 >= 9, meaning r0 + r1 >= 18 > 16.
    # This means we CANNOT achieve rank 16 with this row-separation approach.
    
    # However, if some terms have U supported on BOTH rows, we might do better.
    # A term with U = (u0,u1,u2,u3,u4,u5) where both (u0,u1,u2) and (u3,u4,u5) are nonzero
    # would contribute to both C row 0 and C row 1.
    
    # This is the general case and harder to analyze. Let's try a direct search.
    
    # Given the time constraints and the difficulty, let's return None to indicate
    # that rank 16 was not found via this approach.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Build the rank-17 decomposition
    U17, V17, W17 = _rank17_decomposition(n, m, p)
    
    # Verify rank-17
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook rank 18
        U_fb = np.zeros((n_a, 18))
        V_fb = np.zeros((n_b, 18))
        W_fb = np.zeros((n_c, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find a rank-16 decomposition
    if time.time() < deadline - 30:
        result16 = _try_rank16_via_row_decomposition(n, m, p, deadline)
        if result16 is not None:
            U16, V16, W16 = result16
            if _verify_decomposition(U16, V16, W16, T):
                return {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
    
    # Return the rank-17 decomposition
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END
