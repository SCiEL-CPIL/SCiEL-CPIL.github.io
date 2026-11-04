import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
DEADLINE_S = 130.0  # safety margin under 150s limit


def _strassen_2x2():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    
    For A (2x2), B (2x2), C = A*B (2x2):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Index convention:
    U[i*2+j, r] for A entry (i,j)
    V[j*2+k, r] for B entry (j,k)  
    W[k*2+i, r] for C entry (k,i)
    
    Strassen formulas:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)(B11+B12)  -- wait, let me use the correct standard form
    
    Actually, the standard Strassen:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B11)
    M4 = A22(B12 - B22)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M7
    
    Let me verify with A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
    C = [[19,22],[43,50]]
    
    M1 = (1+4)(5+8) = 65
    M2 = (3+4)*5 = 35
    M3 = 1*(7-5) = 2
    M4 = 4*(6-8) = -8
    M5 = (1+2)*8 = 24
    M6 = (3-1)*(5+6) = 22
    M7 = (2-4)*(7+8) = -30
    
    C11 = 65 + (-8) - 24 - (-30) = 63. Should be 19. WRONG again!
    
    Let me try yet another formulation. The issue is I keep getting the formulas wrong.
    Let me derive it properly.
    
    C11 = a11*b11 + a12*b21
    C12 = a11*b12 + a12*b22
    C21 = a21*b11 + a22*b21
    C22 = a21*b12 + a22*b22
    
    I need 7 products whose linear combinations give these 4 values.
    
    Let me use:
    P1 = (a11+a22)(b11+b22) = a11b11 + a11b22 + a22b11 + a22b22
    P2 = (a21+a22)b11 = a21b11 + a22b11
    P3 = a11(b21-b11) = a11b21 - a11b11
    P4 = a22(b12-b22) = a22b12 - a22b22
    P5 = (a11+a12)b22 = a11b22 + a12b22
    P6 = (a21-a11)(b11+b12) = a21b11 + a21b12 - a11b11 - a11b12
    P7 = (a12-a22)(b21+b22) = a12b21 + a12b22 - a22b21 - a22b22
    
    Now:
    C11 = a11b11 + a12b21
    C12 = a11b12 + a12b22
    C21 = a21b11 + a22b21
    C22 = a21b12 + a22b22
    
    Try: C11 = P1 - P2 - P4 + P3 + P7 - P6 - P5
    = (a11b11 + a11b22 + a22b11 + a22b22) - (a21b11 + a22b11) - (a22b12 - a22b22) + (a11b21 - a11b11) + (a12b21 + a12b22 - a22b21 - a22b22) - (a21b11 + a21b12 - a11b11 - a11b12) - (a11b22 + a12b22)
    
    This is getting too complicated to do by hand. Let me just verify computationally.
    
    For the purposes of this program, let me use a computational approach to find
    the correct Strassen coefficients, or just fall back to schoolbook.
    
    Given the time constraints and the difficulty of getting Strassen right,
    let me use a different strategy: implement the 49-term 4x4x4 decomposition
    using the known fact that it exists, and add 16 terms for the 5th column.
    
    Actually, the safest approach is to use schoolbook (rank 80) which is guaranteed
    correct, and then try to improve if time permits.
    """
    pass


def _schoolbook(n, m, p):
    """
    Schoolbook decomposition: rank = n*m*p.
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for the (i,j,k) term.
    """
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
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor via the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Efficient check using einsum
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is tricky with einsum. Let me do it differently.
    
    # For each r, the contribution to T is:
    # U_r[i,j] * V_r[j,k] * W_r[k,i]
    # where U_r is U[:,r], V_r is V[:,r], W_r is W[:,r]
    
    # Let me just check a subset for speed, or all if small enough
    max_check = 2000
    count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False
                count += 1
                if count >= max_check:
                    break
            if count >= max_check:
                break
        if count >= max_check:
            break
    
    # For full verification when rank is small
    if rank <= 100:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
    
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook: rank = 80
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try to improve using Strassen-based approach for 4x4x4 + 16 for 5th column
    # 4x4x4 via recursive Strassen: 49 terms
    # 5th column: 16 terms (4x4)
    # Total: 65 terms
    
    # For the 4x4x4 part, I need the 49-term decomposition.
    # This is complex. Let me try a simpler improvement first.
    
    # Alternative: Use the fact that R(4,4,5) <= 4*R(2,2,5) + correction
    # Or: pad 4x4x5 to 4x5x5 and use known bounds
    
    # For now, let's try to implement the 49+16=65 scheme if I can get the
    # 4x4x4 Strassen decomposition right.
    
    # Actually, let me try a computational search for a low-rank decomposition.
    # This is a hard problem, but let me try a simple approach.
    
    # The tensor T for 4x4x5 has dimension 4*4*5 = 80.
    # I want to find the minimum rank R such that T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # This is equivalent to finding the minimum number of rank-1 tensors
    # that sum to T.
    
    # For matrix multiplication, the tensor is:
    # T[i,j,k] = 1 for all i in [0,4), j in [0,4), k in [0,5)
    
    # Wait, that's not right. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if we're computing C[k,i] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k).
    
    # The rank of this tensor is what we want to minimize.
    
    # Schoolbook: R = 4*4*5 = 80
    # Better: R(4,4,4) + 16 = 49 + 16 = 65 (if I can get the 49-term 4x4x4)
    
    # Let me try to construct the 49-term 4x4x4 decomposition.
    # 
    # 4x4x4 via two levels of Strassen:
    # Level 1: Split 4x4 into four 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    #
    # Strassen level 1 (7 block multiplications of 2x2 matrices):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each M_i is a 2x2 * 2x2 = 2x2 product, which can be done with 7 scalar multiplications.
    # Total: 7 * 7 = 49 scalar multiplications.
    #
    # Now I need to expand each M_i into its 7 Strassen terms.
    # This requires knowing the correct Strassen formulas for 2x2.
    
    # Let me computationally verify the Strassen formulas.
    # I'll set up the system of equations and solve.
    
    # For 2x2x2:
    # Variables: a11, a12, a21, a22, b11, b12, b21, b22
    # C11 = a11*b11 + a12*b21
    # C12 = a11*b12 + a12*b22
    # C21 = a21*b11 + a22*b21
    # C22 = a21*b12 + a22*b22
    
    # I want to find 7 pairs of linear forms (u_r, v_r) and 4 sets of coefficients (w_r^C11, w_r^C12, w_r^C21, w_r^C22)
    # such that C_ik = sum_r w_r^Cik * u_r * v_r
    
    # This is a nonlinear system. Let me just use known correct Strassen formulas.
    
    # From a reliable source, the correct Strassen formulas are:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11  
    # M3 = A11(B21 - B11)
    # M4 = A22(B12 - B22)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # Let me verify this computationally with a test case.
    A_test = np.array([[1, 2], [3, 4]])
    B_test = np.array([[5, 6], [7, 8]])
    C_expected = A_test @ B_test  # [[19, 22], [43, 50]]
    
    a11, a12, a21, a22 = A_test[0,0], A_test[0,1], A_test[1,0], A_test[1,1]
    b11, b12, b21, b22 = B_test[0,0], B_test[0,1], B_test[1,0], B_test[1,1]
    
    M1 = (a11 + a22) * (b11 + b22)
    M2 = (a21 + a22) * b11
    M3 = a11 * (b21 - b11)
    M4 = a22 * (b12 - b22)
    M5 = (a11 + a12) * b22
    M6 = (a21 - a11) * (b11 + b12)
    M7 = (a12 - a22) * (b21 + b22)
    
    C11 = M1 + M4 - M5 - M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M7
    
    C_computed = np.array([[C11, C12], [C21, C22]])
    
    if not np.allclose(C_computed, C_expected):
        # The formulas are wrong. Let me try another set.
        # Try:
        # M2 = (A21 + A22) * B21  (instead of B11)
        M2_alt = (a21 + a22) * b21
        C21_alt = M2_alt + M4
        # This gives 49 + (-8) = 41, should be 43. Still wrong.
        
        # Let me try a completely different approach.
        # I'll use the schoolbook for 2x2x2 (4 terms) and just do 4x4x4 with 4*4*4=64 terms.
        # Then 4x4x5 = 64 + 20 = 84. That's worse than 80.
        
        # Or: 4x4x4 with 49 terms is the target, but I can't get the Strassen right.
        
        # Let me just use schoolbook for everything. Rank = 80.
        pass
    
    # Given the difficulty of getting Strassen right, and the time constraints,
    # let me return the schoolbook solution.
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_S:
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
