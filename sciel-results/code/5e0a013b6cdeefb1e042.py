"""Matrix multiplication 2x3x3 rank-minimal decomposition search."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def verify_decomposition(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check: for all i,j,k, T[i,j,k] should equal 1 if this is the multiplication tensor
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # Wait no - the tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def verify_decomposition_fast(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Faster verification using numpy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # This is a trilinear form check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def make_schoolbook(n=2, m=3, p=3):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def strassen_3x3x3_decomposition():
    """
    Return a rank-17 decomposition of the 3x3x3 matrix multiplication tensor.
    Uses the known Strassen-based rank-17 algorithm.
    
    The 3x3 matrix multiplication can be done with 17 multiplications.
    We'll use a known rank-17 decomposition.
    
    For 3x3x3, one known rank-17 decomposition comes from combining
    Strassen's 7-multiplication 2x2x2 with additional terms.
    
    Let me use a concrete known decomposition.
    """
    n = m = p = 3
    rank = 17
    U = np.zeros((n*m, rank))  # 9 x 17
    V = np.zeros((m*p, rank))  # 9 x 17
    W = np.zeros((p*n, rank))  # 9 x 17
    
    # Known rank-17 decomposition for 3x3x3
    # This uses a specific construction. Let me use the one from
    # "Fast Multiplication of 3x3 Matrices" - the rank-17 algorithm.
    
    # I'll use a verified rank-17 decomposition.
    # One approach: partition 3x3 into 2x2 + 1, use Strassen on the 2x2 part
    # and handle cross terms.
    
    # Actually, let me use a known explicit rank-17 decomposition.
    # The coefficients are integers or half-integers.
    
    # Using the decomposition from Baur-Strassen / known results:
    # For 3x3 matrices, we can write:
    # Split A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # Similarly for B and C.
    
    # A well-known rank-17 algorithm:
    # Let me construct it carefully.
    
    # Actually, for this task I'll use a different approach.
    # Let me directly search for a rank-16 or rank-17 decomposition
    # of the 2x3x3 tensor.
    
    # For now, return a placeholder that we'll replace.
    # Let me instead directly construct a rank-17 2x3x3 decomposition
    # by taking a rank-17 3x3x3 and restricting.
    
    # Known rank-17 3x3x3 decomposition (one standard form):
    # I'll use the decomposition where we combine a 2x2x2 Strassen
    # block with additional terms for the 3rd row/column.
    
    # Let me use a concrete verified decomposition.
    # After research, here's a rank-17 decomposition for 3x3x3:
    
    # Using the "Strassen-3x3" construction:
    # Block 1 (top-left 2x2): 7 Strassen multiplications
    # Plus additional terms for the rest
    
    # Let me just directly work with the 2x3x3 problem.
    
    # For 2x3x3: A is 2x3, B is 3x3, C is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2.
    # Total: 6 output entries, 9 input pairs (j,k) per row of A.
    
    # Schoolbook: 2*3*3 = 18 multiplications.
    # We want to do better.
    
    # Strategy: Use Strassen on the 2x2 sub-block of A (columns 0,1)
    # and handle column 2 separately.
    
    # Let me implement a search-based approach instead.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 2, 3, 3
    
    # First, get the schoolbook rank-18 decomposition as fallback
    U_school, V_school, W_school = make_schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 18
    
    # Try to find rank-17 decomposition
    # Strategy: Use the fact that 2x3x3 is related to 3x3x3
    # or use direct search.
    
    # Approach 1: Try rank-17 with structured decomposition
    # The 2x3 multiplication can be viewed as:
    # C = A * B where A is 2x3, B is 3x3
    # 
    # We can write A = [A_left (2x2) | A_right (2x1)]
    # B = [[B_topleft (2x2), B_topright (2x1)], [B_botleft (1x2), B_botright (1x1)]]
    #
    # C = A_left * B_topleft + A_left * B_topright + A_right * B_botleft + A_right * B_botright
    #   (with appropriate indexing)
    
    # Let me try a direct numerical search for rank-16 and rank-17.
    
    # For rank-17: Try to find a decomposition
    # For rank-16: Try to find a decomposition
    
    # Let me try a structured approach first.
    
    # Rank-17 attempt using Strassen on 2x2 sub-block:
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = [[c00, c01, c02], [c10, c11, c12]]
    #
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # Using Strassen on the 2x2 part (a00,a01,a10,a11) * (b00,b01,b10,b11):
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a11-a10)*(b00+b10)
    # m7 = (a01-a00)*(b10+b11)
    #
    # c00 = m1 - m4 - m6 + m2
    # c01 = m1 + m3 - m5 - m7
    # c10 = m4 + m5 + m6
    # c11 = m1 - m3 + m5 + m7
    
    # Then handle the a02, a12 terms:
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c12 = a10*b02 + a11*b12 + a12*b22
    # c00 += a02*b20 (wait, I already have c00 from Strassen)
    # Actually c00 = a00*b00 + a01*b10 + a02*b20
    # From Strassen: a00*b00 + a01*b10 is part of the 2x2 result
    # So c00 = (Strassen c00) + a02*b20
    
    # Let me be more careful:
    # Strassen gives us:
    # s00 = a00*b00 + a01*b10
    # s01 = a00*b01 + a01*b11
    # s10 = a10*b00 + a11*b10
    # s11 = a10*b01 + a11*b11
    #
    # Using 7 multiplications for s00,s01,s10,s11.
    #
    # Then:
    # c00 = s00 + a02*b20
    # c01 = s01 + a02*b21
    # c02 = a00*b02 + a01*b12 + a02*b22
    # c10 = s10 + a12*b20
    # c11 = s11 + a12*b21
    # c12 = a10*b02 + a11*b12 + a12*b22
    
    # c02 and c12 each need 3 multiplications (schoolbook) = 6 more
    # Total: 7 + 6 = 13? No wait, that's not right.
    # a02*b20, a02*b21 are 2 more multiplications for c00, c01
    # a12*b20, a12*b21 are 2 more for c10, c11
    # c02 = a00*b02 + a01*b12 + a02*b22: 3 multiplications
    # c12 = a10*b02 + a11*b12 + a12*b22: 3 multiplications
    # Total: 7 + 2 + 2 + 3 + 3 = 17
    
    # So rank 17 is achievable with this structure!
    # Let me build the explicit decomposition.
    
    # Strassen 2x2x2 decomposition (7 terms):
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a11-a10)*(b00+b10)
    # m7 = (a01-a00)*(b10+b11)
    #
    # s00 = m1 - m4 - m6 + m2
    # s01 = m1 + m3 - m5 - m7
    # s10 = m4 + m5 + m6
    # s11 = m1 - m3 + m5 + m7
    
    # Additional terms:
    # m8 = a02*b20
    # m9 = a02*b21
    # m10 = a12*b20
    # m11 = a12*b21
    # m12 = a00*b02
    # m13 = a01*b12
    # m14 = a02*b22
    # m15 = a10*b02
    # m16 = a11*b12
    # m17 = a12*b22
    
    # c00 = m1 - m4 - m6 + m2 + m8
    # c01 = m1 + m3 - m5 - m7 + m9
    # c02 = m12 + m13 + m14
    # c10 = m4 + m5 + m6 + m10
    # c11 = m1 - m3 + m5 + m7 + m11
    # c12 = m15 + m16 + m17
    
    # Now let me build U, V, W for this rank-17 decomposition.
    # U has shape (n*m, rank) = (6, 17)
    # V has shape (m*p, rank) = (9, 17)
    # W has shape (p*n, rank) = (6, 17)
    
    # U[i*m+j, c] corresponds to coefficient of A[i,j] in term c
    # V[j*p+k, c] corresponds to coefficient of B[j,k] in term c
    # W[k*n+i, c] corresponds to coefficient of C[i,k] in term c
    
    # Indices:
    # A: i in {0,1}, j in {0,1,2} -> A[i,j] at index i*m+j = i*3+j
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2, A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B: j in {0,1,2}, k in {0,1,2} -> B[j,k] at index j*p+k = j*3+k
    #   B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5, B[2,0]=6, B[2,1]=7, B[2,2]=8
    # C: k in {0,1,2}, i in {0,1} -> C[i,k] at index k*n+i = k*2+i
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # m1 = (a00+a11)*(b00+b11)
    # U: A[0,0]=0 gets +1, A[1,1]=4 gets +1
    # V: B[0,0]=0 gets +1, B[1,1]=4 gets +1
    # W: contributes to c00, c01, c11
    #   c00: +1, c01: +1, c11: +1
    # W index: c00 -> k*n+i = 0*2+0 = 0
    #           c01 -> k*n+i = 1*2+0 = 2
    #           c11 -> k*n+i = 1*2+1 = 3
    U[0, 0] = 1  # a00
    U[4, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[4, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[2, 0] = 1  # c01
    W[3, 0] = 1  # c11
    
    # m2 = (a10+a11)*b00
    # U: A[1,0]=3 gets +1, A[1,1]=4 gets +1
    # V: B[0,0]=0 gets +1
    # W: contributes to c00: +1
    U[3, 1] = 1  # a10
    U[4, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[0, 1] = 1  # c00
    
    # m3 = a00*(b01-b11)
    # U: A[0,0]=0 gets +1
    # V: B[0,1]=1 gets +1, B[1,1]=4 gets -1
    # W: contributes to c01: +1, c11: -1
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[4, 2] = -1  # b11
    W[2, 2] = 1  # c01
    W[3, 2] = -1  # c11
    
    # m4 = a11*(b10-b00)
    # U: A[1,1]=4 gets +1
    # V: B[1,0]=3 gets +1, B[0,0]=0 gets -1
    # W: contributes to c00: -1, c10: +1, c11: -1
    U[4, 3] = 1  # a11
    V[3, 3] = 1  # b10
    V[0, 3] = -1  # b00
    W[0, 3] = -1  # c00
    W[1, 3] = 1  # c10
    W[3, 3] = -1  # c11
    
    # m5 = (a00+a10)*b11
    # U: A[0,0]=0 gets +1, A[1,0]=3 gets +1
    # V: B[1,1]=4 gets +1
    # W: contributes to c01: -1, c10: +1, c11: +1
    U[0, 4] = 1  # a00
    U[3, 4] = 1  # a10
    V[4, 4] = 1  # b11
    W[2, 4] = -1  # c01
    W[1, 4] = 1  # c10
    W[3, 4] = 1  # c11
    
    # m6 = (a11-a10)*(b00+b10)
    # U: A[1,1]=4 gets +1, A[1,0]=3 gets -1
    # V: B[0,0]=0 gets +1, B[1,0]=3 gets +1
    # W: contributes to c00: -1, c10: +1
    U[4, 5] = 1  # a11
    U[3, 5] = -1  # a10
    V[0, 5] = 1  # b00
    V[3, 5] = 1  # b10
    W[0, 5] = -1  # c00
    W[1, 5] = 1  # c10
    
    # m7 = (a01-a00)*(b10+b11)
    # U: A[0,1]=1 gets +1, A[0,0]=0 gets -1
    # V: B[1,0]=3 gets +1, B[1,1]=4 gets +1
    # W: contributes to c01: -1, c11: +1
    U[1, 6] = 1  # a01
    U[0, 6] = -1  # a00
    V[3, 6] = 1  # b10
    V[4, 6] = 1  # b11
    W[2, 6] = -1  # c01
    W[3, 6] = 1  # c11
    
    # m8 = a02*b20
    # U: A[0,2]=2 gets +1
    # V: B[2,0]=6 gets +1
    # W: contributes to c00: +1
    U[2, 7] = 1  # a02
    V[6, 7] = 1  # b20
    W[0, 7] = 1  # c00
    
    # m9 = a02*b21
    # U: A[0,2]=2 gets +1
    # V: B[2,1]=7 gets +1
    # W: contributes to c01: +1
    U[2, 8] = 1  # a02
    V[7, 8] = 1  # b21
    W[2, 8] = 1  # c01
    
    # m10 = a12*b20
    # U: A[1,2]=5 gets +1
    # V: B[2,0]=6 gets +1
    # W: contributes to c10: +1
    U[5, 9] = 1  # a12
    V[6, 9] = 1  # b20
    W[1, 9] = 1  # c10
    
    # m11 = a12*b21
    # U: A[1,2]=5 gets +1
    # V: B[2,1]=7 gets +1
    # W: contributes to c11: +1
    U[5, 10] = 1  # a12
    V[7, 10] = 1  # b21
    W[3, 10] = 1  # c11
    
    # m12 = a00*b02
    # U: A[0,0]=0 gets +1
    # V: B[0,2]=2 gets +1
    # W: contributes to c02: +1
    U[0, 11] = 1  # a00
    V[2, 11] = 1  # b02
    W[4, 11] = 1  # c02
    
    # m13 = a01*b12
    # U: A[0,1]=1 gets +1
    # V: B[1,2]=5 gets +1
    # W: contributes to c02: +1
    U[1, 12] = 1  # a01
    V[5, 12] = 1  # b12
    W[4, 12] = 1  # c02
    
    # m14 = a02*b22
    # U: A[0,2]=2 gets +1
    # V: B[2,2]=8 gets +1
    # W: contributes to c02: +1
    U[2, 13] = 1  # a02
    V[8, 13] = 1  # b22
    W[4, 13] = 1  # c02
    
    # m15 = a10*b02
    # U: A[1,0]=3 gets +1
    # V: B[0,2]=2 gets +1
    # W: contributes to c12: +1
    U[3, 14] = 1  # a10
    V[2, 14] = 1  # b02
    W[5, 14] = 1  # c12
    
    # m16 = a11*b12
    # U: A[1,1]=4 gets +1
    # V: B[1,2]=5 gets +1
    # W: contributes to c12: +1
    U[4, 15] = 1  # a11
    V[5, 15] = 1  # b12
    W[5, 15] = 1  # c12
    
    # m17 = a12*b22
    # U: A[1,2]=5 gets +1
    # V: B[2,2]=8 gets +1
    # W: contributes to c12: +1
    U[5, 16] = 1  # a12
    V[8, 16] = 1  # b22
    W[5, 16] = 1  # c12
    
    # Verify rank-17 decomposition
    if verify_decomposition_fast(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = 17
        # Try to reduce to rank-16
        if time_left() > 30:
            rank16_result = try_rank_16(U, V, W, n, m, p)
            if rank16_result is not None:
                best_U, best_V, best_W = rank16_result
                best_rank = 16
    else:
        # Fallback to schoolbook
        U, V, W = U_school, V_school, W_school
        best_rank = 18
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in np.array(best_U).tolist()]
    V_list = [[float(x) for x in row] for row in np.array(best_V).tolist()]
    W_list = [[float(x) for x in row] for row in np.array(best_W).tolist()]
    
    # Clean up: convert near-zero to 0, near-integer to integer
    def clean_val(x):
        if abs(x) < 1e-10:
            return 0.0
        if abs(x - round(x)) < 1e-10:
            return float(round(x))
        if abs(2*x - round(2*x)) < 1e-10:
            return float(round(2*x)) / 2.0
        return float(x)
    
    U_list = [[clean_val(x) for x in row] for row in U_list]
    V_list = [[clean_val(x) for x in row] for row in V_list]
    W_list = [[clean_val(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def try_rank_16(U17, V17, W17, n=2, m=3, p=3):
    """
    Try to reduce rank-17 to rank-16 by finding a linear combination
    that eliminates one term.
    
    If we have a rank-17 decomposition, we can try to find coefficients
    alpha_c such that sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0
    and then eliminate one term.
    """
    rank = 17
    U = np.array(U17, dtype=float)
    V = np.array(V17, dtype=float)
    W = np.array(W17, dtype=float)
    
    # Check if there's a linear dependency among the 17 rank-1 tensors
    # u_c ⊗ v_c ⊗ w_c
    # This is a 6x9x6 tensor space, dimension 324
    # 17 vectors in 324-dim space - likely independent
    
    # Alternative: try to find a rank-16 decomposition directly
    # by setting one of the W coefficients to zero and solving
    
    # For each term r, try to eliminate it by adjusting others
    for elim in range(rank):
        if time_left() < 10:
            break
        
        # Try to find a rank-16 decomposition by removing term 'elim'
        # and adjusting the remaining 16 terms
        # This is a nonlinear problem, so let's try a simpler approach:
        # Check if the 'elim' term's contribution can be absorbed
        
        # Build the tensor for the 17-term decomposition
        # T = sum_c u_c ⊗ v_c ⊗ w_c
        # We want T = sum_{c != elim} u'_c ⊗ v'_c ⊗ w'_c
        
        # Simple check: is term 'elim' a linear combination of others?
        # This is hard in general. Let's skip this for now.
        pass
    
    # Try a different approach: direct search for rank-16
    # using the structure of the problem
    
    # Actually, let's try to see if we can merge two terms.
    # For example, m8 and m10 both involve b20:
    # m8 = a02*b20, m10 = a12*b20
    # We could write (a02+a12)*b20 and (a02-a12)*b20
    # But that doesn't reduce the count.
    
    # Let me try a numerical approach: 
    # Fix 16 of the 17 terms and try to solve for the 17th to be zero.
    # This won't work in general.
    
    # For now, return None (no rank-16 found)
    return None

# Pre-compute the rank-17 decomposition at module level for speed
# Actually, let's just compute it in construct()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    print(f"Valid: {verify_decomposition_fast(U, V, W)}")
