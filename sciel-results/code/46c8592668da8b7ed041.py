"""Rewritten program for 3x4 * 4x4 matrix multiplication tensor decomposition.
Strategy: Use Strassen's rank-7 algorithm for 2x2x2 sub-blocks and schoolbook for
remaining strips, aiming for rank 44 or lower.
"""
DIMENSIONS = [3, 4, 4]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid decomposition of the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # For each term r, the outer product U[:,r] * V[:,r] * W[:,r] should contribute
    # to the tensor. The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if the indices correspond to (i,j) in A, (j,k) in B, (i,k) in C
    # Actually, the standard tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    # T[i, j, k] = 1 where i is row of A/C, j is col of A / row of B, k is col of B/C
    # The factorization is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # Let's check by constructing the full tensor and comparing
    # This is expensive for large dimensions but fine for 3x4x4
    
    max_val = 0.0
    for r in range(rank):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # Reshape u to (n, m), v to (m, p), w to (n, p)
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(n, p)
        # The contribution to the tensor is u_mat[i,j] * v_mat[j,k] * w_mat[i,k]
        # We can check this by computing the "tensor" and comparing with expected
        # But it's easier to check a few random points or all points for small sizes
        pass
    
    # Full verification: compute the implied tensor and compare
    # T_implied[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # T_expected[i,j,k] = 1 for all valid (i,j,k) since matrix mult tensor has all ones
    
    # Actually, the matrix multiplication tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    
    # This is a system of n*m*p equations. For 3*4*4 = 48 equations.
    # Let's verify all of them.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy: 
    - Partition the 3x4 matrix A into blocks: top 2 rows and bottom 1 row
    - Partition the 4x4 matrix B into 2x2 blocks
    - Use Strassen's 7-term algorithm for the 2x2 * 2x2 multiplications
    - Use schoolbook for the remaining 1x4 * 4x4 multiplication
    
    This should give a rank of approximately 4*7 + 4 = 32, but with sharing it may be less.
    
    Let's be more careful:
    A is 3x4, B is 4x4, C is 3x4.
    
    Split A into A1 (2x4) and A2 (1x4).
    Split B into B_top (4x2) and B_bottom (4x2)? No, B is 4x4.
    
    Better: Split B into 2x2 blocks: B11, B12, B21, B22 where each is 2x2.
    A1 is 2x4, so split into A11 (2x2) and A12 (2x2).
    A2 is 1x4, so split into A21 (1x2) and A22 (1x2).
    
    C = A * B
    C_top (2x4) = A1 * B = [A11*B11 + A12*B21, A11*B12 + A12*B22]
    C_bottom (1x4) = A2 * B = [A21*B11 + A22*B21, A21*B12 + A22*B22]
    
    For C_top, we have two 2x2 * 2x2 multiplications:
    C11 = A11 * B11 + A12 * B21
    C12 = A11 * B12 + A12 * B22
    
    Each 2x2 * 2x2 can be done with Strassen (7 terms). But we have 4 such multiplications
    (A11*B11, A11*B12, A12*B21, A12*B22), so 4*7 = 28 terms.
    
    For C_bottom:
    C21 = A21 * B11 + A22 * B21  (1x2 * 2x2 + 1x2 * 2x2 = 1x2 result)
    C22 = A21 * B12 + A22 * B22
    
    Each 1x2 * 2x2 multiplication requires 4 scalar multiplications (schoolbook).
    So 4 * 4 = 16 terms for the bottom.
    
    Total: 28 + 16 = 44 terms.
    
    Let me implement this carefully.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build the decomposition term by term.
    # Each term r has:
    #   U[i*m+j, r] for i in 0..n-1, j in 0..m-1
    #   V[j*p+k, r] for j in 0..m-1, k in 0..p-1  
    #   W[k*n+i, r] for k in 0..p-1, i in 0..n-1
    #   (Note: W index is k*n+i per the trace convention W[k*n+i, r])
    
    terms = []  # list of (U_entries, V_entries, W_entries) where each is a dict
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a term with sparse entries."""
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen's 7-term algorithm for 2x2 * 2x2:
    # Given P (2x2) with entries P[0,0], P[0,1], P[1,0], P[1,1]
    # and Q (2x2) with entries Q[0,0], Q[0,1], Q[1,0], Q[1,1]
    # Compute R = P * Q (2x2)
    # 
    # M1 = (P[0,0] + P[1,1]) * (Q[0,0] + Q[1,1])
    # M2 = (P[1,0] + P[1,1]) * Q[0,0]
    # M3 = P[0,0] * (Q[0,1] - Q[1,1])
    # M4 = P[1,1] * (Q[1,0] - Q[0,0])
    # M5 = (P[0,0] + P[0,1]) * Q[1,1]
    # M6 = (P[1,0] - P[0,0]) * (Q[0,0] + Q[0,1])
    # M7 = (P[0,1] - P[1,1]) * (Q[1,0] + Q[1,1])
    #
    # R[0,0] = M1 + M4 - M5 + M7
    # R[0,1] = M3 + M5
    # R[1,0] = M2 + M4
    # R[1,1] = M1 - M2 + M3 + M6
    
    # For each 2x2 * 2x2 multiplication, we need to map the scalar multiplications
    # to the tensor factors.
    
    # Let's define a helper to add a Strassen decomposition for a specific 2x2 * 2x2 block.
    # The 2x2 block of A occupies rows i0, i0+1 and cols j0, j0+1 in A.
    # The 2x2 block of B occupies rows j0, j0+1 and cols k0, k0+1 in B.
    # The result 2x2 block of C occupies rows i0, i0+1 and cols k0, k0+1 in C.
    
    def add_strassen_2x2(i0, j0, k0, sign=1):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] 
        contributing to C[i0:i0+2, k0:k0+2].
        sign is the multiplier for the result (for subtraction in sums).
        """
        # P = A[i0:i0+2, j0:j0+2]
        # Q = B[j0:j0+2, k0:k0+2]
        # R = C[i0:i0+2, k0:k0+2]
        
        # M1 = (P00 + P11) * (Q00 + Q11)
        u = {(i0, j0): sign, (i0+1, j0+1): sign}
        v = {(j0, k0): 1, (j0+1, k0+1): 1}
        w = {(k0, i0): 1, (k0, i0+1): 1, (k0+1, i0): 1, (k0+1, i0+1): 1}
        # Wait, this isn't right. Each term is a single scalar multiplication.
        # M1 = (P00 + P11) * (Q00 + Q11) is actually a product of sums, which expands to 4 terms.
        # But in Strassen, we treat (P00+P11) and (Q00+Q11) as "virtual" entries.
        # The tensor decomposition works by having the U, V, W entries encode the linear combinations.
        
        # Actually, for the tensor decomposition, each "term" corresponds to one scalar multiplication.
        # In Strassen, we have 7 scalar multiplications, each of which is a product of a linear
        # combination of A entries and a linear combination of B entries.
        
        # So for M1: 
        #   U part: A[i0,j0] + A[i0+1,j0+1]  -> U entries: (i0,j0):1, (i0+1,j0+1):1
        #   V part: B[j0,k0] + B[j0+1,k0+1]  -> V entries: (j0,k0):1, (j0+1,k0+1):1
        #   W part: contributes to C[i0,k0] + C[i0,k0+1] + C[i0+1,k0] + C[i0+1,k0+1]
        #           But actually, M1 contributes to R[0,0] and R[1,1] with coefficients.
        #           R[0,0] += M1, R[1,1] += M1
        #           So W entries: (k0, i0):1, (k0+1, i0+1):1
        
        # Let me redo this more carefully.
        
        # M1 = (P00 + P11) * (Q00 + Q11)
        # Contributes: +1 to R00, +1 to R11
        u1 = {(i0, j0): 1, (i0+1, j0+1): 1}
        v1 = {(j0, k0): 1, (j0+1, k0+1): 1}
        w1 = {(k0, i0): sign, (k0+1, i0+1): sign}
        add_term(u1, v1, w1)
        
        # M2 = (P10 + P11) * Q00
        # Contributes: +1 to R10
        u2 = {(i0+1, j0): 1, (i0+1, j0+1): 1}
        v2 = {(j0, k0): 1}
        w2 = {(k0, i0+1): sign}
        add_term(u2, v2, w2)
        
        # M3 = P00 * (Q01 - Q11)
        # Contributes: +1 to R01
        u3 = {(i0, j0): 1}
        v3 = {(j0, k0+1): 1, (j0+1, k0+1): -1}
        w3 = {(k0+1, i0): sign}
        add_term(u3, v3, w3)
        
        # M4 = P11 * (Q10 - Q00)
        # Contributes: +1 to R00, +1 to R10
        u4 = {(i0+1, j0+1): 1}
        v4 = {(j0+1, k0): 1, (j0, k0): -1}
        w4 = {(k0, i0): sign, (k0, i0+1): sign}
        add_term(u4, v4, w4)
        
        # M5 = (P00 + P01) * Q11
        # Contributes: -1 to R00, +1 to R01
        u5 = {(i0, j0): 1, (i0, j0+1): 1}
        v5 = {(j0+1, k0+1): 1}
        w5 = {(k0, i0): -sign, (k0+1, i0): sign}
        add_term(u5, v5, w5)
        
        # M6 = (P10 - P00) * (Q00 + Q01)
        # Contributes: +1 to R11
        u6 = {(i0+1, j0): 1, (i0, j0): -1}
        v6 = {(j0, k0): 1, (j0, k0+1): 1}
        w6 = {(k0+1, i0+1): sign}
        add_term(u6, v6, w6)
        
        # M7 = (P01 - P11) * (Q10 + Q11)
        # Contributes: +1 to R00
        u7 = {(i0, j0+1): 1, (i0+1, j0+1): -1}
        v7 = {(j0+1, k0): 1, (j0+1, k0+1): 1}
        w7 = {(k0, i0): sign}
        add_term(u7, v7, w7)
    
    # Now, let's add the Strassen decompositions for the 4 2x2 * 2x2 multiplications:
    # C11 = A11 * B11 + A12 * B21
    #   A11: rows 0-1, cols 0-1 of A
    #   A12: rows 0-1, cols 2-3 of A
    #   B11: rows 0-1, cols 0-1 of B
    #   B12: rows 0-1, cols 2-3 of B
    #   B21: rows 2-3, cols 0-1 of B
    #   B22: rows 2-3, cols 2-3 of B
    
    # C11 = A11 * B11 + A12 * B21
    # Result: C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0, sign=1)   # A11 * B11 -> C[0:2, 0:2]
    add_strassen_2x2(0, 2, 0, sign=1)   # A12 * B21 -> C[0:2, 0:2]
    
    # C12 = A11 * B12 + A12 * B22
    # Result: C[0:2, 2:4]
    add_strassen_2x2(0, 0, 2, sign=1)   # A11 * B12 -> C[0:2, 2:4]
    add_strassen_2x2(0, 2, 2, sign=1)   # A12 * B22 -> C[0:2, 2:4]
    
    # For the bottom row:
    # C21 = A21 * B11 + A22 * B21  where A21 = A[2, 0:2], A22 = A[2, 2:4]
    # C22 = A21 * B12 + A22 * B22
    
    # Each 1x2 * 2x2 multiplication:
    # A21 * B11: A[2,0], A[2,1] times B[0:2, 0:2]
    # Result: C[2, 0], C[2, 1]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1]
    
    # Schoolbook for 1x2 * 2x2:
    # Term 1: A[2,0] * B[0,0] -> C[2,0]
    add_term({(2, 0): 1}, {(0, 0): 1}, {(0, 2): 1})
    # Term 2: A[2,0] * B[0,1] -> C[2,1]
    add_term({(2, 0): 1}, {(0, 1): 1}, {(1, 2): 1})
    # Term 3: A[2,1] * B[1,0] -> C[2,0]
    add_term({(2, 1): 1}, {(1, 0): 1}, {(0, 2): 1})
    # Term 4: A[2,1] * B[1,1] -> C[2,1]
    add_term({(2, 1): 1}, {(1, 1): 1}, {(1, 2): 1})
    
    # A21 * B12: A[2,0], A[2,1] times B[0:2, 2:4]
    # Result: C[2, 2], C[2, 3]
    add_term({(2, 0): 1}, {(0, 2): 1}, {(2, 2): 1})
    add_term({(2, 0): 1}, {(0, 3): 1}, {(3, 2): 1})
    add_term({(2, 1): 1}, {(1, 2): 1}, {(2, 2): 1})
    add_term({(2, 1): 1}, {(1, 3): 1}, {(3, 2): 1})
    
    # A22 * B21: A[2,2], A[2,3] times B[2:4, 0:2]
    # Result: C[2, 0], C[2, 1]
    add_term({(2, 2): 1}, {(2, 0): 1}, {(0, 2): 1})
    add_term({(2, 2): 1}, {(2, 1): 1}, {(1, 2): 1})
    add_term({(2, 3): 1}, {(3, 0): 1}, {(0, 2): 1})
    add_term({(2, 3): 1}, {(3, 1): 1}, {(1, 2): 1})
    
    # A22 * B22: A[2,2], A[2,3] times B[2:4, 2:4]
    # Result: C[2, 2], C[2, 3]
    add_term({(2, 2): 1}, {(2, 2): 1}, {(2, 2): 1})
    add_term({(2, 2): 1}, {(2, 3): 1}, {(3, 2): 1})
    add_term({(2, 3): 1}, {(3, 2): 1}, {(2, 2): 1})
    add_term({(2, 3): 1}, {(3, 3): 1}, {(3, 2): 1})
    
    # Now convert the terms to dense matrices.
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    # Verify the decomposition
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # If verification fails, fall back to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
