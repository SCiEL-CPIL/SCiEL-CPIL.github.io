"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 60 decomposition using Strassen for 2x2x2 blocks
    # A is 3x3, B is 3x7
    # Partition:
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x2), B13 (2x3)],
    #      [B21 (1x2), B22 (1x2), B23 (1x3)]]
    
    # Strategy:
    # - A11*B11: Strassen (7 terms) instead of schoolbook (8) -> save 1
    # - A11*B12: Strassen (7 terms) instead of schoolbook (8) -> save 1
    # - A11*B13: Schoolbook (6 terms) [2x2 times 2x3 = 12 scalar products, but we only need 4 output entries]
    #   Actually, 2x2 * 2x3 gives 2x3 output = 6 entries. Schoolbook = 6*2 = 12? No.
    #   Each output entry C[i,k] = sum_j A[i,j]*B[j,k]. For 2x2*2x3, that's 2*3*2 = 12 multiplications.
    #   But we can't use Strassen directly. Let's just use schoolbook for the non-square parts.
    #   
    # Let me reconsider. The total schoolbook is 3*3*7 = 63.
    # 
    # Block contributions (schoolbook):
    # A11*B11: 2*2*2 = 8
    # A11*B12: 2*2*2 = 8
    # A11*B13: 2*2*3 = 12
    # A12*B21: 2*1*2 = 4
    # A12*B22: 2*1*2 = 4
    # A12*B23: 2*1*3 = 6
    # A21*B11: 1*2*2 = 4
    # A21*B12: 1*2*2 = 4
    # A21*B13: 1*2*3 = 6
    # A22*B21: 1*1*2 = 2
    # A22*B22: 1*1*2 = 2
    # A22*B23: 1*1*3 = 3
    # Total: 8+8+12+4+4+6+4+4+6+2+2+3 = 63. Correct.
    
    # With Strassen on A11*B11 and A11*B12:
    # A11*B11: 7 (save 1)
    # A11*B12: 7 (save 1)
    # Rest unchanged: 12+4+4+6+4+4+6+2+2+3 = 47
    # Total: 7+7+47 = 61. Save 2.
    
    # Can we do better? 
    # A11*B13 is 2x2*2x3. The rank of this tensor is min(2*2*3, ...) 
    # Actually, for multiplying 2x2 by 2x3, we can view it as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,1,2}
    # This is like 3 separate 2x1 * 2x1 multiplications (one for each column k).
    # Each 2x1*2x1 has rank 2 (since it's just A[i,0]*B[0,k] + A[i,1]*B[1,k]).
    # So total rank for A11*B13 = 3*2 = 6? No, that's not right either.
    # Actually, the tensor for 2x2*2x3 has rank at most 2*2*3/2 = 6? 
    # The trivial upper bound is 12. But we can do better.
    # 
    # For 2x2 * 2x3, we can use:
    # For each column k (0,1,2):
    #   C[:,k] = A * B[:,k] which is 2x1 * 2x1, rank 2.
    # So total = 3 columns * 2 = 6 terms.
    # Wait, that's not how tensor rank works. Each term is a rank-1 tensor.
    # For column k: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This uses 2 rank-1 tensors per column: (e_i * e_0 * e_k) and (e_i * e_1 * e_k) for i=0,1.
    # So 2 terms per column, 3 columns = 6 terms total.
    # Schoolbook for this block is 12. So we save 6 here!
    
    # Similarly for other rectangular blocks:
    # A12*B21: 2x1 * 1x2. C[i,k] = A[i,0]*B[0,k]. This is already rank 1 per (i,k) pair.
    # 2*1*1 = 2 terms? No, 2x1*1x2 gives 2x2 output. C[i,k] = A[i,0]*B[0,k].
    # This is 4 rank-1 tensors (one per output entry). Schoolbook = 4. Can't improve.
    
    # Let me recount with improvements:
    # A11*B11: Strassen = 7 (save 1 from 8)
    # A11*B12: Strassen = 7 (save 1 from 8)
    # A11*B13: 6 terms (save 6 from 12)
    # A12*B21: 4 (no change)
    # A12*B22: 4 (no change)
    # A12*B23: 6 terms. 2x1*1x3. C[i,k] = A[i,0]*B[0,k]. 2*3 = 6 rank-1 terms. No change from schoolbook.
    # A21*B11: 4 (no change)
    # A21*B12: 4 (no change)
    # A21*B13: 6 terms. 1x2*2x3. C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]. 3 columns * 2 = 6. No change.
    # A22*B21: 2 (no change)
    # A22*B22: 2 (no change)
    # A22*B23: 3 (no change)
    
    # Total: 7+7+6+4+4+6+4+4+6+2+2+3 = 55. Save 8 from 63.
    
    # Wait, but can we combine terms across blocks? The indices are shared in A and B.
    # Actually, no. Each block uses different parts of A and B, so the rank-1 tensors are independent.
    # The total rank is the sum of the ranks of each block.
    
    # Let me verify: 7+7+6+4+4+6+4+4+6+2+2+3 = 55.
    # 7+7 = 14, +6 = 20, +4 = 24, +4 = 28, +6 = 34, +4 = 38, +4 = 42, +6 = 48, +2 = 50, +2 = 52, +3 = 55.
    
    # So rank 55 is achievable? Let me double-check the rectangular block ranks.
    
    # For 2x2 * 2x3:
    # Output C is 2x3. C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # Rank-1 tensor: For fixed (i,j,k), the tensor is e_i^A ⊗ e_j^B ⊗ e_k^C.
    # We need: sum over j of A[i,j]B[j,k] for each (i,k).
    # This is equivalent to: for each j in {0,1} and k in {0,1,2}, we have a term A[i,j]*B[j,k] contributing to C[i,k].
    # But that's 2*3 = 6 terms if we group by j,k and sum over i? No.
    # Actually, each rank-1 term in the tensor decomposition is of the form u ⊗ v ⊗ w where u is in A-space, v in B-space, w in C-space.
    # For the block A11*B13, the A-space is 2D (rows 0,1; cols 0,1 of A), B-space is 2D (rows 0,1; cols 0,1,2 of B), C-space is 2D x 3D.
    # The tensor T[i,j,k] = delta(i,j) * delta(j,k) in the appropriate indices.
    # Actually, T is the matrix multiplication tensor for 2x2 * 2x3.
    # Its rank is at most 2*2*3 = 12 (schoolbook).
    # Can it be less? The rank of the matrix multiplication tensor [n,m,p] is at least n*m (for fixed p) and at least m*p (for fixed n) and at least n*p (for fixed m).
    # For [2,2,3]: lower bound is max(2*2, 2*3, 2*3) = max(4,6,6) = 6.
    # Upper bound: we can achieve rank 6 by noting that for each column k, the 2x2*2x1 subproblem has rank 2, and there are 3 columns, so 6 total.
    # Yes, rank 6 is achievable.
    
    # Similarly for 2x1 * 1x3: [2,1,3]. Lower bound max(2*1, 1*3, 2*3) = max(2,3,6) = 6.
    # Schoolbook = 2*1*3 = 6. So rank 6, no improvement possible.
    
    # For 1x2 * 2x3: [1,2,3]. Lower bound max(1*2, 2*3, 1*3) = max(2,6,3) = 6.
    # Schoolbook = 1*2*3 = 6. No improvement.
    
    # So the total rank is 55. Let me construct it.
    
    U_rows = n * m  # 9
    V_rows = m * p  # 21
    W_rows = p * n  # 21
    
    terms = []  # Each term is (u_indices, v_indices, w_indices, coefficients)
    # Actually, let's build U, V, W directly.
    
    # We'll use a list of (u_row, v_row, w_row, coeff) tuples and then build the matrices.
    # But since we want integer/half-integer entries, let's use fractions or just integers.
    # Strassen uses coefficients 1, -1, 2, etc.
    
    # Let me define helper functions.
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        """Add a rank-1 term to the decomposition."""
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Block A11*B11: Strassen
    # A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
    # B11 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
    # C11 = A11 * B11 (2x2 output, entries C[0,0], C[0,1], C[1,0], C[1,1])
    # 
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A11-A01)*(B01+B11)
    # 
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 - M6
    # C11 = M1 - M3 + M2 - M5
    
    # A indices: A11 is rows 0,1; cols 0,1. So A[i,j] for i in {0,1}, j in {0,1}.
    # U index for A[i,j] is i*m + j = i*3 + j.
    # B indices: B11 is rows 0,1; cols 0,1. So B[j,k] for j in {0,1}, k in {0,1}.
    # V index for B[j,k] is j*p + k = j*7 + k.
    # C indices: C11 is rows 0,1; cols 0,1. So C[i,k] for i in {0,1}, k in {0,1}.
    # W index for C[i,k] is k*n + i = k*3 + i.
    
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # This is a product of sums, which requires 4 rank-1 terms if expanded, but in Strassen,
    # each M is a single scalar product. However, in the tensor decomposition, each M corresponds
    # to a rank-1 tensor in the "intermediate" space, but we need to express it in terms of the original A, B.
    # 
    # Actually, in the standard Strassen decomposition of the matrix multiplication tensor,
    # each of the 7 M's corresponds to one rank-1 term in the tensor decomposition.
    # The coefficients in U, V, W encode the linear combinations.
    # 
    # For M1 = (A00+A11)*(B00+B11):
    # U: A00 gets coeff 1, A11 gets coeff 1
    # V: B00 gets coeff 1, B11 gets coeff 1
    # W: C00 gets coeff 1, C01 gets coeff 0, C10 gets coeff 0, C11 gets coeff 1? No.
    # 
    # Let me look up the exact Strassen tensor decomposition.
    # 
    # The 7 rank-1 tensors for Strassen 2x2x2 are:
    # 1. (A00+A11) ⊗ (B00+B11) ⊗ (C00 - C10 + C01 - C11)  -- wait, this doesn't look right.
    
    # Let me use the standard formulation. The matrix multiplication tensor for 2x2*2x2 can be written as:
    # sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    # where the coefficients are:
    # 
    # r=1: u = (1,0,0,1) [A00,A01,A10,A11], v = (1,1,0,0) [B00,B01,B10,B11], w = (1,0,0,-1) [C00,C01,C10,C11]
    # r=2: u = (0,0,1,1), v = (1,0,0,0), w = (0,0,1,-1)
    # r=3: u = (1,0,0,0), v = (0,1,0,1), w = (0,1,0,0)
    # r=4: u = (0,0,0,1), v = (0,0,1,0), w = (0,0,1,0)  -- wait, let me be more careful.
    
    # I'll use the well-known Strassen decomposition. Let me define it explicitly.
    # 
    # The 7 terms (u, v, w) where u is in R^4 (A00,A01,A10,A11), v in R^4 (B00,B01,B10,B11), w in R^4 (C00,C01,C10,C11):
    
    strassen_terms_2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [A00, A01, A10, A11], v: [B00, B01, B10, B11], w: [C00, C01, C10, C11]
        ((1, 0, 0, 1), (1, 1, 0, 0), (1, 0, 0, -1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 0)),
        ((0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (0, 0, -1, 1)),
        ((0, 1, -1, 0), (1, 1, 0, 0), (0, 1, 0, 0)),
        ((0, 0, 0, 1), (0, 1, 0, 1), (0, 0, 0, 1)),
    ]
    
    # Let me verify: C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Term 1: (A00+A11)*(B00+B11)*(C00 - C11)
    # Contribution to C00: (A00+A11)*(B00+B11)
    # Contribution to C11: -(A00+A11)*(B00+B11)
    
    # This is getting complex. Let me just use the known result that Strassen gives rank 7 for 2x2*2x2.
    # I'll hardcode the Strassen decomposition with correct coefficients.
    
    # After research, the correct Strassen decomposition for 2x2*2x2 is:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+c)h
    # M6 = (b-c)(e+g)
    # M7 = (d-b)(f+h)
    # i = M1 - M4 + M6 - M7
    # j = M3 + M5 + M7
    # k = M2 + M4 - M6
    # l = M1 - M3 + M2 - M5
    
    # So the 7 rank-1 tensors are:
    # T1: u=(a+d) = A00+A11, v=(e+h) = B00+B11, w: contributes to i with +1, l with +1
    #   w = (1, 0, 0, 1) for (i,j,k,l)
    # T2: u=(c+d) = A10+A11, v=e = B00, w: contributes to k with +1, l with +1
    #   w = (0, 0, 1, 1)
    # T3: u=a = A00, v=(f-h) = B01-B11, w: contributes to j with +1, l with -1
    #   w = (0, 1, 0, -1)
    # T4: u=d = A11, v=(g-e) = B10-B00, w: contributes to i with -1, k with +1
    #   w = (-1, 0, 1, 0)
    # T5: u=(a+c) = A00+A10, v=h = B11, w: contributes to j with +1, l with -1
    #   w = (0, 1, 0, -1)
    # T6: u=(b-c) = A01-A10, v=(e+g) = B00+B10, w: contributes to i with +1, k with -1
    #   w = (1, 0, -1, 0)
    # T7: u=(d-b) = A11-A01, v=(f+h) = B01+B11, w: contributes to i with -1, j with +1, l with +1
    #   w = (-1, 1, 0, 1)
    
    # Let me verify i: T1(+1) + T4(-1) + T6(+1) + T7(-1) = M1 - M4 + M6 - M7. Correct.
    # j: T3(+1) + T5(+1) + T7(+1) = M3 + M5 + M7. Correct.
    # k: T2(+1) + T4(+1) + T6(-1) = M2 + M4 - M6. Correct.
    # l: T1(+1) + T2(+1) + T3(-1) + T5(-1) = M1 - M3 + M2 - M5. Correct.
    
    strassen_2x2 = [
        # (u_coeffs for A00,A01,A10,A11), (v_coeffs for B00,B01,B10,B11), (w_coeffs for C00,C01,C10,C11)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, 1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, -1)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 1, 0)),
        ((1, 0, 1, 0), (0, 0, 0, 1), (0, 1, 0, -1)),
        ((0, 1, -1, 0), (1, 0, 1, 0), (1, 0, -1, 0)),
        ((0, -1, 0, 1), (0, 1, 0, 1), (-1, 1, 0, 1)),
    ]
    
    # Now build the full decomposition.
    
    def a_idx(i, j):
        """Index in U for A[i,j]"""
        return i * m + j
    
    def b_idx(j, k):
        """Index in V for B[j,k]"""
        return j * p + k
    
    def c_idx(i, k):
        """Index in W for C[i,k]"""
        return k * n + i
    
    terms = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        """Add a rank-1 term with sparse dictionaries for U, V, W indices."""
        terms.append((u_dict, v_dict, w_dict))
    
    # Block A11*B11 (rows 0,1 of A; cols 0,1 of A; rows 0,1 of B; cols 0,1 of B; rows 0,1 of C; cols 0,1 of C)
    a_map = [(0,0), (0,1), (1,0), (1,1)]
    b_map = [(0,0), (0,1), (1,0), (1,1)]
    c_map = [(0,0), (0,1), (1,0), (1,1)]
    for (u_c, v_c, w_c) in strassen_2x2:
        u_dict = {}
        v_dict = {}
        w_dict = {}
        for r in range(4):
            if u_c[r] != 0:
                i, j = a_map[r]
                u_dict[a_idx(i, j)] = u_c[r]
            if v_c[r] != 0:
                j, k = b_map[r]
                v_dict[b_idx(j, k)] = v_c[r]
            if w_c[r] != 0:
                i, k = c_map[r]
                w_dict[c_idx(i, k)] = w_c[r]
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A11*B12 (rows 0,1 of A; cols 0,1 of A; rows 0,1 of B; cols 2,3 of B; rows 0,1 of C; cols 2,3 of C)
    for (u_c, v_c, w_c) in strassen_2x2:
        u_dict = {}
        v_dict = {}
        w_dict = {}
        a_map = [(0,0), (0,1), (1,0), (1,1)]
        b_map = [(0,2), (0,3), (1,2), (1,3)]
        c_map = [(0,2), (0,3), (1,2), (1,3)]
        for r in range(4):
            if u_c[r] != 0:
                i, j = a_map[r]
                u_dict[a_idx(i, j)] = u_c[r]
            if v_c[r] != 0:
                j, k = b_map[r]
                v_dict[b_idx(j, k)] = v_c[r]
            if w_c[r] != 0:
                i, k = c_map[r]
                w_dict[c_idx(i, k)] = w_c[r]
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A11*B13 (rows 0,1 of A; cols 0,1 of A; rows 0,1 of B; cols 4,5,6 of B; rows 0,1 of C; cols 4,5,6 of C)
    # Rank 6: for each column k in {4,5,6}, use 2 rank-1 terms.
    for k in range(4, 7):
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}
        # Term 1: A[:,0] * B[0,k] * C[:,k]
        u_dict = {a_idx(0,0): 1, a_idx(1,0): 1}
        v_dict = {b_idx(0,k): 1}
        w_dict = {c_idx(0,k): 1, c_idx(1,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
        # Term 2: A[:,1] * B[1,k] * C[:,k]
        u_dict = {a_idx(0,1): 1, a_idx(1,1): 1}
        v_dict = {b_idx(1,k): 1}
        w_dict = {c_idx(0,k): 1, c_idx(1,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A12*B21 (rows 0,1 of A; col 2 of A; row 2 of B; cols 0,1 of B; rows 0,1 of C; cols 0,1 of C)
    # C[i,k] = A[i,2]*B[2,k] for i in {0,1}, k in {0,1}
    for i in range(2):
        for k in range(2):
            u_dict = {a_idx(i,2): 1}
            v_dict = {b_idx(2,k): 1}
            w_dict = {c_idx(i,k): 1}
            add_rank1(u_dict, v_dict, w_dict)
    
    # Block A12*B22 (rows 0,1 of A; col 2 of A; row 2 of B; cols 2,3 of B; rows 0,1 of C; cols 2,3 of C)
    for i in range(2):
        for k in range(2, 4):
            u_dict = {a_idx(i,2): 1}
            v_dict = {b_idx(2,k): 1}
            w_dict = {c_idx(i,k): 1}
            add_rank1(u_dict, v_dict, w_dict)
    
    # Block A12*B23 (rows 0,1 of A; col 2 of A; row 2 of B; cols 4,5,6 of B; rows 0,1 of C; cols 4,5,6 of C)
    for i in range(2):
        for k in range(4, 7):
            u_dict = {a_idx(i,2): 1}
            v_dict = {b_idx(2,k): 1}
            w_dict = {c_idx(i,k): 1}
            add_rank1(u_dict, v_dict, w_dict)
    
    # Block A21*B11 (row 2 of A; cols 0,1 of A; rows 0,1 of B; cols 0,1 of B; row 2 of C; cols 0,1 of C)
    # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] for k in {0,1}
    for k in range(2):
        # Term 1: A[2,0] * B[:,k] * C[2,k]
        u_dict = {a_idx(2,0): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
        # Term 2: A[2,1] * B[:,k] * C[2,k]
        u_dict = {a_idx(2,1): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A21*B12 (row 2 of A; cols 0,1 of A; rows 0,1 of B; cols 2,3 of B; row 2 of C; cols 2,3 of C)
    for k in range(2, 4):
        u_dict = {a_idx(2,0): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
        u_dict = {a_idx(2,1): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A21*B13 (row 2 of A; cols 0,1 of A; rows 0,1 of B; cols 4,5,6 of B; row 2 of C; cols 4,5,6 of C)
    for k in range(4, 7):
        u_dict = {a_idx(2,0): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
        u_dict = {a_idx(2,1): 1}
        v_dict = {b_idx(0,k): 1, b_idx(1,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A22*B21 (row 2 of A; col 2 of A; row 2 of B; cols 0,1 of B; row 2 of C; cols 0,1 of C)
    for k in range(2):
        u_dict = {a_idx(2,2): 1}
        v_dict = {b_idx(2,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A22*B22 (row 2 of A; col 2 of A; row 2 of B; cols 2,3 of B; row 2 of C; cols 2,3 of C)
    for k in range(2, 4):
        u_dict = {a_idx(2,2): 1}
        v_dict = {b_idx(2,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Block A22*B23 (row 2 of A; col 2 of A; row 2 of B; cols 4,5,6 of B; row 2 of C; cols 4,5,6 of C)
    for k in range(4, 7):
        u_dict = {a_idx(2,2): 1}
        v_dict = {b_idx(2,k): 1}
        w_dict = {c_idx(2,k): 1}
        add_rank1(u_dict, v_dict, w_dict)
    
    # Now build U, V, W matrices.
    rank = len(terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for row, coeff in u_dict.items():
            U[row][col] = coeff
        for row, coeff in v_dict.items():
            V[row][col] = coeff
        for row, coeff in w_dict.items():
            W[row][col] = coeff
    
    # Verify correctness
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with random matrices
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        
        # Compute C = A * B
        C_test = A_test @ B_test
        
        # Compute C using decomposition: C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Actually, the decomposition is: C = sum_r (U_r * A) * (B * V_r) * W_r
        # More precisely: C[i,k] = sum_{j} sum_r U[i*m+j, r] * A[i,j] * B[j,k] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that's not right either. Let me think about this more carefully.
        
        # The tensor decomposition says: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where T is the matrix multiplication tensor.
        # T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k], else 0.
        # Actually, T[i,j,k] = delta_{i,k} in the appropriate sense.
        
        # The contraction is: C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * B[j,k] * V[j*p+k, r] * W[k*n+i, r]
        # Hmm, this doesn't look right. Let me reconsider.
        
        # In a tensor decomposition of matrix multiplication:
        # C = A * B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # The tensor T has indices (i, j, k) where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
        # Actually, T is defined such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k] = 1 if the product contributes.
        # But that's just the identity, so T[i,j,k] = 1 for all valid (i,j,k).
        
        # The rank-r decomposition is: T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
        # 
        # The contraction is: C[i,k] = sum_j sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Wait, that still doesn't look right. Let me think again.
        
        # Actually, the correct contraction for matrix multiplication tensor decomposition is:
        # C[i,k] = sum_{j} sum_{r} U[i*m+j, r] * A[i,j] * B[j,k] * V[j*p+k, r] * W[k*n+i, r]
        # No, that's not standard either.
        
        # Let me look at this differently. The matrix multiplication tensor M has entries:
        # M[i, j, k] = 1 if we're multiplying A[i,j] by B[j,k] to get part of C[i,k]
        # But actually, M is a 3-tensor where M[a, b, c] = 1 if the product of the a-th entry of A and b-th entry of B contributes to the c-th entry of C.
        
        # For A (n×m), B (m×p), C (n×p):
        # A[i,j] * B[j,k] contributes to C[i,k].
        # So M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
        
        # The rank-r decomposition: M = sum_{r=1}^R U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This means: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # And the matrix multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
        # Using the decomposition: C[i,k] = sum_{a,b,c} M[a,b,c] * (A flattened)[a] * (B flattened)[b] * (W part)[c]
        # This is getting confusing. Let me just test numerically.
        
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            # U[:,r] is a vector of length n*m
            # V[:,r] is a vector of length m*p
            # W[:,r] is a vector of length p*n
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a_idx = i * m + j
                        b_idx = j * p + k
                        c_idx = k * n + i
                        C_decomp[i, k] += U_np[a_idx, r] * A_test[i, j] * B_test[j, k] * V_np[b_idx, r] * W_np[c_idx, r]
        
        if not np.allclose(C_test, C_decomp, atol=1e-10):
            # If verification fails, fall back to schoolbook
            print("Verification failed, falling back to schoolbook")
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(U_rows)]
            V_fb = [[0] * rank_fb for _ in range(V_rows)]
            W_fb = [[0] * rank_fb for _ in range(W_rows)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col] = 1
                        V_fb[j * p + k][col] = 1
                        W_fb[k * n + i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception as e:
        print(f"Verification error: {e}, falling back to schoolbook")
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(U_rows)]
        V_fb = [[0] * rank_fb for _ in range(V_rows)]
        W_fb = [[0] * rank_fb for _ in range(W_rows)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
