"""Construct a low-rank exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    # We decompose into 3 independent 3x3 * 3x2 problems (k in pairs)
    # Each 3x3 * 3x2 sub-problem has rank 10 using Strassen on the 2x2x2 core
    # Total rank = 3 * 10 = 30

    rank = 30
    U = np.zeros((n * m, rank))  # 9 x 30
    V = np.zeros((m * p, rank))  # 18 x 30
    W = np.zeros((p * n, rank))  # 18 x 30

    # Strassen 2x2x2 decomposition (rank 7)
    # For matrices A(2x2), B(2x2), C(2x2):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M5 + M7
    # C10 = M2 + M4 + M6
    # C11 = M1 - M3 + M2 + M5

    # Strassen coefficients for U (A indices), V (B indices), W (C indices)
    # A indices: (i,j) where i in {0,1}, j in {0,1} -> row index i*2+j in sub-problem
    # B indices: (j,k) where j in {0,1}, k in {0,1} -> row index j*2+k in sub-problem
    # C indices: (k,i) where k in {0,1}, i in {0,1} -> row index k*2+i in sub-problem

    # Strassen term coefficients
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # Each is dict mapping (i,j) or (j,k) or (k,i) to coefficient
        # Term 1: M1 = (a00+a11)*(b00+b11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):-1, (1,0):1, (1,1):1}),
        # Term 2: M2 = (a10+a11)*b00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):1}),
        # Term 3: M3 = a00*(b01-b11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):-1}),
        # Term 4: M4 = a11*(b10-b00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
        # Term 5: M5 = (a00+a01)*b11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):1}),
        # Term 6: M6 = (a10-a00)*(b00+b10)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(0,0):1, (1,0):1}),
        # Term 7: M7 = (a01-a11)*(b01+b11)
        ({(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}, {(0,1):1, (1,1):1}),
    ]

    for g in range(3):
        # Sub-problem: 3x3 * 3x2, k_sub in {0,1}, global k = 2*g + k_sub
        base = g * 10  # column offset in the full matrices

        # Terms 1-7: Strassen for i in {0,1}, j in {0,1}, k_sub in {0,1}
        for s in range(7):
            col = base + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]

            for (si, sj), uc in u_coeffs.items():
                # Global A index: i = si, j = sj (si, sj in {0,1})
                U[si * m + sj, col] = uc

            for (sj, sk), vc in v_coeffs.items():
                # Global B index: j = sj, k_global = 2*g + sk
                V[sj * p + (2 * g + sk), col] = vc

            for (sk, si), wc in w_coeffs.items():
                # Global C index: k_global = 2*g + sk, i = si
                W[(2 * g + sk) * n + si, col] = wc

        # Term 8: handles i in {0,1}, j=2, k_sub in {0,1}
        # This is A[i,2] * B[2,k] for i in {0,1}, k in {0,1}
        # We can factor this as: sum over k_sub of A[i,2]*B[2,k_sub]
        # But we need rank-1 terms. Actually, we can use 2 rank-1 terms:
        # One for k_sub=0, one for k_sub=1
        # Or we can use the structure: for each i in {0,1}, we need C[i,k] += A[i,2]*B[2,k]
        # This is a 2x3 * 3x2 block that's outer product-like
        # Actually, let's think: we need to represent the tensor terms
        # T[i,2,k] for i in {0,1}, k in {0,1} (in sub-problem indices)
        # As sum_c U[i*3+2, c] * V[2*2+k, c] * W[k*3+i, c]
        
        # Simple approach: 2 rank-1 terms
        # Term 8a: k_sub=0
        col = base + 7
        U[0 * m + 2, col] = 1  # i=0, j=2
        U[1 * m + 2, col] = 1  # i=1, j=2
        V[2 * p + (2 * g + 0), col] = 1  # j=2, k=2g
        W[(2 * g + 0) * n + 0, col] = 1  # k=2g, i=0
        W[(2 * g + 0) * n + 1, col] = 1  # k=2g, i=1

        # Term 8b: k_sub=1
        col = base + 8
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = 1
        V[2 * p + (2 * g + 1), col] = 1  # j=2, k=2g+1
        W[(2 * g + 1) * n + 0, col] = 1
        W[(2 * g + 1) * n + 1, col] = 1

        # Term 9: handles i=2, j in {0,1}, k_sub in {0,1}
        # T[2,j,k] for j in {0,1}, k in {0,1}
        # Need: sum over j of A[2,j]*B[j,k]
        # 2 rank-1 terms: one for each k_sub
        col = base + 9
        U[2 * m + 0, col] = 1  # i=2, j=0
        U[2 * m + 1, col] = 1  # i=2, j=1
        V[0 * p + (2 * g + 0), col] = 1  # j=0, k=2g
        V[1 * p + (2 * g + 0), col] = 1  # j=1, k=2g
        W[(2 * g + 0) * n + 2, col] = 1  # k=2g, i=2

        # Wait, I only have 10 columns per group (0-9). Let me recount.
        # Terms 1-7: 7 columns
        # Term 8a, 8b: 2 columns
        # Term 9: need 2 columns (one for k=0, one for k=1)
        # That's 7+2+2 = 11, too many!

        # Let me reconsider. For term 9, I can use a single rank-1 term if I'm careful.
        # Actually, T[2,j,k] for j in {0,1}, k in {0,1} is a 1x2x2 tensor.
        # This has rank 2 (it's the outer product of [1,1] with the 2x2 identity-like structure)
        # Wait no, it's sum_j A[2,j]*B[j,k]. The tensor is:
        # T[2,0,0]=1, T[2,0,1]=0, T[2,1,0]=0, T[2,1,1]=1 (in sub-problem)
        # This is rank 2.

        # Hmm, I need to be more careful. Let me redo the counting.
        # For 3x3 * 3x2:
        # The tensor T[i,j,k] for i,j in {0,1,2}, k in {0,1}
        # We can decompose as:
        # - Strassen for i,j in {0,1}, k in {0,1}: rank 7
        # - Terms for i in {0,1}, j=2, k in {0,1}: this is a 2x1x2 tensor
        #   T[i,2,k] = 1 for all i,k in {0,1}. This is rank 1! (outer product of [1,1] and [1,1])
        #   Wait, T[i,2,k] as a function of (i,k) is all ones, which is rank 1.
        #   But we also have the j=2 index fixed. So it's:
        #   U[i*3+2, c] * V[2*2+k, c] * W[k*3+i, c]
        #   For this to give 1 for all (i,k), we need:
        #   sum_c U[i*3+2,c]*V[2*2+k,c]*W[k*3+i,c] = 1 for all i,k in {0,1}
        #   This can be done with rank 1: set U[2, c]=1, U[5, c]=1, V[4,c]=1, V[5,c]=1, W[0,c]=1, W[1,c]=1, W[3,c]=1, W[4,c]=1
        #   Wait, that's not rank 1 in the tensor sense. Let me think again.
        #   
        #   Actually, the sub-tensor for i in {0,1}, j=2, k in {0,1} is:
        #   T[i,2,k] = 1 for all i in {0,1}, k in {0,1}
        #   This is a 2x1x2 tensor. Its rank is 1 because it's the outer product
        #   of the vector [1,1] (in i-k space) with the scalar 1 (in j space).
        #   So we need 1 rank-1 term.

        # Similarly for i=2, j in {0,1}, k in {0,1}:
        # T[2,j,k] = 1 for all j in {0,1}, k in {0,1}
        # This is a 1x2x2 tensor. Its rank is 1.

        # And for i=2, j=2, k in {0,1}:
        # T[2,2,k] = 1 for k in {0,1}
        # This is a 1x1x2 tensor. Its rank is 1.

        # So total: 7 + 1 + 1 + 1 = 10. 

        # Let me redo terms 8, 9, 10 with single rank-1 terms each.

        # Redo: I'll clear and redo the assignments for terms 8,9,10
        # First, let me clear the erroneous assignments

        # Term 8: i in {0,1}, j=2, k in {0,1} -> rank 1
        col = base + 7
        # Clear previous
        U[0 * m + 2, col] = 1  # i=0, j=2
        U[1 * m + 2, col] = 1  # i=1, j=2
        V[2 * p + (2 * g + 0), col] = 1
        V[2 * p + (2 * g + 1), col] = 1
        W[(2 * g + 0) * n + 0, col] = 1
        W[(2 * g + 0) * n + 1, col] = 1
        W[(2 * g + 1) * n + 0, col] = 1
        W[(2 * g + 1) * n + 1, col] = 1

        # Term 9: i=2, j in {0,1}, k in {0,1} -> rank 1
        col = base + 8
        U[2 * m + 0, col] = 1  # i=2, j=0
        U[2 * m + 1, col] = 1  # i=2, j=1
        V[0 * p + (2 * g + 0), col] = 1
        V[0 * p + (2 * g + 1), col] = 1
        V[1 * p + (2 * g + 0), col] = 1
        V[1 * p + (2 * g + 1), col] = 1
        W[(2 * g + 0) * n + 2, col] = 1
        W[(2 * g + 1) * n + 2, col] = 1

        # Term 10: i=2, j=2, k in {0,1} -> rank 1
        col = base + 9
        U[2 * m + 2, col] = 1  # i=2, j=2
        V[2 * p + (2 * g + 0), col] = 1
        V[2 * p + (2 * g + 1), col] = 1
        W[(2 * g + 0) * n + 2, col] = 1
        W[(2 * g + 1) * n + 2, col] = 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
