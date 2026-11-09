"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # For 4x4 * 4x7 matrix multiplication:
    # A is 4x4, B is 4x7, C = A*B is 4x7
    # 
    # Split into blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B12,B21,B22 are 2x2 and B13,B23 are 2x3
    # C = [[C11, C12, C13], [C21, C22, C23]] where C11,C12,C21,C22 are 2x2 and C13,C23 are 2x3
    #
    # For the 4x4x4 part (columns 0-3 of B), use Strassen (7 multiplications of 2x2)
    # For the 4x4x3 part (columns 4-6 of B), use schoolbook (4 multiplications of 2x2x3)
    #
    # Total rank: 7*4 + 4*4*3/2... let me compute more carefully.
    # Actually, each 2x2*2x2 multiplication has rank 4 in scalar terms.
    # Each 2x2*2x3 multiplication has rank 4*3=12 in scalar terms... no wait.
    # 
    # Let me think differently. In tensor decomposition:
    # Each rank-1 term picks (i,j,k) with coefficients.
    # Schoolbook for 4x4*4x7: rank = 4*4*7 = 112
    # 
    # Strassen for the 4x4x4 part: 7 block multiplications, each 2x2*2x2
    # Each 2x2*2x2 can be done with rank 4 (schoolbook at that level)
    # So 7*4 = 28 terms for the 4x4x4 part
    # 
    # Schoolbook for the 4x4x3 part: 4*4*3 = 48 terms
    # Total: 28 + 48 = 76 terms
    # 
    # Wait, that's not right either. Let me reconsider.
    #
    # Actually, the Strassen algorithm for 2x2*2x2 matrices has rank 7 in the 
    # "block" sense, but when we expand to scalars, each block multiplication
    # (which is a 2x2*2x2 product) requires 4 scalar multiplications.
    # So 7 block multiplications * 4 scalar mults each = 28 scalar terms.
    #
    # But wait - that's not how tensor rank works. Let me reconsider.
    #
    # The tensor T[i,j,k] for matrix mult satisfies T[i,j,k] = delta condition.
    # In Strassen, we replace some of the rank-1 terms with combinations.
    #
    # Let me just build it carefully.
    
    terms = []  # Each term: (i, j, k, u_coeff, v_coeff, w_coeff)
    
    # For the 4x4x4 part (k in 0..3), use Strassen
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    # A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4]
    # B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    
    # Strassen multiplications:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Results:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2 matrix product. In scalar tensor terms,
    # M[di,dk] = sum_dj X[di,dj]*Y[dj,dk]
    # where X and Y are 2x2 blocks of A and B.
    
    # For M1 = (A11+A22)*(B11+B22):
    # X[di,dj] = A[di,dj] + A[di+2,dj+2]
    # Y[dj,dk] = B[dj,dk] + B[dj+2,dk+2]
    # M1[di,dk] = sum_dj (A[di,dj]+A[di+2,dj+2])*(B[dj,dk]+B[dj+2,dk+2])
    # = sum_dj A[di,dj]*B[dj,dk] + A[di,dj]*B[dj+2,dk+2] + A[di+2,dj+2]*B[dj,dk] + A[di+2,dj+2]*B[dj+2,dk+2]
    
    # M1 contributes to C11 (C[0:2,0:2]) with +1 and C22 (C[2:4,2:4]) with +1
    
    def add_strassen_term(di, dj, dk, i_offset, j_offset, k_offset, c_offset, coeff):
        """Add a term for A[i_offset+di, j_offset+dj] * B[j_offset+dj, k_offset+dk] -> C[c_offset+di, c_offset+dk]"""
        i = i_offset + di
        j = j_offset + dj
        k = k_offset + dk
        ci = c_offset + di
        ck = c_offset + dk
        terms.append((i, j, k, 1, 1, coeff))
    
    # M1 = (A11+A22)*(B11+B22)
    # Contributes +1 to C11 (offset 0,0) and +1 to C22 (offset 2,2)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                # A[di,dj]*B[dj,dk] -> C[di,dk]
                terms.append((di, dj, dk, 1, 1, 1))
                # A[di,dj]*B[dj+2,dk+2] -> C[di,dk]
                terms.append((di, dj, dk+2, 1, 1, 1))
                # A[di+2,dj+2]*B[dj,dk] -> C[di,dk]
                terms.append((di+2, dj+2, dk, 1, 1, 1))
                # A[di+2,dj+2]*B[dj+2,dk+2] -> C[di,dk]
                terms.append((di+2, dj+2, dk+2, 1, 1, 1))
                # Same terms contribute to C22 with +1
                terms.append((di, dj, dk, 1, 1, 1))
                terms.append((di, dj, dk+2, 1, 1, 1))
                terms.append((di+2, dj+2, dk, 1, 1, 1))
                terms.append((di+2, dj+2, dk+2, 1, 1, 1))
    
    # This approach is getting too complex and error-prone.
    # Let me just use schoolbook to ensure correctness, as the Strassen
    # implementation in tensor form is very tricky to get right.
    
    # Reset terms
    terms = []
    
    # Schoolbook: each (i,j,k) triple gives one term
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k, 1, 1, 1))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (i, j, k, u, v, w) in enumerate(terms):
        U[i * m + j][r] = u
        V[j * p + k][r] = v
        W[k * n + i][r] = w
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
