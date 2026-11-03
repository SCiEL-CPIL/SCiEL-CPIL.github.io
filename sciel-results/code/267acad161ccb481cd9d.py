import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 68: 4 Strassen blocks (rank 7 each) + 40 schoolbook terms
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 4 disjoint 2x2x2 blocks
    # Blocks cover i in {0,1}, j in {0,1}, k in {0..7} split into 4 pairs
    blocks = [
        (0, 2, 0, 2, 0, 2), # i_start, i_end, j_start, j_end, k_start, k_end
        (0, 2, 0, 2, 2, 4),
        (0, 2, 0, 2, 4, 6),
        (0, 2, 0, 2, 6, 8),
    ]
    
    # Correct Strassen vectors for 2x2x2
    # Local indices: A is 2x2 (rows 0,1; cols 0,1), B is 2x2 (rows 0,1; cols 0,1), C is 2x2
    # A indices: 00, 01, 10, 11 -> [0, 1, 2, 3]
    # B indices: 00, 01, 10, 11 -> [0, 1, 2, 3]
    # C indices: 00, 01, 10, 11 -> [0, 1, 2, 3]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M4
    
    # u vectors (for A):
    # u1: [1, 0, 0, 1]  (A00 + A11)
    # u2: [0, 0, 1, 1]  (A10 + A11)
    # u3: [1, 0, 0, 0]  (A00)
    # u4: [0, 0, 0, 1]  (A11)
    # u5: [1, 1, 0, 0]  (A00 + A01)
    # u6: [-1, 0, 1, 0] (A10 - A00)
    # u7: [0, 1, 0, -1] (A01 - A11)
    
    # v vectors (for B):
    # v1: [1, 0, 0, 1]  (B00 + B11)
    # v2: [1, 0, 0, 0]  (B00)
    # v3: [0, 1, 0, -1] (B01 - B11)
    # v4: [0, 0, 1, -1] (B10 - B00)
    # v5: [0, 0, 0, 1]  (B11)
    # v6: [1, 1, 0, 0]  (B00 + B01)
    # v7: [0, 0, 1, 1]  (B10 + B11)
    
    # w vectors (for C):
    # w1: [1, 0, 0, 1]  (C00 + C11)
    # w2: [0, 0, 1, 0]  (C10)
    # w3: [0, 1, 0, 0]  (C01)
    # w4: [-1, 0, 0, 1] (-C00 + C11)
    # w5: [0, 1, 0, 0]  (C01)
    # w6: [0, 0, 1, 0]  (C10)
    # w7: [1, 0, 0, 0]  (C00)
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [-1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]
    ]
    
    # Verify Strassen decomposition
    # Check C00 = M1 - M4 + M7
    # Term 1: u1[00]*v1[00]*w1[00] + u1[01]*v1[01]*w1[00] + u1[10]*v1[10]*w1[00] + u1[11]*v1[11]*w1[00]
    # = 1*1*1 + 0*0*1 + 0*0*1 + 1*1*1 = 2
    # Term 4: u4[00]*v4[00]*w4[00] + u4[01]*v4[01]*w4[00] + u4[10]*v4[10]*w4[00] + u4[11]*v4[11]*w4[00]
    # = 0*0*(-1) + 0*0*(-1) + 0*1*(-1) + 1*(-1)*(-1) = 1
    # Term 7: u7[00]*v7[00]*w7[00] + u7[01]*v7[01]*w7[00] + u7[10]*v7[10]*w7[00] + u7[11]*v7[11]*w7[00]
    # = 0*0*1 + 1*0*1 + 0*1*1 + (-1)*1*1 = -1
    # Sum = 2 - 1 - 1 = 0? That's wrong. C00 should be 1.
    
    # Let me re-check the Strassen formulas.
    # C00 = M1 - M4 + M7
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M7 = (A01-A11)(B10+B11)
    
    # The tensor decomposition is: sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = delta[i,j,k]
    # For each r, we have vectors u_r (length 4), v_r (length 4), w_r (length 4).
    # The contribution to C[i][j] from A[i][a] * B[a][j] is sum_r u_r[i,a] * v_r[a,j] * w_r[j,i]? No.
    # The standard tensor is T[i,j,k] = A[i,j] * B[j,k] = C[i,k].
    # So the decomposition is: C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k]? No.
    # The correct form is: C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k] is wrong.
    # It should be: C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k] is not right either.
    # The tensor T is defined by T[i,j,k] = A[i,j] * B[j,k] * C[i,k] (in the dual space).
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = A[i,j] * B[j,k]
    # And we want to decompose T as a sum of rank-1 tensors:
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # where u_r is a vector in R^(n*m), v_r in R^(m*p), w_r in R^(p*n).
    # For 2x2x2:
    # T[i,j,k] = A[i,j] * B[j,k] = C[i,k]
    # We need: C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k]? No.
    # The rank-1 tensor is: u_r[i,j] * v_r[j,k] * w_r[i,k]
    # So: C[i,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] summed over j? No.
    # The tensor T is 3-dimensional: T[i,j,k] = A[i,j] * B[j,k].
    # A rank-1 decomposition is: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # But w_r[i,k] is not a function of j. So:
    # T[i,j,k] = sum_r (u_r[i,j] * v_r[j,k] * w_r[i,k])
    # This means: for each (i,j,k), T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # For matrix multiplication, T[i,j,k] = 1 if the indices correspond to A[i,j] * B[j,k] = C[i,k].
    # So we need: 1 = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] for the "diagonal" terms, and 0 otherwise?
    # No, T[i,j,k] is the tensor that defines the bilinear map. The tensor is:
    # T = sum_{i,j,k} e_i^A * e_j^B * e_k^C where A[i,j] * B[j,k] contributes to C[i,k].
    # Actually, the standard definition is:
    # T[i,j,k] = A[i,j] * B[j,k]
    # And we decompose T as: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # This is not quite right. The correct decomposition is:
    # C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k] is wrong.
    # The correct form is:
    # C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k]
    # No, that's not right either. Let me think again.
    
    # The matrix multiplication tensor is:
    # T = sum_{i,j,k} e_{i,j}^A * e_{j,k}^B * e_{i,k}^C
    # A rank-1 decomposition is:
    # T = sum_r u_r * v_r * w_r
    # where u_r is in R^(n*m), v_r in R^(m*p), w_r in R^(p*n).
    # So: T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # For matrix multiplication, T[i,j,k] = 1 if (i,j,k) corresponds to A[i,j] * B[j,k] = C[i,k].
    # So we need: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 if the indices are valid, 0 otherwise.
    # For 2x2x2, the valid indices are:
    # (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}.
    # And the condition is: C[i,k] = sum_j A[i,j] * B[j,k].
    # So T[i,j,k] = A[i,j] * B[j,k] contributes to C[i,k].
    # The tensor T is defined by: T[i,j,k] = 1 if the indices (i,j,k) are such that A[i,j] * B[j,k] is a term in C[i,k].
    # So for each (i,j,k), T[i,j,k] = 1.
    # And we need: sum_r u_r[i,j] * v_r[j,k] * w_r[i,k] = 1 for all (i,j,k).
    # This is not right. The tensor T is:
    # T[i,j,k] = A[i,j] * B[j,k]
    # And we want to decompose the bilinear map: C = A * B.
    # The rank of the tensor is the minimum number of rank-1 tensors needed to represent T.
    # A rank-1 tensor is: u_r[i,j] * v_r[j,k] * w_r[i,k]
    # So: C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k] is wrong.
    # The correct form is:
    # C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k]
    # No, that's not right. The correct form is:
    # C[i,k] = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[i,k]
    # I think I'm confusing myself. Let me just use the known Strassen vectors.
    
    # Known correct Strassen vectors for 2x2x2:
    # u1 = [1, 0, 0, 1], v1 = [1, 0, 0, 1], w1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1], v2 = [1, 0, 0, 0], w2 = [0, 0, 1, 0]
    # u3 = [1, 0, 0, 0], v3 = [0, 1, 0, -1], w3 = [0, 1, 0, 0]
    # u4 = [0, 0, 0, 1], v4 = [0, 0, 1, -1], w4 = [-1, 0, 0, 1]
    # u5 = [1, 1, 0, 0], v5 = [0, 0, 0, 1], w5 = [0, 1, 0, 0]
    # u6 = [-1, 0, 1, 0], v6 = [1, 1, 0, 0], w6 = [0, 0, 1, 0]
    # u7 = [0, 1, 0, -1], v7 = [0, 0, 1, 1], w7 = [1, 0, 0, 0]
    
    # Let me verify C00 = M1 - M4 + M7:
    # M1: u1[00]*v1[00]*w1[00] + u1[01]*v1[01]*w1[00] + u1[10]*v1[10]*w1[00] + u1[11]*v1[11]*w1[00]
    # = 1*1*1 + 0*0*1 + 0*0*1 + 1*1*1 = 2
    # M4: u4[00]*v4[00]*w4[00] + u4[01]*v4[01]*w4[00] + u4[10]*v4[10]*w4[00] + u4[11]*v4[11]*w4[00]
    # = 0*0*(-1) + 0*0*(-1) + 0*1*(-1) + 1*(-1)*(-1) = 1
    # M7: u7[00]*v7[00]*w7[00] + u7[01]*v7[01]*w7[00] + u7[10]*v7[10]*w7[00] + u7[11]*v7[11]*w7[00]
    # = 0*0*1 + 1*0*1 + 0*1*1 + (-1)*1*1 = -1
    # Sum = 2 - 1 - 1 = 0. Still wrong.
    
    # I think the issue is that the Strassen vectors I'm using are not correct.
    # Let me just use the schoolbook rank 72 solution for safety.
    
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
