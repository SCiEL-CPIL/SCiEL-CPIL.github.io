"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _build_strassen_222():
    """
    Rank-7 decomposition of 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*m + j, c] where m=2
    V[j*p + k, c] where p=2
    W[k*n + i, c] where n=2
    """
    # Strassen's 7 multiplication formulas for 2x2 matrices
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # c00 = M1 - M4 + M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 + M7
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a11 - a10) * (b10 + b11)
    
    # U rows: (i,j) for i in 0..1, j in 0..1 -> row index i*2+j
    # V rows: (j,k) for j in 0..1, k in 0..1 -> row index j*2+k
    # W rows: (k,i) for k in 0..1, i in 0..1 -> row index k*2+i
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Column 0: M1
    # (a00 + a11) * (b00 + b11)
    U[0][0] = 1  # a00
    U[3][0] = 1  # a11
    V[0][0] = 1  # b00
    V[3][0] = 1  # b11
    # Contribution to C:
    # c00 += 1, c01 += 0, c10 += 0, c11 += 0? No.
    # The term is (a00+a11)(b00+b11).
    # In tensor form, this rank-1 term contributes to C[i,k] as sum over j of U[i,j] V[j,k] W[k,i]?
    # Wait, the decomposition is C[i,k] = sum_c U[i,j_c] V[j_c,k] W[k,i]? No.
    # The standard form is: C[i,k] = sum_c U[i, c] V[c, k] ... no, that's not right for matrix mult.
    # The tensor T[i, j, k] = delta_{i,k} ... no.
    # Matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i, j, k] where A[i,j] and B[j,k] are inputs, C[i,k] is output.
    # Decomposition: T[i, j, k] = sum_c U[i, c] V[j, c] W[k, c]?
    # Let's check the interface:
    # U: (n*m, rank) -> U[i*m + j, c]
    # V: (m*p, rank) -> V[j*p + k, c]
    # W: (p*n, rank) -> W[k*n + i, c]
    # The identity is: C[i, k] = sum_j A[i, j] B[j, k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) * W[k*n+i, c]?
    # No, that's not how it works.
    # The correct identity for the tensor decomposition of matrix multiplication is:
    # sum_c U[i, j, c] V[j, k, c] W[k, i, c] ? No.
    # The tensor is T_{i, j, k} such that <A, B> = sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k}?
    # Actually, the standard definition is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as: C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) ... no.
    # Let's look at the schoolbook example in the prompt.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # For a specific c corresponding to (i0, j0, k0):
    # U[i*m+j, c] is 1 only if i=i0, j=j0.
    # V[j*p+k, c] is 1 only if j=j0, k=k0.
    # W[k*n+i, c] is 1 only if k=k0, i=i0.
    # So the term for c is:
    # (sum_{i,j} U[i*m+j, c] A[i,j]) * (sum_{j,k} V[j*p+k, c] B[j,k]) * ... wait.
    # The output is C. How is C constructed from U, V, W?
    # The prompt says "W[k*n+i, r]".
    # Usually, the decomposition is:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])?
    # No, that would be a product of three sums, which is cubic.
    # It must be:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) ... no.
    # Let's re-read the schoolbook.
    # For each (i0, j0, k0), we have a column c.
    # U[i0*m+j0, c] = 1.
    # V[j0*p+k0, c] = 1.
    # W[k0*n+i0, c] = 1.
    # The contribution to C[i, k] from this term must be A[i0, j0] * B[j0, k0] if i=i0 and k=k0, else 0.
    # So C[i, k] += A[i0, j0] * B[j0, k0] * (1 if i==i0 and k==k0 else 0).
    # This implies:
    # C[i, k] = sum_c (sum_{i', j'} U[i'*m+j', c] A[i', j']) * (sum_{j', k'} V[j'*p+k', c] B[j', k']) * W[k*n+i, c]?
    # No, that's not linear in A and B.
    # The correct bilinear form is:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k]) ... no.
    # It is:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i, j]) * (sum_j V[j*p+k, c] B[j, k]) ... no.
    # Let's look at the dimensions.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    # The standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    # T_{i, j, k} = sum_c U_{i, c} V_{j, c} W_{k, c} ? No, indices don't match.
    # The tensor is T_{i, j, k} where i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # T_{i, j, k} = 1 if i==j==k? No.
    # T_{i, j, k} is such that sum_{i,j,k} A_{i,j} B_{j,k} T_{i,j,k} is not C.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So T_{i, j, k} = delta_{i, i} delta_{k, k} ... no.
    # T_{i, j, k} is defined such that C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # This is a 3-way tensor T of size n x m x p? No, n x m x p is not right.
    # The tensor is of size (n*p) x (n*m) x (m*p)?
    # Let's stick to the prompt's interface.
    # The evaluator likely checks:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * ... no.
    # It must be:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) ... no.
    # Actually, the standard decomposition is:
    # C_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) ... no.
    # Let's assume the identity is:
    # C_{i,k} = sum_c W_{k,n+i,c} * (sum_j U_{i,m+j,c} A_{i,j}) * (sum_j V_{j,p+k,c} B_{j,k}) ... no.
    # Wait, the schoolbook example has U, V, W all 1s for specific indices.
    # For a specific (i0, j0, k0), the term is A[i0, j0] * B[j0, k0].
    # This term contributes to C[i0, k0].
    # So C[i0, k0] += A[i0, j0] B[j0, k0].
    # This is consistent with:
    # C[i, k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * W[k*n+i, c]?
    # If U, V, W are 1s, then sum_j U[...] A[i,j] = A[i0, j0] (if i=i0).
    # sum_j V[...] B[j,k] = B[j0, k0] (if k=k0).
    # W[k*n+i, c] = 1 (if i=i0, k=k0).
    # So the product is A[i0, j0] B[j0, k0] * 1.
    # This works!
    # So the identity is:
    # C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # Wait, this is cubic in the inputs? No, A and B are inputs, U, V, W are constants.
    # It is bilinear in A and B.
    # Let's verify.
    # Term c:
    # S_A = sum_j U[i*m+j, c] A[i,j]
    # S_B = sum_j V[j*p+k, c] B[j,k]
    # C[i,k] += W[k*n+i, c] * S_A * S_B
    # This is correct.
    
    # Now, Strassen's algorithm.
    # We need to express C[i,k] as a sum of 7 terms of the form W_c * (sum_j U_c A) * (sum_j V_c B).
    # Actually, Strassen's algorithm computes 7 products M1..M7.
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a11-a10)(b10+b11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 + M7
    
    # Each M is a product of two linear combinations.
    # M1 = (a00+a11) * (b00+b11)
    # For M1, U part: a00+a11. So U[0,0]=1, U[3,0]=1 (for row 0 and 3 of U, which correspond to (0,0) and (1,1)).
    # V part: b00+b11. So V[0,0]=1, V[3,0]=1 (for row 0 and 3 of V, which correspond to (0,0) and (1,1)).
    # W part: M1 contributes to C00 (+1) and C11 (+1).
    # So W[0,0]=1 (for k=0, i=0 -> index 0*2+0=0) and W[3,0]=1 (for k=1, i=1 -> index 1*2+1=3).
    
    # Let's build the matrices for M1..M7.
    # U: 4x7, V: 4x7, W: 4x7.
    # Rows of U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Rows of V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Rows of W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 += 1, C11 += 1
    
    # M2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10 += 1
    
    # M3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01 += 1
    
    # M4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1  # C00 -= 1, C10 += 1
    
    # M5: (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1  # C00 += 1, C01 += 1
    
    # M6: (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11 += 1
    
    # M7: (a11-a10)(b10+b11)
    U[3][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1  # C00 += 1, C11 += 1
    
    return U, V, W

def _build_strassen_223():
    """
    Rank-11 decomposition of 2x2 * 2x3.
    This is Strassen 2x2*2x2 (rank 7) + trivial 2x2*2x1 (rank 4).
    Total rank 11.
    """
    U_222, V_222, W_222 = _build_strassen_222()
    
    # Trivial part for the third column of B.
    # We need to compute C[:, 2] = A * B[:, 2].
    # This is a 2x2 * 2x1 multiplication.
    # Rank 4 decomposition.
    # U: 4x4, V: 4x4, W: 4x4.
    # But we are building 2x2*2x3, so V has 6 rows (2*3), W has 6 rows (3*2).
    # The trivial part uses the last column of B (k=2) and the last column of C (k=2).
    # U_triv: 4x4, V_triv: 6x4 (only rows 4,5 non-zero), W_triv: 6x4 (only rows 4,5 non-zero).
    
    U_triv = np.zeros((4, 4), dtype=float)
    V_triv = np.zeros((6, 4), dtype=float)
    W_triv = np.zeros((6, 4), dtype=float)
    
    # Schoolbook for 2x2 * 2x1.
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    
    # Term 0: A[0,0]B[0,2] -> C[0,2]
    U_triv[0][0] = 1
    V_triv[4][0] = 1  # j=0, k=2 -> index 0*3+2=4
    W_triv[4][0] = 1  # k=2, i=0 -> index 2*2+0=4
    
    # Term 1: A[0,1]B[1,2] -> C[0,2]
    U_triv[1][1] = 1
    V_triv[5][1] = 1  # j=1, k=2 -> index 1*3+2=5
    W_triv[4][1] = 1
    
    # Term 2: A[1,0]B[0,2] -> C[1,2]
    U_triv[2][2] = 1
    V_triv[4][2] = 1
    W_triv[5][2] = 1  # k=2, i=1 -> index 2*2+1=5
    
    # Term 3: A[1,1]B[1,2] -> C[1,2]
    U_triv[3][3] = 1
    V_triv[5][3] = 1
    W_triv[5][3] = 1
    
    # Combine
    U = np.hstack([U_222, U_triv])
    V = np.vstack([V_222, V_triv[:4, :]])  # V_222 is 4x7, V_triv is 6x4. We need to pad V_222 to 6x7.
    # Wait, V_222 is for 2x2*2x2, so it has 4 rows. V_triv has 6 rows.
    # We need to pad V_222 with 2 rows of zeros.
    V_222_padded = np.vstack([V_222, np.zeros((2, 7), dtype=float)])
    V = np.hstack([V_222_padded, V_triv])
    
    W_222_padded = np.vstack([W_222, np.zeros((2, 7), dtype=float)])
    W = np.hstack([W_222_padded, W_triv])
    
    return U, V, W

def construct():
    """
    Construct a rank-33 decomposition for 2x2 * 2x9.
    This is 3 * rank-11 decomposition of 2x2 * 2x3.
    """
    U_223, V_223, W_223 = _build_strassen_223()
    
    # U is 4x11, V is 6x11, W is 6x11.
    # We need to replicate this 3 times for the 3 blocks of 2x3.
    # U: 4x33
    # V: 18x33 (6*3)
    # W: 18x33 (6*3)
    
    U = np.zeros((4, 33), dtype=float)
    V = np.zeros((18, 33), dtype=float)
    W = np.zeros((18, 33), dtype=float)
    
    for block in range(3):
        U[:, block*11:(block+1)*11] = U_223
        V[block*6:(block+1)*6, block*11:(block+1)*11] = V_223
        W[block*6:(block+1)*6, block*11:(block+1)*11] = W_223
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
