import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=3
    # A is n x m, B is m x p, C is n x p
    
    # Block decomposition:
    # A = [[A11, A12], [A21, A22]] where A11,A21 are 1x2, A12,A22 are 1x1
    # Actually: A is 2x3. Let's split columns: first 2 cols form 2x2, last col forms 2x1
    # B is 3x3. Split rows: first 2 rows form 2x3, last row forms 1x3
    
    # C = A*B
    # C = [[A11, A12], [A21, A22]] * [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    # where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B11 is 2x2, B12 is 2x1, B13 is 2x1, B21 is 1x2, B22 is 1x1, B23 is 1x1, B31 is 1x2, B32 is 1x1, B33 is 1x1
    
    # Let me use a different block structure:
    # Split A into 2x2 and 2x1 (columns)
    # Split B into 2x3 and 1x3 (rows)
    # C = A_left * B_top + A_right * B_bottom
    # A_left is 2x2, B_top is 2x3 -> product is 2x3 (use Strassen-like)
    # A_right is 2x1, B_bottom is 1x3 -> product is 2x3 (rank 2)
    # But we also need cross terms...
    
    # Actually, let's think more carefully.
    # A (2x3) * B (3x3) = C (2x3)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's split the sum over j=0,1,2:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # Group: (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + A[i,2]*B[2,k]
    # First part: 2x2 matrix mult (A[:,0:2] * B[0:2,:]) - use Strassen rank 7
    # Second part: 2x1 * 1x3 - rank 2 (each output element needs one multiplication)
    # Total: 7 + 2 = 9? But wait, Strassen works for 2x2*2x2.
    
    # Let me reconsider. A is 2x3, B is 3x3.
    # A_left = A[:, 0:2] is 2x2, A_right = A[:, 2:3] is 2x1
    # B_top = B[0:2, :] is 2x3, B_bottom = B[2:3, :] is 1x3
    
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For A_left @ B_top: this is (2x2) @ (2x3)
    # We can view this as: for each column k of B_top (which is 2x1), compute A_left @ B_top[:,k]
    # That's 3 separate 2x2 @ 2x1 multiplications, each rank 2, total rank 6
    # OR we can use a different approach.
    
    # Actually, let's just enumerate all possible rank-r decompositions.
    # The trivial rank is n*m*p = 18.
    # Let's try to find a lower rank by systematic search.
    
    # For now, let's use the trivial decomposition and see if we can improve.
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
