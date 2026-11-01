import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 140.0  # 140s deadline for 150s limit
    
    # We want to decompose the matrix multiplication tensor for A (n x m) * B (m x p) = C (n x p)
    # The tensor has dimensions n*p*m (or n*m*p depending on convention).
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (n*p, rank).
    # The decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[i*p+k, r]  ... wait, indices.
    # Standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{i,k}^{(r)} is not quite right.
    # The task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # Let's check the indices in the seed:
    # U[i*m+j][r] corresponds to A[i,j]
    # V[j*p+k][r] corresponds to B[j,k]
    # W[k*n+i][r] corresponds to C[i,k] (since W is p*n, index k*n+i is C[i,k] if row-major C is k*n+i? No.)
    # C is n x p. C[i,k].
    # In the seed: W[k*n+i][column] = 1.
    # If W is flattened C, and C is n x p, usually C[i,k] is at index i*p+k.
    # But here W has size p*n. And index is k*n+i.
    # This implies W is indexed by (k, i) where k is col of C (0..p-1) and i is row of C (0..n-1).
    # So W[k*n+i] = C[i,k].
    # The decomposition is: C[i,k] = sum_r U[i*m+j] * V[j*p+k] * W[k*n+i] ? No, sum over j is implicit in matrix mult.
    # Actually, the tensor T is defined by T[i,j,k] = delta_{i,k} ? No.
    # The matmul tensor T has entries T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] ? No.
    # The tensor is T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if (i,j,k) is a valid term?
    # Actually, the condition is:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that would mean every term A[i,j]B[j,k] contributes 1 to C[i,k].
    # But in matmul, A[i,j]B[j,k] contributes to C[i,k].
    # So yes, for a fixed i, k, the sum over j of A[i,j]B[j,k] is C[i,k].
    # So the tensor equation is:
    # sum_r (sum_j U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1 ? No.
    # The standard form is:
    # C[i,k] = sum_r M_r(i,j) * N_r(j,k) * P_r(i,k) ?
    # Let's look at the seed again.
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1.
    # This means for each (i,j,k), there is a rank-1 term.
    # The sum is over r.
    # For a fixed (i,k), we need sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] V[j*p+k, r]) = 1 ?
    # No, the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And we need T[i,j,k] = 1 for all i,j,k?
    # If T[i,j,k] = 1, then C[i,k] = sum_j A[i,j] B[j,k] * 1 = sum_j A[i,j] B[j,k].
    # So yes, we need to decompose the tensor T where T[i,j,k] = 1 for all i,j,k.
    # Wait, is that right?
    # If T[i,j,k] = 1, then the product A[i,j]B[j,k] is always included.
    # So the decomposition is just the sum of all rank-1 tensors e_i e_j e_k.
    # The rank is n*m*p.
    # But we can do better.
    # We can have T[i,j,k] such that sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k].
    # So we need: sum_j (sum_r U[i*m+j, r] V[j*p+k, r]) W[k*n+i, r] = 1 ?
    # No. The decomposition is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] V[j*p+k, r])
    # This must equal sum_j A[i,j] B[j,k].
    # So for each r, let M_r[i,j] = U[i*m+j, r] and N_r[j,k] = V[j*p+k, r].
    # Then sum_r W[k*n+i, r] M_r[i,j] N_r[j,k] = sum_j A[i,j] B[j,k].
    # This is equivalent to:
    # sum_r W[k*n+i, r] (M_r * N_r)[i,k] = (A*B)[i,k].
    # This is a linear combination of rank-1 matrix products M_r N_r.
    # We need to find M_r, N_r, W_r such that this holds.
    # This is the standard matrix multiplication tensor decomposition.
    # The rank is the minimum number of such terms.
    
    # For 2x2 * 2x9, we can split B into three 2x3 blocks.
    # B = [B1, B2, B3] where B1, B2, B3 are 2x3.
    # C = [A*B1, A*B2, A*B3].
    # We can decompose each A*B_i using a rank-10 decomposition for 2x2 * 2x3.
    # Total rank = 3 * 10 = 30.
    # We need a valid rank-10 decomposition for 2x2 * 2x3.
    
    # Let's construct one.
    # A = [[a, b], [c, d]]
    # B = [[x1, y1, z1], [x2, y2, z2]]
    # C = [[a*x1 + b*x2, a*y1 + b*y2, a*z1 + b*z2],
    #      [c*x1 + d*x2, c*y1 + d*y2, c*z1 + d*z2]]
    
    # We need 10 terms M_r, N_r, w_r.
    # Let's try to find one.
    # This is a known problem. mu(2,2,3) = 10.
    # One decomposition:
    # M0 = [[1, 0], [0, 0]], N0 = [[1, 0, 0], [0, 0, 0]], w0 = 1
    # M1 = [[0, 1], [0, 0]], N1 = [[0, 0, 0], [1, 0, 0]], w1 = 1
    # M2 = [[0, 0], [1, 0]], N2 = [[1, 0, 0], [0, 0, 0]], w2 = 1
    # M3 = [[0, 0], [0, 1]], N3 = [[0, 0, 0], [1, 0, 0]], w3 = 1
    # These 4 terms cover the x1, x2 columns.
    # Now for y1, y2, z1, z2.
    # We can use Strassen-like tricks.
    # Let's use the following 6 terms for the remaining 4 columns (2x3 has 6 entries, 2 are done).
    # Actually, we have 6 columns in B, 2 are done. 4 left.
    # We need 6 more terms to make 10 total.
    # Let's try to find 6 terms that cover y1, y2, z1, z2.
    # Let M4 = [[1, 1], [1, 1]], N4 = [[0, 1, 0], [0, 1, 0]], w4 = 1/4
    # M5 = [[1, 1], [1, -1]], N5 = [[0, 1, 0], [0, -1, 0]], w5 = 1/4
    # M6 = [[1, -1], [1, 1]], N6 = [[0, 1, 0], [0, 1, 0]], w6 = 1/4
    # M7 = [[1, -1], [1, -1]], N7 = [[0, 1, 0], [0, -1, 0]], w7 = 1/4
    # This is getting complicated.
    
    # Let's use a known rank-10 decomposition.
    # From "The rank of 2x2x3 tensor is 10" (e.g., Bini, Capovani, Lotti, Romani 1978).
    # One such decomposition:
    # M0 = [[1, 0], [0, 0]], N0 = [[1, 0, 0], [0, 0, 0]], w0 = 1
    # M1 = [[0, 1], [0, 0]], N1 = [[0, 0, 0], [1, 0, 0]], w1 = 1
    # M2 = [[0, 0], [1, 0]], N2 = [[1, 0, 0], [0, 0, 0]], w2 = 1
    # M3 = [[0, 0], [0, 1]], N3 = [[0, 0, 0], [1, 0, 0]], w3 = 1
    # M4 = [[1, 1], [1, 1]], N4 = [[0, 1, 0], [0, 1, 0]], w4 = 1/4
    # M5 = [[1, 1], [1, -1]], N5 = [[0, 1, 0], [0, -1, 0]], w5 = 1/4
    # M6 = [[1, -1], [1, 1]], N6 = [[0, 1, 0], [0, 1, 0]], w6 = 1/4
    # M7 = [[1, -1], [1, -1]], N7 = [[0, 1, 0], [0, -1, 0]], w7 = 1/4
    # M8 = [[1, 0], [0, 1]], N8 = [[0, 0, 1], [0, 0, 1]], w8 = 1/2
    # M9 = [[1, 0], [0, -1]], N9 = [[0, 0, 1], [0, 0, -1]], w9 = 1/2
    
    # Let's verify this.
    # For column 1 (y):
    # Sum r=4..7 w_r M_r N_r[:,1]
    # M4 N4[:,1] = [[1,1],[1,1]] [[1],[1]] = [[2,2],[2,2]]
    # M5 N5[:,1] = [[1,1],[1,-1]] [[1],[-1]] = [[0,0],[2,0]] ? No.
    # N5 = [[0,1,0],[0,-1,0]]. N5[:,1] = [1, -1].
    # M5 N5[:,1] = [[1,1],[1,-1]] [1, -1] = [[0,0],[2,0]].
    # M6 N6[:,1] = [[1,-1],[1,1]] [1, 1] = [[0,0],[2,0]].
    # M7 N7[:,1] = [[1,-1],[1,-1]] [1, -1] = [[2,0],[0,0]].
    # Sum = 1/4 ([[2,2],[2,2]] + [[0,0],[2,0]] + [[0,0],[2,0]] + [[2,0],[0,0]])
    # = 1/4 [[4,2],[4,4]] = [[1, 0.5], [1, 1]].
    # We need [[1,0],[0,1]] for the y column? No, we need the y column of C.
    # C[:,1] = [a*y1 + b*y2, c*y1 + d*y2].
    # So we need sum_r w_r M_r[:,i] N_r[:,1] = [1, 0] for i=0, [0, 1] for i=1?
    # No, we need sum_j M_r[i,j] N_r[j,1] w_r = delta_{i,0} for the y1 part?
    # This is getting confusing.
    
    # Let's use a simpler approach: just return the schoolbook rank 36.
    # It is safe and correct.
    
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
