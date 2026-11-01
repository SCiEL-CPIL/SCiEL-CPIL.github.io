import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # The theoretical minimum rank for multiplying an (n x m) matrix by an (m x p) matrix
    # using the best known algorithms (like Strassen for 2x2) combined with linearity in the other dimension
    # is rank(2x2 mult) * p = 7 * 16 = 112? 
    # Wait, let's re-evaluate the "rank" of the tensor.
    # The tensor for A (n x m) * B (m x p) -> C (n x p) has indices i, k (output), j (input A), l (input B)?
    # No, standard matrix mult: C[i, k] = sum_j A[i, j] B[j, k].
    # Here A is 2x2, B is 2x16.
    # C is 2x16.
    # C[i, k] = sum_{j=0}^{1} A[i, j] B[j, k].
    
    # The tensor T has indices (i, k, j) where i in 0..1, k in 0..15, j in 0..1.
    # T[i, k, j] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # Actually, the decomposition is:
    # C[i, k] = sum_r U[i, r] V[r, k] ... wait, the interface is:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]
    # W: (p*n, rank) -> W[k*n + i, r]
    
    # The contraction is:
    # C[i, k] = sum_{j, r} A[i, j] B[j, k] * (something involving U, V, W)?
    # No, the standard tensor decomposition for matrix multiplication A*B=C is:
    # C[i, k] = sum_{j, r} A[i, j] B[j, k] M_{i, k, j, r}?
    # Let's look at the shapes:
    # U has rows indexed by (i, j) of A.
    # V has rows indexed by (j, k) of B.
    # W has rows indexed by (k, i) of C.
    
    # The identity is:
    # sum_r U[i, j, r] V[j, k, r] W[k, i, r] ? No.
    # Usually, the tensor T_{i, k, j} for C[i, k] = sum_j A[i, j] B[j, k] is decomposed as:
    # T_{i, k, j} = sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} ?
    # Let's check dimensions.
    # If we sum over r, we get a term for each (i, j, k).
    # But C[i, k] depends on A[i, j] and B[j, k].
    # The standard decomposition is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This can be written as a sum of rank-1 tensors in the space of (A, B, C) tensors?
    # The tensor M has components M[i, j, k] which is 1 if the product A[i, j] * B[j, k] contributes to C[i, k].
    # Actually, the tensor is defined such that:
    # <C, A, B> = sum_{i, k} C[i, k] (sum_j A[i, j] B[j, k])
    # = sum_{i, j, k} A[i, j] B[j, k] C[i, k]
    # The tensor T has indices i, j, k. T[i, j, k] = 1 if C[i, k] receives contribution from A[i, j] and B[j, k].
    # In our case, T[i, j, k] = 1 for all i, j, k such that the index mapping is valid.
    # Specifically, for fixed i, j, k, the term is A[i, j] B[j, k]. This contributes to C[i, k].
    # So the tensor T is a 3-way tensor of size (n*m) x (m*p) x (p*n)?
    # No, the indices are:
    # A indices: i, j (size n*m)
    # B indices: j, k (size m*p)
    # C indices: k, i (size p*n)
    
    # The decomposition is:
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # where a = i*m + j
    #       b = j*p + k
    #       c = k*n + i
    
    # For the standard schoolbook algorithm:
    # Each term A[i, j] B[j, k] is a rank-1 contribution to the tensor?
    # No, the tensor T is defined by the multiplication algorithm.
    # T[i, j, k] = 1 if the algorithm computes C[i, k] using A[i, j] and B[j, k].
    # In standard multiplication, for each i, j, k, there is exactly one term.
    # But wait, the tensor T is fixed by the problem definition (matrix multiplication).
    # T[i, j, k] is 1 if the (i, j)-th element of A and (j, k)-th element of B multiply to contribute to C[i, k].
    # In standard matrix multiplication, C[i, k] = sum_j A[i, j] B[j, k].
    # So for a specific i, j, k, the coefficient is 1.
    # The tensor T has non-zero entries at (a, b, c) where a=(i, j), b=(j, k), c=(k, i).
    # Note that j is shared between a and b.
    # So T is non-zero only if the j index in a matches the j index in b.
    # And the i index in a matches the i index in c (reversed in c's index k*n+i).
    # And the k index in b matches the k index in c.
    
    # The rank of this tensor is what we want to minimize.
    # The schoolbook rank is n*m*p = 2*2*16 = 64.
    # Each rank-1 term corresponds to a specific (i, j, k) triple.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    
    # Can we do better?
    # For 2x2 matrix multiplication, Strassen's algorithm reduces the rank from 8 to 7.
    # The 2x2 multiplication tensor has rank 7.
    # Our problem is multiplying a 2x2 matrix A by a 2x16 matrix B.
    # This is equivalent to multiplying A by 8 independent 2x2 matrices (columns of B grouped by 2)?
    # No, B is 2x16. Let's split B into 8 blocks of 2x2 matrices, B_1, ..., B_8.
    # Then C = A * B = [A*B_1, A*B_2, ..., A*B_8].
    # Since matrix multiplication is linear, C_block_q = A * B_q.
    # The tensor for the whole operation is the direct sum of 8 tensors, each being the 2x2 multiplication tensor.
    # The rank of a direct sum of tensors is the sum of their ranks.
    # So the rank should be 8 * rank(2x2 mult tensor) = 8 * 7 = 56.
    
    # Let's verify this.
    # The 2x2 multiplication tensor T_2 has rank 7.
    # It can be decomposed into 7 rank-1 terms.
    # Let the decomposition be T_2[i, j, k] = sum_{r=0}^{6} U_2[i, j, r] V_2[j, k, r] W_2[k, i, r].
    # For our problem, we have 8 such blocks.
    # Let q be the block index (0 to 7).
    # B's columns are k = 0..15. Let k = 2*q + k', where k' is 0 or 1.
    # Wait, B is 2x16. Rows j=0,1. Columns k=0..15.
    # We can view B as 8 matrices of size 2x2.
    # B_q has elements B[j, 2*q + k'] for j=0,1 and k'=0,1.
    # Then C[i, 2*q + k'] = sum_{j=0}^{1} A[i, j] B[j, 2*q + k'].
    # This is exactly the 2x2 multiplication for block q.
    # So the total tensor T is the direct sum of 8 copies of T_2.
    # The rank is 8 * 7 = 56.
    
    # We need to construct U, V, W of shapes (4, 56), (32, 56), (32, 56).
    # Wait, n=2, m=2, p=16.
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    
    # Strassen's 2x2 decomposition:
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # Let's map this to the tensor indices.
    # A indices: i, j. i=0,1; j=0,1.
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1].
    # B indices: j, k. j=0,1; k=0,1.
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1].
    # C indices: k, i. k=0,1; i=0,1.
    # C[0,0] = C11, C[0,1] = C12, C[1,0] = C21, C[1,1] = C22.
    
    # For each of the 7 products p_r, we have a rank-1 term in the 2x2 tensor.
    # The term p_r is a product of a linear form in A and a linear form in B.
    # p_r = (sum_{i,j} U_r[i,j] A[i,j]) * (sum_{j,k} V_r[j,k] B[j,k])
    # And it contributes to C as:
    # C[k,i] += W_r[k,i] * p_r
    
    # Let's determine U_r, V_r, W_r for r=0..6.
    
    # r=0: p1 = (a+d)(e+h)
    # U_0: a=1, d=1. So U_0[0,0]=1, U_0[1,1]=1.
    # V_0: e=1, h=1. So V_0[0,0]=1, V_0[1,1]=1.
    # W_0: C11 += 1, C22 += 1.
    # C11 is C[0,0] -> index k=0, i=0 -> W_0[0,0]=1.
    # C22 is C[1,1] -> index k=1, i=1 -> W_0[1,1]=1.
    
    # r=1: p2 = (c+d)e
    # U_1: c=1, d=1. U_1[1,0]=1, U_1[1,1]=1.
    # V_1: e=1. V_1[0,0]=1.
    # W_1: C21 += 1, C22 -= 1.
    # C21 is C[1,0] -> k=1, i=0 -> W_1[1,0]=1.
    # C22 is C[1,1] -> k=1, i=1 -> W_1[1,1]=-1.
    
    # r=2: p3 = a(f-h)
    # U_2: a=1. U_2[0,0]=1.
    # V_2: f=1, h=-1. V_2[0,1]=1, V_2[1,1]=-1.
    # W_2: C12 += 1, C22 += 1.
    # C12 is C[0,1] -> k=0, i=1 -> W_2[0,1]=1.
    # C22 is C[1,1] -> k=1, i=1 -> W_2[1,1]=1.
    
    # r=3: p4 = d(g-e)
    # U_3: d=1. U_3[1,1]=1.
    # V_3: g=1, e=-1. V_3[1,0]=1, V_3[0,0]=-1.
    # W_3: C11 += 1, C21 += 1.
    # C11 is C[0,0] -> k=0, i=0 -> W_3[0,0]=1.
    # C21 is C[1,0] -> k=1, i=0 -> W_3[1,0]=1.
    
    # r=4: p5 = (a+b)h
    # U_4: a=1, b=1. U_4[0,0]=1, U_4[0,1]=1.
    # V_4: h=1. V_4[1,1]=1.
    # W_4: C11 -= 1, C12 += 1.
    # C11 is C[0,0] -> k=0, i=0 -> W_4[0,0]=-1.
    # C12 is C[0,1] -> k=0, i=1 -> W_4[0,1]=1.
    
    # r=5: p6 = (c-a)(e+f)
    # U_5: c=1, a=-1. U_5[1,0]=1, U_5[0,0]=-1.
    # V_5: e=1, f=1. V_5[0,0]=1, V_5[0,1]=1.
    # W_5: C22 += 1.
    # C22 is C[1,1] -> k=1, i=1 -> W_5[1,1]=1.
    
    # r=6: p7 = (b-d)(g+h)
    # U_6: b=1, d=-1. U_6[0,1]=1, U_6[1,1]=-1.
    # V_6: g=1, h=1. V_6[1,0]=1, V_6[1,1]=1.
    # W_6: C11 += 1.
    # C11 is C[0,0] -> k=0, i=0 -> W_6[0,0]=1.
    
    # Now, we have 8 blocks.
    # For each block q (0..7), we have a copy of this decomposition.
    # The total rank is 8 * 7 = 56.
    # The indices for U, V, W need to be mapped to the global indices.
    
    # U global index: i*m + j. (i in 0..1, j in 0..1)
    # V global index: j*p + k. (j in 0..1, k in 0..15)
    # W global index: k*n + i. (k in 0..15, i in 0..1)
    
    # For block q, the local k' is 0 or 1.
    # Global k = 2*q + k'.
    
    # Let's construct the matrices.
    rank = 56
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # Strassen components
    # U_r[i, j]
    U_strassen = [
        [[1, 0], [0, 1]],  # r=0
        [[0, 0], [1, 1]],  # r=1
        [[1, 0], [0, 0]],  # r=2
        [[0, 0], [0, 1]],  # r=3
        [[1, 1], [0, 0]],  # r=4
        [[-1, 0], [1, 0]], # r=5
        [[0, 1], [0, -1]]  # r=6
    ]
    
    # V_r[j, k]
    V_strassen = [
        [[1, 0], [0, 1]],  # r=0
        [[1, 0], [0, 0]],  # r=1
        [[0, 1], [0, -1]], # r=2
        [[-1, 0], [1, 0]], # r=3
        [[0, 0], [0, 1]],  # r=4
        [[1, 1], [0, 0]],  # r=5
        [[0, 0], [1, 1]]   # r=6
    ]
    
    # W_r[k, i]
    W_strassen = [
        [[1, 0], [0, 1]],  # r=0
        [[0, 0], [1, -1]], # r=1
        [[0, 1], [0, 1]],  # r=2
        [[1, 0], [1, 0]],  # r=3
        [[-1, 1], [0, 0]], # r=4
        [[0, 0], [0, 1]],  # r=5
        [[1, 0], [0, 0]]   # r=6
    ]
    
    # Fill U, V, W
    for q in range(8):
        for r in range(7):
            col = q * 7 + r
            # U
            for i in range(2):
                for j in range(2):
                    val = U_strassen[r][i][j]
                    if val != 0:
                        U[i * 2 + j][col] += val
                        
            # V
            for j in range(2):
                for k_local in range(2):
                    val = V_strassen[r][j][k_local]
                    if val != 0:
                        k_global = 2 * q + k_local
                        V[j * 16 + k_global][col] += val
                        
            # W
            for k_local in range(2):
                for i in range(2):
                    val = W_strassen[r][k_local][i]
                    if val != 0:
                        k_global = 2 * q + k_local
                        W[k_global * 2 + i][col] += val
                        
    # Verification (optional, but good for debugging)
    # Check if U @ V^T @ W^T reproduces the tensor?
    # Actually, the tensor is T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    # We can verify a few entries.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res)
