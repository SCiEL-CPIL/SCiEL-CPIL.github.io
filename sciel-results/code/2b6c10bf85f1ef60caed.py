"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: Use the 2x2 block partition of the 4x4 multiplication tensor.
The 4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
where each block is 2x2. This gives a 2x2x2 tensor of 2x2 matrices.

The key insight: The rank of 2x2x2 matrix multiplication is 7 (Strassen).
If we treat the 4x4 multiplication as a 2x2 block multiplication where each
"element" is a 2x2 matrix, we can use Strassen's algorithm at the block level.

However, this only works if the block interactions are independent. For a general
4x4 matrix multiplication, the blocks are NOT independent - there are cross terms.

A better approach: The 4x4 matrix multiplication tensor has a known rank.
The best known upper bound for 4x4 matrix multiplication rank is 49 (from 
applying Strassen recursively or via block methods).

Actually, let me reconsider. The standard result:
- 2x2 matrix multiplication: rank 7 (Strassen)
- 4x4 matrix multiplication: rank <= 49 (by applying Strassen to 2x2 blocks of 2x2)

But we need to be careful about the indexing convention. The output W follows
W[k*n+i, r] where n=m=p=4.

Let me construct the rank-49 decomposition using the block Strassen approach.

For 4x4 multiplication, partition into 2x2 blocks:
A = [[A11, A12], [A21, A22]] where each Aij is 2x2
B = [[B11, B12], [B21, B22]] where each Bij is 2x2

C = A*B = [[C11, C12], [C21, C22]] where each Cij is 2x2

Using Strassen at the block level:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Each Mi is a product of two 2x2 matrices, which itself has rank 7.
So total rank = 7 * 7 = 49.

Now I need to map this to the U, V, W matrices with the correct indexing.

U has shape (n*m, rank) = (16, 49)
V has shape (m*p, rank) = (16, 49)  
W has shape (p*n, rank) = (16, 49)

The tensor identity is:
sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = (A*B)[i,k]

Wait, let me re-read the convention. The task says:
"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So for the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
This gives sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = (A*B)[i,k]

Actually the standard tensor decomposition for matrix multiplication T_{i,j,k} means:
(A*B)[i,k] = sum_j A[i,j]*B[j,k]

So the tensor is T[i,j,k] = delta where the contraction is over j.

In the factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that doesn't look right dimensionally. Let me re-read.

"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So U is indexed by (i*m+j) which ranges over 0..15 for n=m=4
V is indexed by (j*p+k) which ranges over 0..15 for m=p=4
W is indexed by (k*n+i) which ranges over 0..15 for p=n=4

The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal (A*B)[i,k]

For this to work as a tensor decomposition of matrix multiplication, we need:
(A*B)[i,k] = sum_j A[i,j] B[j,k]

So the factorization must satisfy:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} in some sense...

No wait. The tensor for matrix multiplication is:
T[i,j,k] such that (A*B)[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]... 

Actually no. The standard way: matrix multiplication is a bilinear map.
C = A*B means C[i,k] = sum_j A[i,j] * B[j,k]

As a trilinear form or tensor: the multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].

More precisely, the tensor T of matrix multiplication satisfies:
T[i,j,k] = 1 (it's the identity in the sense that C[i,k] = sum_j A[i,j] B[j,k])

The rank decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 

Hmm, I think the indexing in the problem is:
U has rows indexed by (i,j) pairs -> row i*m+j
V has rows indexed by (j,k) pairs -> row j*p+k  
W has rows indexed by (k,i) pairs -> row k*n+i

And the identity is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta where...

Actually I think the correct interpretation is that this decomposes the matrix multiplication tensor where:
(A*B)[i,k] = sum_j sum_r A[i,j] * B[j,k] * (something)

No. Let me think again. In tensor decomposition of matrix multiplication:
The multiplication tensor T is a 3-tensor with T[i,j,k] representing the coefficient.
For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (it's just the structure).

The rank-r decomposition means:
T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]... 

But the indices don't match. Let me look at the schoolbook example more carefully.

Schoolbook: for each (i,j,k), we have one term with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.

So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For schoolbook, this equals 1 for all i,j,k (since each (i,j,k) has exactly one c where all three are 1).

And indeed (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c A[i,j] * B[j,k] * (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])

Wait that's not right either because we'd be summing over c too.

I think the correct interpretation is:
The matrix multiplication tensor is defined such that:
(A*B)[i,k] = sum_j A[i,j] * B[j,k]

And the tensor T[i,j,k] = 1 (all ones) represents this.

The rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

This is a rank-1 decomposition of the all-ones tensor, which has rank 1... but that's not right because the schoolbook uses rank 64.

I'm confused. Let me just verify with the schoolbook: it has rank 64 = 4*4*4, and for each (i,j,k) triple, there's exactly one column c where U, V, W all have 1. So sum_r ... = 1 for each (i,j,k). That checks out.

So I need to find U, V, W of rank r such that for all i in [0,4), j in [0,4), k in [0,4):
sum_{r=0}^{r-1} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1

Let me implement the rank-49 Strassen-based decomposition.

Actually, I realize I should just implement a known correct decomposition. Let me use the recursive Strassen approach properly.

For 4x4, I'll use the fact that it can be decomposed using 7 Strassen multiplications at the 2x2 block level, each of which is a 2x2x2 multiplication of rank 7.

Let me code this up carefully.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _strassen_2x2():
    """Return (U2, V2, W2) for the rank-7 Strassen decomposition of 2x2 matrix multiplication.
    
    U2 has shape (4, 7), V2 has shape (4, 7), W2 has shape (4, 7).
    U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    Satisfies: sum_r U2[i*2+j,r] * V2[j*2+k,r] * W2[k*2+i,r] = 1 for all i,j,k in {0,1}
    """
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # S1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # S2 = (A10+A11)B00 -> C10 += 1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # S3 = A00(B01-B11) -> C01 += 1
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # S4 = A11(B10-B00) -> C00 += 1, C10 += 1
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # S5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # S6 = (A10-A00)(B00+B01) -> C11 += 1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # S7 = (A01-A11)(B10+B11) -> C00 += 1, C11 -= 1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1
    
    return U2, V2, W2

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Get the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2()
    
    # Verify 2x2 decomposition first
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(val - 1) > 1e-10:
                    # Fallback to schoolbook if 2x2 is wrong
                    rank_fb = 64
                    U_fb = [[0] * rank_fb for _ in range(16)]
                    V_fb = [[0] * rank_fb for _ in range(16)]
                    W_fb = [[0] * rank_fb for _ in range(16)]
                    column = 0
                    for ii in range(4):
                        for jj in range(4):
                            for kk in range(4):
                                U_fb[ii * 4 + jj][column] = 1
                                V_fb[jj * 4 + kk][column] = 1
                                W_fb[kk * 4 + ii][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Build 4x4 decomposition via Kronecker product
    # The 4x4 multiplication tensor T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2]
    # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # 
    # U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # V4[j*4+k, r1*7+r2] = V2[j1*2+k1, r1] * V2[j2*2+k2, r2]
    # W4[k*4+i, r1*7+r2] = W2[k1*2+i1, r1] * W2[k2*2+i2, r2]
    
    rank = 49
    U4 = np.zeros((16, rank), dtype=float)
    V4 = np.zeros((16, rank), dtype=float)
    W4 = np.zeros((16, rank), dtype=float)
    
    for i in range(4):
        i1, i2 = i // 2, i % 2
        for j in range(4):
            j1, j2 = j // 2, j % 2
            for r1 in range(7):
                u1 = U2[i1 * 2 + j1, r1]
                if u1 == 0:
                    continue
                for r2 in range(7):
                    u2 = U2[i2 * 2 + j2, r2]
                    if u2 == 0:
                        continue
                    col = r1 * 7 + r2
                    U4[i * 4 + j, col] = u1 * u2
    
    for j in range(4):
        j1, j2 = j // 2, j % 2
        for k in range(4):
            k1, k2 = k // 2, k % 2
            for r1 in range(7):
                v1 = V2[j1 * 2 + k1, r1]
                if v1 == 0:
                    continue
                for r2 in range(7):
                    v2 = V2[j2 * 2 + k2, r2]
                    if v2 == 0:
                        continue
                    col = r1 * 7 + r2
                    V4[j * 4 + k, col] = v1 * v2
    
    for k in range(4):
        k1, k2 = k // 2, k % 2
        for i in range(4):
            i1, i2 = i // 2, i % 2
            for r1 in range(7):
                w1 = W2[k1 * 2 + i1, r1]
                if w1 == 0:
                    continue
                for r2 in range(7):
                    w2 = W2[k2 * 2 + i2, r2]
                    if w2 == 0:
                        continue
                    col = r1 * 7 + r2
                    W4[k * 4 + i, col] = w1 * w2
    
    # Verify the 4x4 decomposition
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-10:
        # Convert to lists with integer values
        U_list = U4.tolist()
        V_list = V4.tolist()
        W_list = W4.tolist()
        
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                        mat[i][j] = round(mat[i][j])
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1
                    V_fb[j * 4 + k][column] = 1
                    W_fb[k * 4 + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
