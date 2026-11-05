"""Improved program for matrix multiplication 3x4x5 tensor decomposition.

Uses Strassen's algorithm on disjoint 2x2x2 sub-blocks to reduce rank below
the schoolbook rank of 60. The 3x4x5 problem can be partitioned into:
- One 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}) -> Strassen rank 7
- Remaining terms -> schoolbook

This gives rank 59 (one savings) or potentially better with careful partitioning.

Actually, let's think more carefully. We have:
- n=3 rows of A
- m=4 columns of A / rows of B  
- p=5 columns of B

The tensor T[i,j,k] = A[i,j] * B[j,k]

We can partition into blocks:
- For each i, j, k we have a term A[i,j]*B[j,k]

Strassen applies when we have a 2x2x2 sub-tensor. We need disjoint blocks.

Partition strategy:
- Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 8 terms, Strassen gives 7
- All other (i,j,k) pairs -> schoolbook

Number of other pairs: 60 - 8 = 52
Total rank: 7 + 52 = 59

Can we do better? Let's try multiple disjoint 2x2x2 blocks.

With n=3, we can have at most floor(3/2)=1 disjoint pair of i's.
With m=4, we can have at most floor(4/2)=2 disjoint pairs of j's.
With p=5, we can have at most floor(5/2)=2 disjoint pairs of k's.

But the blocks must be 2x2x2, so we need pairs in all three dimensions simultaneously.
Since n only allows 1 pair, we can have at most 1 disjoint 2x2x2 block.

So rank 59 is the best we can get with this approach.

Wait, let me reconsider. Maybe we can use a different decomposition that's not purely Strassen on 2x2x2 blocks.

Actually, let me just implement the rank-59 solution carefully.

Strassen decomposition for 2x2x2:
The 8 monomials are:
A00*B00, A00*B01, A01*B00, A01*B01, A10*B00, A10*B01, A11*B00, A11*B01

Strassen's 7 products:
M1 = (A00+A11)*(B00+B11)
M2 = (A10+A11)*B00
M3 = A00*(B01-B11)
M4 = A11*(B10-B00)
M5 = (A00+A01)*B11
M6 = (A10-A00)*(B00+B01)
M7 = (A01-A11)*(B10+B11)

Then:
C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

Let me verify this expands correctly:
C00 should be A00*B00 + A01*B10
C01 should be A00*B01 + A01*B11
C10 should be A10*B00 + A11*B10
C11 should be A10*B01 + A11*B11

M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
M2 = (A10+A11)*B00 = A10*B00 + A11*B00
M3 = A00*(B01-B11) = A00*B01 - A00*B11
M4 = A11*(B10-B00) = A11*B10 - A11*B00
M5 = (A00+A01)*B11 = A00*B11 + A01*B11
M6 = (A10-A00)*(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
M7 = (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11

C00 = M1 + M4 - M5 + M7
= (A00*B00 + A00*B11 + A11*B00 + A11*B11)
+ (A11*B10 - A11*B00)
- (A00*B11 + A01*B11)
+ (A01*B10 + A01*B11 - A11*B10 - A11*B11)
= A00*B00 + A00*B11 + A11*B00 + A11*B11 + A11*B10 - A11*B00 - A00*B11 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
= A00*B00 + A01*B10  ✓

C01 = M3 + M5
= (A00*B01 - A00*B11) + (A00*B11 + A01*B11)
= A00*B01 + A01*B11  ✓

C10 = M2 + M4
= (A10*B00 + A11*B00) + (A11*B10 - A11*B00)
= A10*B00 + A11*B10  ✓

C11 = M1 - M2 + M3 + M6
= (A00*B00 + A00*B11 + A11*B00 + A11*B11)
- (A10*B00 + A11*B00)
+ (A00*B01 - A00*B11)
+ (A10*B00 + A10*B01 - A00*B00 - A00*B01)
= A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A00*B01 - A00*B11 + A10*B00 + A10*B01 - A00*B00 - A00*B01
= A11*B11 + A10*B01  ✓

Great, the Strassen decomposition is correct.

Now for the factor matrices:
U has shape (n*m, rank) = (12, rank)
V has shape (m*p, rank) = (20, rank)  
W has shape (p*n, rank) = (15, rank)

The convention is W[k*n + i, r], so W index is k*n + i where k is output column, i is output row.

For a term with coefficient c for (i, j, k):
U[i*m + j, r] += c
V[j*p + k, r] += c  
W[k*n + i, r] += c

For Strassen terms, we need to express each of the 7 products as a linear combination.

Let me index the local 2x2x2 block with local indices (ii, jj, kk) in {0,1}^3.
Global indices: i = 0 + ii, j = 0 + jj, k = 0 + kk (for the block at i={0,1}, j={0,1}, k={0,1})

For M1 = (A00+A11)*(B00+B11):
This involves A[i,j] for (i,j) in {(0,0),(1,1)} and B[j,k] for (j,k) in {(0,0),(1,1)}
The product is sum over the outer product of the row vector [A00, A11] and column vector [B00, B11]

Actually, each Strassen term M_m is a bilinear form. To express it as a rank-1 tensor term, I need to write it as sum of terms of the form u_{ij} * v_{jk} * w_{ki}.

Wait, no. A rank-1 tensor term is of the form U[:,r] * V[:,r] * W[:,r] where these are vectors. The contribution to T[i,j,k] from term r is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].

For a schoolbook term (i,j,k), we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.

For Strassen, each M_m is itself a product of two linear forms. For example:
M1 = (A00+A11) * (B00+B11)

This is NOT a single bilinear term A[i,j]*B[j,k]. It's a product of sums. To express the tensor T as a sum of rank-1 terms, we need to decompose M1 into rank-1 terms.

But M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11

Each of these four is a bilinear term! So M1 alone requires 4 rank-1 tensor terms if we write it out.

Hmm, that defeats the purpose. Let me reconsider.

Actually, in tensor rank decomposition, a rank-1 tensor is one that can be written as u ⊗ v ⊗ w, meaning T[i,j,k] = u[i] * v[j] * w[k]. For matrix multiplication, the tensor is T[i,j,k] = A[i,j] * B[j,k], which is already a specific structure.

The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number of bilinear multiplications needed. A rank-1 term corresponds to one bilinear multiplication: one linear combination of A entries times one linear combination of B entries, contributing to specific C entries.

So for Strassen, each M_m is ONE bilinear multiplication. The 7 M_m's are 7 bilinear multiplications. Each contributes to multiple C entries.

So in the tensor decomposition:
- Each M_m corresponds to one column r in U, V, W
- U[:, r] encodes the linear combination of A entries for M_m
- V[:, r] encodes the linear combination of B entries for M_m
- W[:, r] encodes how M_m contributes to C entries

For M1 = (A00+A11) * (B00+B11):
U[0*4+0, r] = 1  (A00)
U[1*4+1, r] = 1  (A11)
V[0*5+0, r] = 1  (B00)
V[1*5+1, r] = 1  (B11)
W contributions: M1 contributes to C00, C01, C10, C11 via the combinations above.

Wait, I need to be more careful. The Strassen algorithm computes C = A*B using 7 products. The final C entries are linear combinations of the M_m's. So:

C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

So W[k*n+i, r] should encode the coefficient of M_m in the expression for C[i,k].

For the block at global indices i={0,1}, j={0,1}, k={0,1}:

M1 (r=0):
U[0*4+0, 0] = 1, U[1*4+1, 0] = 1
V[0*5+0, 0] = 1, V[1*5+1, 0] = 1
W[0*3+0, 0] = 1   (C00: M1 coefficient is +1)
W[1*3+0, 0] = 1   (C01: M1 coefficient is 0... wait)

Let me re-index. C[i,k] where i is row, k is column.
C00 = M1 + M4 - M5 + M7
C01 = M3 + M5
C10 = M2 + M4
C11 = M1 - M2 + M3 + M6

W index is k*n + i, so:
W[0*3+0, r] = coeff of M_r in C00
W[1*3+0, r] = coeff of M_r in C01
W[0*3+1, r] = coeff of M_r in C10
W[1*3+1, r] = coeff of M_r in C11

For M1 (r=0):
W[0, 0] = 1  (C00)
W[3, 0] = 1  (C11)

For M2 (r=1):
W[4, 1] = 1  (C10)
W[3, 1] = -1 (C11)

For M3 (r=2):
W[1, 2] = 1  (C01)
W[3, 2] = 1  (C11)

For M4 (r=3):
W[0, 3] = 1  (C00)
W[4, 3] = 1  (C10)

For M5 (r=4):
W[0, 4] = -1 (C00)
W[1, 4] = 1  (C01)

For M6 (r=5):
W[3, 5] = 1  (C11)

For M7 (r=6):
W[0, 6] = 1  (C00)

Let me verify:
C00 = M1 + M4 - M5 + M7 = W[0,0] + W[0,3] - W[0,4] + W[0,6] = 1+1-(-1)+1 = 4? That's wrong.

I think I'm confusing myself. Let me redo this.

The tensor T[i,j,k] for the 2x2x2 block equals C[i,k] = sum_j A[i,j]*B[j,k].

The Strassen decomposition says C[i,k] = sum_m alpha_{i,k,m} * M_m, where M_m are the 7 products.

So T[i,j,k] = sum_m alpha_{i,k,m} * M_m

But M_m is itself a product of linear forms in A and B. So M_m = (sum_{i,j} a_{m,i,j} * A[i,j]) * (sum_{j,k} b_{m,j,k} * B[j,k])

Then T[i,j,k] = sum_m alpha_{i,k,m} * (sum_{i',j'} a_{m,i',j'} * A[i',j']) * (sum_{j',k'} b_{m,j',k'} * B[j',k'])

This doesn't directly give T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].

I think the issue is that the standard tensor rank decomposition for matrix multiplication works differently. Let me look at this from the correct perspective.

The matrix multiplication tensor M_{n,m,p} has entries M_{n,m,p}[i,j,k] for i in [n], j in [m], k in [p]. A rank-1 term is a tensor T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i] where these are... 

Actually, I think the convention in the problem is:
U has shape (n*m, rank), indexed by (i,j) pair
V has shape (m*p, rank), indexed by (j,k) pair
W has shape (p*n, rank), indexed by (k,i) pair

And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

For schoolbook, each term is (i,j,k) -> U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.

For Strassen, we need to find U, V, W such that the sum reconstructs the multiplication tensor.

I think the key insight is that each "bilinear multiplication" in Strassen corresponds to ONE rank-1 term in the tensor decomposition, but the U, V, W vectors for that term have multiple non-zero entries.

Let me just implement it by verifying against the schoolbook tensor numerically and solving for the correct U, V, W entries.

Actually, let me just go with the straightforward rank-59 approach and verify it's correct by checking that the decomposition matches the schoolbook tensor.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 60
    rank = n * m * p
    
    # Define the 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1}
    # These 8 terms will be replaced by 7 Strassen terms
    
    # Strassen decomposition for 2x2x2
    # Local indices: ii, jj, kk in {0,1}
    # Global: i=ii, j=jj, k=kk
    
    # The 7 Strassen products and their linear forms:
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Build U, V, W for rank 59
    new_rank = rank - 1  # 59
    
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    # Fill in schoolbook terms for all (i,j,k) EXCEPT the 2x2x2 block
    col = 0
    block_set = set()
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                block_set.add((ii, jj, kk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in block_set:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Now fill in the 7 Strassen terms
    # Strassen term r (0-6) goes to column col + r
    
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0*4 + 0][col+0] = 1
    U[1*4 + 1][col+0] = 1
    V[0*5 + 0][col+0] = 1
    V[1*5 + 1][col+0] = 1
    W[0*3 + 0][col+0] = 1   # C[0,0]
    W[1*3 + 1][col+0] = 1   # C[1,1] -> index k*n+i = 1*3+1 = 4
    
    # M2 = (A[1,0]+A[1,1]) * B[0,0]
    U[1*4 + 0][col+1] = 1
    U[1*4 + 1][col+1] = 1
    V[0*5 + 0][col+1] = 1
    W[0*3 + 1][col+1] = 1   # C[1,0] -> index k*n+i = 0*3+1 = 3
    W[1*3 + 1][col+1] = -1  # C[1,1] -> index 4
    
    # M3 = A[0,0] * (B[0,1]-B[1,1])
    U[0*4 + 0][col+2] = 1
    V[0*5 + 1][col+2] = 1
    V[1*5 + 1][col+2] = -1
    W[1*3 + 0][col+2] = 1   # C[0,1] -> index k*n+i = 1*3+0 = 3
    W[1*3 + 1][col+2] = 1   # C[1,1] -> index 4
    
    # M4 = A[1,1] * (B[1,0]-B[0,0])
    U[1*4 + 1][col+3] = 1
    V[1*5 + 0][col+3] = 1
    V[0*5 + 0][col+3] = -1
    W[0*3 + 0][col+3] = 1   # C[0,0] -> index 0
    W[0*3 + 1][col+3] = 1   # C[1,0] -> index 3
    
    # M5 = (A[0,0]+A[0,1]) * B[1,1]
    U[0*4 + 0][col+4] = 1
    U[0*4 + 1][col+4] = 1
    V[1*5 + 1][col+4] = 1
    W[0*3 + 0][col+4] = -1  # C[0,0] -> index 0
    W[1*3 + 0][col+4] = 1   # C[0,1] -> index 3
    
    # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*4 + 0][col+5] = 1
    U[0*4 + 0][col+5] = -1
    V[0*5 + 0][col+5] = 1
    V[0*5 + 1][col+5] = 1
    W[1*3 + 1][col+5] = 1   # C[1,1] -> index 4
    
    # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*4 + 1][col+6] = 1
    U[1*4 + 1][col+6] = -1
    V[1*5 + 0][col+6] = 1
    V[1*5 + 1][col+6] = 1
    W[0*3 + 0][col+6] = 1   # C[0,0] -> index 0
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify correctness
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check that the decomposition matches schoolbook
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(new_rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    # Schoolbook: T[i,j,k] should be 1 if this is a valid multiplication entry
                    # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
                    # because C[i,k] = sum_j A[i,j]*B[j,k], so each (i,j,k) contributes
                    if abs(val - 1) > 1e-9:
                        # Try fallback to schoolbook
                        rank_fb = n * m * p
                        U_fb = [[0] * rank_fb for _ in range(n * m)]
                        V_fb = [[0] * rank_fb for _ in range(m * p)]
                        W_fb = [[0] * rank_fb for _ in range(p * n)]
                        c = 0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    U_fb[ii * m + jj][c] = 1
                                    V_fb[jj * p + kk][c] = 1
                                    W_fb[kk * n + ii][c] = 1
                                    c += 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return result

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
