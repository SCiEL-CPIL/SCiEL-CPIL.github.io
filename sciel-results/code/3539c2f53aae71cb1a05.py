"""
Matrix multiplication 6x6x6 using 3x3 block partition with Strassen on each 2x2 block.
This yields a rank of 3^3 * 7 = 189, which is a significant improvement over the schoolbook rank of 216.

The key insight is that we can partition the 6x6 matrix into a 3x3 grid of 2x2 blocks.
Each 2x2 block multiplication can be computed using Strassen's algorithm with rank 7.
Since the blocks are independent (no cross-block interactions in the standard block multiplication),
the total rank is the product of the number of block positions and the rank per block.

Wait, that's not quite right. Let me reconsider.

Actually, for block matrix multiplication, if we have a 3x3 grid of 2x2 blocks, the multiplication 
of the full 6x6 matrices involves summing over the middle dimension. For each output block (i,j),
we need to compute sum_k A[i,k] * B[k,j] where each A[i,k] and B[k,j] is a 2x2 matrix.

This means for each of the 9 output block positions, we need to compute a sum of 3 products of 2x2 matrices.
Each 2x2 matrix product has rank 7 (Strassen), so each term A[i,k]*B[k,j] has rank 7.
But we're summing 3 such terms for each output block.

Hmm, this is getting complex. Let me think differently.

Actually, the standard approach for block-structured matrix multiplication using tensor decomposition 
is more subtle. Let me reconsider the problem.

For n x m x p matrix multiplication, the tensor T has entries T[i,j,k] = delta_{i,j+k} (in some indexing).
The rank of this tensor is the bilinear complexity of matrix multiplication.

For 6x6x6, the best known upper bound is 216 (schoolbook), but there are better algorithms.
Using the 3x3 block partition with 2x2 blocks:
- We have 3x3 = 9 output blocks
- Each output block is a 2x2 matrix
- To compute each output block, we need to multiply 2x2 matrices and sum

Actually, I think I was overcomplicating this. Let me look at this from the perspective of 
the tensor decomposition directly.

The matrix multiplication tensor for n x n x n has rank at most n^3 (schoolbook).
For n=6, that's 216.

Using block structure: if we partition into blocks of size b x b, and have (n/b) x (n/b) blocks,
then we can use the block structure to potentially reduce the rank.

For 6x6 with 2x2 blocks, we have a 3x3 grid of blocks.
The multiplication of the block matrix involves:
C[i,j] = sum_k A[i,k] * B[k,j]

where A[i,k], B[k,j], C[i,j] are 2x2 matrices.

The key insight is that the tensor for 6x6x6 multiplication can be viewed as a tensor over 
the block indices (i,j,k) in {0,1,2}^3, and for each such triple, we have a 2x2x2 matrix multiplication tensor.

So the 6x6x6 tensor is essentially a "tensor product" of the 3x3x3 block index tensor (which is just the identity in some sense) 
and the 2x2x2 matrix multiplication tensor.

Wait, that's not quite right either. Let me think more carefully.

Actually, the 6x6x6 matrix multiplication tensor T_6 can be expressed in terms of the 2x2x2 tensor T_2 and the 3x3x3 "block index" tensor.

More precisely, if we write the indices as (i1,i2) where i1 in {0,1,2} is the block index and i2 in {0,1} is the within-block index,
then T_6[(i1,i2),(j1,j2),(k1,k2)] = delta_{i1,j1+k1} * T_2[(i2,j2,k2)]

Wait, that's not correct. The delta should be on the block indices: delta_{i1,k1} for the summation, and then the within-block multiplication.

Let me be more careful. In matrix multiplication C = A * B, we have:
C[p,q] = sum_r A[p,r] * B[r,q]

For 6x6 matrices with 2x2 blocks:
- p = (p1, p2) where p1 in {0,1,2} is block row, p2 in {0,1} is within-block row
- q = (q1, q2) where q1 in {0,1,2} is block col, q2 in {0,1} is within-block col  
- r = (r1, r2) where r1 in {0,1,2} is block row for B / block col for A, r2 in {0,1} is within-block

Then:
C[(p1,p2),(q1,q2)] = sum_{r1,r2} A[(p1,p2),(r1,r2)] * B[(r1,r2),(q1,q2)]

= sum_{r1} sum_{r2} A_block[p1,r1][(p2,r2)] * B_block[r1,q1][(r2,q2)]

= sum_{r1} [A_block[p1,r1] * B_block[r1,q1]][(p2,q2)]

So for each fixed (p1,q1), the within-block computation is:
C_block[p1,q1][(p2,q2)] = sum_{r1} sum_{r2} A_block[p1,r1][(p2,r2)] * B_block[r1,q1][(r2,q2)]

This is a sum over r1 of 2x2 matrix products.

The tensor for this is more complex than a simple tensor product.

Given the complexity, and that the directive says to try the 3x3 block partition with Strassen on each 2x2 block yielding rank 189, 
let me just implement that directly.

The idea is:
- For each of the 9 block positions (p1,q1), we need to compute a 2x2 matrix C_block[p1,q1]
- C_block[p1,q1] = sum_{r1=0}^{2} A_block[p1,r1] * B_block[r1,q1]
- Each A_block[p1,r1] * B_block[r1,q1] is a 2x2 matrix product, which has rank 7

But we're summing 3 such products. The rank of a sum of tensors is at most the sum of the ranks, so each C_block[p1,q1] computation has rank at most 3*7 = 21.

Total rank would be 9 * 21 = 189.

Yes! That matches. So the construction is:
- For each block position (p1,q1) in {0,1,2}x{0,1,2}
- For each r1 in {0,1,2}
- Use the Strassen decomposition for the 2x2 matrix product A_block[p1,r1] * B_block[r1,q1]
- Sum these 3 terms

Each Strassen decomposition uses 7 rank-1 terms, so each (p1,q1,r1) triple contributes 7 terms.
Total: 9 * 3 * 7 = 189 terms.

Let me implement this.

Strassen's 7 terms for 2x2 matrices:
Given A = [[a1,a2],[a3,a4]] and B = [[b1,b2],[b3,b4]], the product C = A*B has:
C11 = a1*b1 + a3*b4
C12 = a2*b1 + a4*b3  
C21 = a1*b3 + a3*b2
C22 = a2*b4 + a4*b1

Wait, that's the standard 4-term formula. Strassen uses 7 multiplications:
m1 = (a1+a4)*(b1+b4)
m2 = (a3+a4)*b1
m3 = a1*(b2-b4)
m4 = a4*(b3-b1)
m5 = (a1+a2)*b4
m6 = (a3-a1)*(b1+b2)
m7 = (a2-a4)*(b3+b4)

Then:
C11 = m1 + m4 - m5 + m7
C12 = m3 + m5
C21 = m2 + m4
C22 = m1 - m2 + m3 + m6

Each mj is a product of a linear combination of A entries and a linear combination of B entries, so each is rank 1.
Thus the 2x2x2 matrix multiplication tensor has rank 7.

Now I need to express this in terms of the U, V, W matrices for the full 6x6x6 problem.

For the full problem:
- U has shape (n*m, rank) = (36, 189)
- V has shape (m*p, rank) = (36, 189)
- W has shape (p*n, rank) = (36, 189)

The index conventions:
- U[i*m+j, c] where i is row of A, j is col of A
- V[j*p+k, c] where j is row of B, k is col of B
- W[k*n+i, c] where k is row of B, i is row of A (output row)

Wait, let me re-read the task: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So for A (n x m), index is i*m+j
For B (m x p), index is j*p+k
For C (p x n)? No, C is n x p typically. But W[k*n+i, c] suggests k is from B's rows (which is m) and i is from A's rows (n).

Actually, in the tensor decomposition for matrix multiplication, we have:
C[i,k] = sum_j A[i,j] * B[j,k]

So the tensor T[i,j,k] = delta_{i,k} in some sense... no.

The matrix multiplication tensor is T where T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].

In the rank-1 decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product)

Where:
- U has index (i,j) for A
- V has index (j,k) for B  
- W has index (k,i) for C (since C[k,i]... wait, that doesn't match)

Let me re-read: "W[k*n+i, r]"

So W's first index is k*n+i, where k ranges over... B's row index (which is m=6) and i ranges over A's row index (n=6).

So W[k*n+i, c] corresponds to C[i,k] (output row i, col k).

Yes, that makes sense. C[i,k] = sum_j A[i,j] * B[j,k]

So:
- U[i*m+j, c]: A's (i,j) entry
- V[j*p+k, c]: B's (j,k) entry
- W[k*n+i, c]: C's (i,k) entry (indexed as k*n+i)

Now for the block structure:
- A[i,j] where i = p1*2+p2, j = r1*2+r2 (p1,p2 in {0,1}, r1,r2 in {0,1})
  Wait, i ranges 0..5, so p1 in {0,1,2}, p2 in {0,1}
  j ranges 0..5, so r1 in {0,1,2}, r2 in {0,1}
  
- B[j,k] where j = r1*2+r2, k = q1*2+q2
  j ranges 0..5, k ranges 0..5
  So r1 in {0,1,2}, r2 in {0,1}, q1 in {0,1,2}, q2 in {0,1}

- C[i,k] where i = p1*2+p2, k = q1*2+q2
  p1 in {0,1,2}, p2 in {0,1}, q1 in {0,1,2}, q2 in {0,1}

For each (p1,q1) block of C, we compute:
C_block[p1,q1][(p2,q2)] = sum_{r1=0}^{2} sum_{r2=0}^{1} A_block[p1,r1][(p2,r2)] * B_block[r1,q1][(r2,q2)]

= sum_{r1=0}^{2} [A_block[p1,r1] * B_block[r1,q1]][(p2,q2)]

For each (p1,q1,r1), we use Strassen to compute the 2x2 product A_block[p1,r1] * B_block[r1,q1].

The 7 Strassen terms for this 2x2 product:
Let A_block[p1,r1] = [[a1,a2],[a3,a4]] where:
  a1 = A[(p1*2+0),(r1*2+0)] = A[p1*2, r1*2]
  a2 = A[(p1*2+0),(r1*2+1)] = A[p1*2, r1*2+1]
  a3 = A[(p1*2+1),(r1*2+0)] = A[p1*2+1, r1*2]
  a4 = A[(p1*2+1),(r1*2+1)] = A[p1*2+1, r1*2+1]

Let B_block[r1,q1] = [[b1,b2],[b3,b4]] where:
  b1 = B[(r1*2+0),(q1*2+0)] = B[r1*2, q1*2]
  b2 = B[(r1*2+0),(q1*2+1)] = B[r1*2, q1*2+1]
  b3 = B[(r1*2+1),(q1*2+0)] = B[r1*2+1, q1*2]
  b4 = B[(r1*2+1),(q1*2+1)] = B[r1*2+1, q1*2+1]

Strassen terms:
m1 = (a1+a4)*(b1+b4)
m2 = (a3+a4)*b1
m3 = a1*(b2-b4)
m4 = a4*(b3-b1)
m5 = (a1+a2)*b4
m6 = (a3-a1)*(b1+b2)
m7 = (a2-a4)*(b3+b4)

Result:
C11 = m1 + m4 - m5 + m7
C12 = m3 + m5
C21 = m2 + m4
C22 = m1 - m2 + m3 + m6

Now, for the full tensor decomposition, each mj is a rank-1 term.
For example, m1 = (a1+a4)*(b1+b4)

This contributes to C11, C22 (with coefficients +1, +1 respectively)

In the U,V,W representation:
For this particular term (p1,q1,r1, m1):
- U contribution: a1+a4 = A[p1*2, r1*2] + A[p1*2+1, r1*2+1]
  So U[(p1*2)*6 + r1*2, c] += 1
     U[(p1*2+1)*6 + r1*2+1, c] += 1
  
- V contribution: b1+b4 = B[r1*2, q1*2] + B[r1*2+1, q1*2+1]
  So V[(r1*2)*6 + q1*2, c] += 1
     V[(r1*2+1)*6 + q1*2+1, c] += 1

- W contribution: This term contributes +1 to C11 and +1 to C22
  C11 is at position (i,k) = (p1*2+0, q1*2+0), so W index = (q1*2+0)*6 + (p1*2+0)
  C22 is at position (i,k) = (p1*2+1, q1*2+1), so W index = (q1*2+1)*6 + (p1*2+1)
  
  So W[(q1*2+0)*6 + (p1*2+0), c] += 1
     W[(q1*2+1)*6 + (p1*2+1), c] += 1

I'll implement this for all 7 Strassen terms, for all (p1,q1,r1) in {0,1,2}^3.

Total rank = 9 * 3 * 7 = 189.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=6, m=6, p=6
    
    # Block size 2x2, so 3x3 grid of blocks
    block_size = 2
    num_blocks = n // block_size  # 3
    
    # Total rank
    rank = num_blocks * num_blocks * num_blocks * 7  # 3^3 * 7 = 189
    
    # Initialize U, V, W as lists of lists (integer entries)
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    col = 0
    
    # Strassen coefficients for the 7 terms
    # Each term has:
    #   - A linear combination for U (coefficients on A entries)
    #   - A linear combination for V (coefficients on B entries)
    #   - Contributions to C entries (coefficients on C entries)
    
    # For a 2x2 product, the 7 Strassen terms are:
    # m1 = (a1+a4)*(b1+b4) -> contributes to C11 (+1), C22 (+1)
    # m2 = (a3+a4)*b1       -> contributes to C21 (+1), C22 (-1)
    # m3 = a1*(b2-b4)       -> contributes to C12 (+1), C22 (+1)
    # m4 = a4*(b3-b1)       -> contributes to C11 (+1), C21 (+1)
    # m5 = (a1+a2)*b4       -> contributes to C11 (-1), C12 (+1)
    # m6 = (a3-a1)*(b1+b2)  -> contributes to C22 (+1)
    # m7 = (a2-a4)*(b3+b4)  -> contributes to C11 (+1)
    
    # Define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (a_idx) to coeff, where a_idx is 0..3 for a1,a2,a3,a4
    # V_coeffs: dict mapping (b_idx) to coeff, where b_idx is 0..3 for b1,b2,b3,b4
    # W_coeffs: dict mapping (c_idx) to coeff, where c_idx is 0..3 for C11,C12,C21,C22
    
    strassen_terms = [
        # m1 = (a1+a4)*(b1+b4)
        (
            {0: 1, 3: 1},  # a1 + a4
            {0: 1, 3: 1},  # b1 + b4
            {0: 1, 3: 1}   # C11 +1, C22 +1
        ),
        # m2 = (a3+a4)*b1
        (
            {2: 1, 3: 1},  # a3 + a4
            {0: 1},        # b1
            {2: 1, 3: -1}  # C21 +1, C22 -1
        ),
        # m3 = a1*(b2-b4)
        (
            {0: 1},        # a1
            {1: 1, 3: -1}, # b2 - b4
            {1: 1, 3: 1}   # C12 +1, C22 +1
        ),
        # m4 = a4*(b3-b1)
        (
            {3: 1},        # a4
            {2: 1, 0: -1}, # b3 - b1
            {0: 1, 2: 1}   # C11 +1, C21 +1
        ),
        # m5 = (a1+a2)*b4
        (
            {0: 1, 1: 1},  # a1 + a2
            {3: 1},        # b4
            {0: -1, 1: 1}  # C11 -1, C12 +1
        ),
        # m6 = (a3-a1)*(b1+b2)
        (
            {2: 1, 0: -1}, # a3 - a1
            {0: 1, 1: 1},  # b1 + b2
            {3: 1}         # C22 +1
        ),
        # m7 = (a2-a4)*(b3+b4)
        (
            {1: 1, 3: -1}, # a2 - a4
            {2: 1, 3: 1},  # b3 + b4
            {0: 1}         # C11 +1
        ),
    ]
    
    for p1 in range(num_blocks):
        for q1 in range(num_blocks):
            for r1 in range(num_blocks):
                for term_idx in range(7):
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[term_idx]
                    
                    # Add U contributions
                    for a_idx, coeff in u_coeffs.items():
                        # a_idx 0->a1, 1->a2, 2->a3, 3->a4
                        # a1 = A[p1*2+0, r1*2+0], a2 = A[p1*2+0, r1*2+1]
                        # a3 = A[p1*2+1, r1*2+0], a4 = A[p1*2+1, r1*2+1]
                        if a_idx == 0:
                            i_a = p1 * block_size + 0
                            j_a = r1 * block_size + 0
                        elif a_idx == 1:
                            i_a = p1 * block_size + 0
                            j_a = r1 * block_size + 1
                        elif a_idx == 2:
                            i_a = p1 * block_size + 1
                            j_a = r1 * block_size + 0
                        else:  # a_idx == 3
                            i_a = p1 * block_size + 1
                            j_a = r1 * block_size + 1
                        
                        u_idx = i_a * m + j_a
                        U[u_idx][col] += coeff
                    
                    # Add V contributions
                    for b_idx, coeff in v_coeffs.items():
                        # b_idx 0->b1, 1->b2, 2->b3, 3->b4
                        # b1 = B[r1*2+0, q1*2+0], b2 = B[r1*2+0, q1*2+1]
                        # b3 = B[r1*2+1, q1*2+0], b4 = B[r1*2+1, q1*2+1]
                        if b_idx == 0:
                            i_b = r1 * block_size + 0
                            j_b = q1 * block_size + 0
                        elif b_idx == 1:
                            i_b = r1 * block_size + 0
                            j_b = q1 * block_size + 1
                        elif b_idx == 2:
                            i_b = r1 * block_size + 1
                            j_b = q1 * block_size + 0
                        else:  # b_idx == 3
                            i_b = r1 * block_size + 1
                            j_b = q1 * block_size + 1
                        
                        v_idx = i_b * p + j_b
                        V[v_idx][col] += coeff
                    
                    # Add W contributions
                    for c_idx, coeff in w_coeffs.items():
                        # c_idx 0->C11, 1->C12, 2->C21, 3->C22
                        # C11 = C[p1*2+0, q1*2+0], C12 = C[p1*2+0, q1*2+1]
                        # C21 = C[p1*2+1, q1*2+0], C22 = C[p1*2+1, q1*2+1]
                        if c_idx == 0:
                            i_c = p1 * block_size + 0
                            k_c = q1 * block_size + 0
                        elif c_idx == 1:
                            i_c = p1 * block_size + 0
                            k_c = q1 * block_size + 1
                        elif c_idx == 2:
                            i_c = p1 * block_size + 1
                            k_c = q1 * block_size + 0
                        else:  # c_idx == 3
                            i_c = p1 * block_size + 1
                            k_c = q1 * block_size + 1
                        
                        # W index is k*n + i
                        w_idx = k_c * n + i_c
                        W[w_idx][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    
    n, m, p = DIMENSIONS
    
    # Verify the decomposition
    # C = sum_c U[:,c] * V[:,c] * W[:,c] (in the appropriate sense)
    # More precisely: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify with random matrices
    np.random.seed(42)
    A = np.random.randint(-2, 3, size=(n, m))
    B = np.random.randint(-2, 3, size=(m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p), dtype=int)
    for c in range(U.shape[1]):
        u_col = U[:, c]
        v_col = V[:, c]
        w_col = W[:, c]
        # u_col has shape (n*m,), v_col has shape (m*p,), w_col has shape (p*n,)
        # Reshape and multiply
        u_mat = u_col.reshape(n, m)
        v_mat = v_col.reshape(m, p)
        w_mat = w_col.reshape(p, n).T  # w_mat[i,k] = W[k*n+i, c]
        
        # The contribution is u_mat * v_mat (outer product in some sense)
        # Actually: for each j, sum_j U[i*m+j,c] * V[j*p+k,c] gives a matrix, then multiply by W
        # More carefully:
        # term[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's compute this directly
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += U[i*m+j, c] * V[j*p+k, c]
                C_computed[i, k] += s * W[k*n+i, c]
    
    if np.allclose(C_computed, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print("Expected:", C_expected)
        print("Computed:", C_computed)
        print("Difference:", C_computed - C_expected)
    
    print(f"Rank: {U.shape[1]}")
