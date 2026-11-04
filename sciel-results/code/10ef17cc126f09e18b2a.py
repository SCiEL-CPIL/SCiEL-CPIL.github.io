"""Find rank-minimal tensor decomposition for 4x4 x 4x8 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 8]

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    best = None
    
    # Build the multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to valid mult
    # T is a 3-tensor of shape (n*m, m*p, p*n)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The decomposition: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # where i indexes (row_of_A, col_of_A), j indexes (col_of_A, row_of_B), k indexes (col_of_B, row_of_output)
    
    # Actually, let me be more careful about the indexing.
    # Matrix multiplication: C = A * B where A is 4x4, B is 4x8, C is 4x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor T has entries T[i,j,k] = 1 for each triple (i,j,k)
    # U has shape (n*m, rank) where the row index is (i,j)
    # V has shape (m*p, rank) where the row index is (j,k)  
    # W has shape (p*n, rank) where the row index is (k,i)
    # 
    # So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # The full tensor T is of size (n*m) x (m*p) x (p*n) = 16 x 32 x 32
    # We need to find U, V, W of rank r such that for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Strategy: use random sampling + iterative rank reduction
    # Start with a known valid decomposition (schoolbook, rank=128)
    # Then try to reduce rank by finding linear dependencies
    
    # First, let's try a structured approach based on the 4x4x4 base
    # The 4x4x4 matrix multiplication has rank at most 49 (Landsberg bound ~ 4^3 * 1.5 = 96, 
    # but Strassen gives 7 for 2x2x2, and for 4x4x4 the best known is 49)
    
    # For 4x4x8: we can think of B as two 4x4 matrices side by side
    # C[:,0:4] = A * B[:,0:4] and C[:,4:8] = A * B[:,4:8]
    # So we need two copies of 4x4x4 multiplication
    # If 4x4x4 has rank r4, then 4x4x8 has rank at most 2*r4
    # But we can share the A-dependent computations!
    
    # Key insight: In C[i,k] = sum_j A[i,j]*B[j,k], the A[i,j] factor is shared 
    # across all k for fixed (i,j). So we can factor out the A-dependence.
    
    # Let's use a "bilinear" decomposition approach:
    # For each (i,j), we need to multiply A[i,j] by all B[j,k] for k=0..7
    # This is like: for each (i,j), compute A[i,j] * [B[j,0],...,B[j,7]]
    
    # A decomposition of rank r:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # For fixed (i,j), sum over k and c: W[k*n+i, c] * V[j*p+k, c] must give B[j,k] scaled by A[i,j]
    # 
    # Let me try: set U[i*m+j, c] = 1 for certain c, and use V and W to handle the rest
    # 
    # Simpler: use the fact that for each (i,j), we need rank-1 updates to the output row i
    # C[i,:] += A[i,j] * B[j,:]
    # Each such update is a rank-1 operation on the output row
    # There are n*m = 16 such updates, each of rank 1 in the output dimension
    # Total rank = 16 * 1 = 16? No, that's not how tensor rank works.
    
    # Let me think differently. The tensor is T[i,j,k] = delta(i,j,k) where the "1" 
    # means this triple contributes to the product.
    # 
    # Actually T[i,j,k] = 1 for ALL valid (i,j,k) triples. There are 4*4*8 = 128 such triples.
    # The tensor has 128 non-zero entries (all 1s).
    # 
    # We want: T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    # 
    # This is a 3-way tensor of rank r.
    # 
    # Let me try to find a low-rank decomposition using a greedy/randomized approach.
    # 
    # Approach: 
    # 1. Start with r = 128 (schoolbook)
    # 2. Randomly perturb and try to reduce
    # 
    # Better approach: Use the structure.
    # 
    # For 4x4x8, note that p=8=2*4. Think of B as B = [B1 | B2] where B1, B2 are 4x4.
    # C = [A*B1 | A*B2]
    # 
    # If we have a rank-r4 decomposition of 4x4x4:
    #   (A*B)[i,k] = sum_{c=1}^{r4} U4[i,j,c]*V4[j,k,c]*W4[k,i,c]
    # 
    # Then for 4x4x8:
    #   C[i,k] for k<4: sum_c U4[i,j,c]*V4[j,k,c]*W4[k,i,c]
    #   C[i,k] for k>=4: sum_c U4[i,j,c]*V4[j,k-4,c]*W4[k-4,i,c]
    # 
    # We can combine: define V8[j,k,c] = V4[j,k%4,c] and W8[k,i,c] = W4[k%4,i,c]
    # Then rank is still r4, but we need separate U for each "block"?
    # 
    # No wait, U doesn't depend on k, so we can reuse the same U!
    # U8[i,j,c] = U4[i,j,c] for all c
    # V8[j,k,c] = V4[j, k%4, c] 
    # W8[k,i,c] = W4[k%4, i, c]
    # 
    # This gives a valid decomposition of rank r4 for 4x4x8!
    # 
    # So if 4x4x4 has rank r4, then 4x4x8 has rank at most r4.
    # 
    # Best known rank for 4x4x4 matrix multiplication: 49 (from 2x2x2 Strassen rank 7, 
    # 4x4x4 can be done with 49 using various methods)
    # 
    # Actually, let me recall: 
    # - 2x2x2: rank 7 (Strassen)
    # - 4x4x4: The best known upper bound is 49. This comes from 
    #   using the 2x2x2 rank-7 decomposition recursively, but with sharing.
    #   Actually, 4x4x4 = 2x(2x2x2) with some sharing.
    #   The tensor rank of 4x4x4 is known to be at most 49 and at least 48.
    # 
    # Wait, let me reconsider. The standard result:
    # - [2,2,2] tensor rank = 7
    # - [4,4,4] tensor rank <= 49 (this is a well-known result)
    # - [4,4,4] tensor rank >= 48 (Landsberg lower bound)
    # 
    # So if I can construct a rank-49 decomposition of 4x4x4, I get rank 49 for 4x4x8.
    # 
    # Let me construct the 4x4x4 rank-49 decomposition.
    # 
    # The 4x4x4 matrix multiplication can be decomposed using 49 terms.
    # One way: use the fact that 4x4 = 2x(2x2) and apply Strassen-like methods.
    # 
    # Actually, a cleaner approach: 
    # The 4x4x4 tensor can be decomposed as follows.
    # Write 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B where C is also 2x2 blocks of 2x2 matrices.
    # 
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # 
    # Each Ci,j is a 2x2 matrix product of 2x2 matrices plus another.
    # Using Strassen for each 2x2x2 product takes 7 terms.
    # But we have 4 such products (one for each Ci,j), so naive would be 4*7=28.
    # But we can share computations!
    # 
    # Actually, the standard 4x4x4 rank-49 construction is more involved.
    # Let me just use a known explicit construction.
    
    # For now, let me try a different approach: use random search to find a low-rank decomposition.
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Check all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using matrix operations."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # For each (i,j,k): sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # This is a batched operation
        # Let's do it efficiently
        for i in range(n):
            for j in range(m):
                # sum_k sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
                # = sum_c U3[i,j,c] * (sum_k V3[j,k,c] * W3[k,i,c])
                # For fixed i,j: sum_c U3[i,j,c] * dot(V3[j,:,c], W3[:,i,c])
                for c in range(rank):
                    pass
        # Simpler: just do the full check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to construct a decomposition based on 4x4x4 rank-49
    # 
    # The 4x4x4 matrix multiplication tensor of rank 49:
    # I'll use a construction based on the following:
    # 
    # For 4x4x4, one known rank-49 decomposition uses the following idea:
    # Split into 4 quadrants and use 7 Strassen terms for each "layer" with sharing.
    # 
    # Actually, let me try a more direct approach: random search for low rank.
    
    # Try ranks from low to high
    for target_rank in [49, 50, 52, 54, 56, 60, 64, 70, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128]:
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition of this rank
        # Use a randomized approach:
        # 1. Initialize U, V, W randomly
        # 2. Use iterative refinement
        
        # Alternative: use the structure to construct a candidate
        # 
        # For 4x4x8, let me try the following structured approach:
        # 
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # 
        # Decomposition of rank r:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # 
        # Idea: for each (i,j), we need to "broadcast" A[i,j] to all k.
        # So for each (i,j), assign some columns c to this (i,j).
        # For those columns, set U[i*m+j, c] = 1 (or some value).
        # Then V[j*p+k, c] and W[k*n+i, c] must satisfy:
        #   sum_{c in cols(i,j)} V[j*p+k, c] * W[k*n+i, c] = B[j,k] (for the B part)
        # 
        # But W depends on i, so this couples different i values.
        # 
        # Let me try: assign 1 column per (i,j) pair. That's 16 columns.
        # For column c corresponding to (i0,j0):
        #   U[i*m+j, c] = 1 if (i,j)=(i0,j0), else 0
        #   V[j*p+k, c] = v_{j,k} 
        #   W[k*n+i, c] = w_{k,i}
        #   Then for (i0,j0,k): contribution = v_{j0,k} * w_{k,i0}
        #   We need: sum_{(i,j)} [ (i,j)=(i0,j0) ] * ... 
        #   Actually for fixed (i,j,k), the contribution from column c(i,j) is:
        #     if c corresponds to (i,j): U=1, so contribution = V[j*p+k,c]*W[k*n+i,c]
        #   And we need sum over all (i',j') with (i',j')=(i,j) ... wait, only one column per (i,j).
        #   
        #   So for (i,j,k): only the column c(i,j) contributes, giving V[j*p+k, c(i,j)] * W[k*n+i, c(i,j)]
        #   We need this to equal 1 for all (i,j,k).
        #   So V[j*p+k, c(i,j)] * W[k*n+i, c(i,j)] = 1 for all (i,j,k).
        #   This means V[j*p+k, c(i,j)] = 1/W[k*n+i, c(i,j)]
        #   But V depends on (j,k) and W depends on (k,i), and they must be consistent.
        #   For fixed c(i,j), we need V[j*p+k, c] * W[k*n+i, c] = 1 for all i,k.
        #   This is only possible if V[j*p+k, c] and W[k*n+i, c] are both 1 (or both -1, etc.)
        #   But W[k*n+i, c] must be 1 for all i for fixed k, and V[j*p+k, c] must be 1 for all k.
        #   So V[j*p+k, c(i,j)] = 1 and W[k*n+i, c(i,j)] = 1 for all k,i.
        #   Then the product is 1. 
        #   But wait, this means every column c(i,j) has V=1 and W=1 everywhere!
        #   Then for (i,j,k): contribution = 1*1*1 = 1. 
        #   But we also need that for (i',j',k) with (i',j') != (i,j), the contribution is 0.
        #   U[i'*m+j', c(i,j)] = 0 if (i',j') != (i,j). 
        #   So the total for (i,j,k) = sum_{(i',j')} U[i'*m+j', c(i',j')] * V[j*p+k, c(i',j')] * W[k*n+i, c(i',j')]
        #   = U[i*m+j, c(i,j)] * V[j*p+k, c(i,j)] * W[k*n+i, c(i,j)]
        #   = 1 * 1 * 1 = 1. 
        #   
        #   Wait, but what about other columns? For column c(i',j') with (i',j') != (i,j):
        #   U[i*m+j, c(i',j')] = 0, so no contribution.
        #   
        #   So rank 16 works?! Let me double-check.
        #   
        #   U has shape (16, 16), V has shape (32, 16), W has shape (32, 16).
        #   U[i*m+j, c] = 1 if c == i*m+j, else 0. So U is the identity matrix (16x16).
        #   V[j*p+k, c] = 1 for all j,k,c. So V is all ones (32x16).
        #   W[k*n+i, c] = 1 for all k,i,c. So W is all ones (32x16).
        #   
        #   For (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        #   = sum_c delta(i*m+j, c) * 1 * 1
        #   = 1 (since c = i*m+j is in range 0..15)
        #   
        #   Yes! Rank 16 works!
    
    # Wait, that seems too good to be true. Let me verify more carefully.
    # 
    # The tensor T[i,j,k] should be 1 for all valid (i,j,k) where i in [0,4), j in [0,4), k in [0,8).
    # With U = I_16, V = ones(32,16), W = ones(32,16):
    # T[i,j,k] = sum_{c=0}^{15} U[i*4+j, c] * V[j*8+k, c] * W[k*4+i, c]
    #          = sum_{c=0}^{15} delta(i*4+j, c) * 1 * 1
    #          = 1
    # 
    # This is correct! The rank is 16.
    # 
    # But wait, is this a valid "tensor decomposition" in the sense intended?
    # The task says "rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix 
    # by a 4 by 8 matrix."
    # 
    # The standard matrix multiplication tensor for m1 x m2 * m2 x m3 has rank at most m1*m2*m3 
    # (schoolbook) and at least ... well, the tensor rank is what we're minimizing.
    # 
    # My construction gives rank 16. Let me make sure this is valid.
    # 
    # Actually, I realize I might be confusing the tensor structure. Let me re-read the task.
    # 
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    # 
    # The matrix multiplication tensor T has T[i,j,k] = 1 if and only if the (i,j) entry of A 
    # and the (j,k) entry of B contribute to the (i,k) entry of C. In standard matrix multiplication,
    # every (i,j,k) triple contributes, so T[i,j,k] = 1 for all valid (i,j,k).
    # 
    # My decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # With U = I_{n*m}, V = ones_{m*p, n*m}, W = ones_{p*n, n*m}:
    # T[i,j,k] = sum_c delta(i*m+j, c) * 1 * 1 = 1. 
    # 
    # This is valid! And the rank is n*m = 16.
    # 
    # Can we do better? The rank must be at least max(n*m, m*p, p*n) / something... 
    # Actually, the rank of the matrix multiplication tensor [n,m,p] is at least 
    # max(n*m, m*p, p*n) / p ... no, that's not right.
    # 
    # The tensor rank of [n,m,p] matrix multiplication is at least n*m (by a simple argument:
    # slice the tensor along the third mode, each slice is an n*m x m*p matrix of rank at most p,
    # but that doesn't directly give a lower bound on tensor rank).
    # 
    # Actually, a known lower bound: rank([n,m,p]) >= max(n*m, m*p, p*n) / max(n,m,p) ... 
    # no, I'm not sure of the exact lower bound.
    # 
    # For [4,4,8]: my construction gives rank 16. Is this minimal?
    # 
    # Consider: the tensor T has 16*32*32 = 16384 entries, all 1.
    # A rank-r decomposition uses 3*r*(dim) parameters.
    # 
    # I believe rank 16 might not be minimal. Let me think about whether we can do better.
    # 
    # Actually, wait. Let me reconsider. In my construction, U is the identity, so each "column" 
    # c corresponds to exactly one (i,j) pair. The V and W are all ones. This is essentially 
    # the schoolbook method but with a different indexing.
    # 
    # The schoolbook method has rank n*m*p = 128, where each (i,j,k) triple gets its own column.
    # My method has rank n*m = 16, where each (i,j) pair gets one column, and the k-dependence 
    # is handled by V and W being all ones.
    # 
    # This works because the tensor T[i,j,k] = 1 for all k, so we don't need separate 
    # k-dependence in the decomposition.
    # 
    # Can we reduce below 16? 
    # 
    # The rank must be at least the rank of any matrix flattening of T.
    # Flattening T as (n*m) x (m*p * p*n) = 16 x 1024:
    # Each row i*m+j has 1s in positions (j*p+k)*p*n + k*n+i for all k.
    # The rank of this flattening is at most 16 (number of rows), so that doesn't help.
    # 
    # Flattening as (n*m * m*p) x (p*n) = 512 x 32:
    # The rank of this flattening gives a lower bound on tensor rank.
    # 
    # Let me compute: T as a 512x32 matrix, entry [(i,j),(k)] = T[i,j,k] = 1.
    # So this is just a 512x32 matrix of all 1s, which has rank 1.
    # That doesn't give a useful lower bound.
    # 
    # Flattening as (n*m * p*n) x (m*p) = 512 x 32:
    # Entry [(i,k),(j)] = T[i,j,k] = 1. Also all 1s, rank 1.
    # 
    # Flattening as (m*p * p*n) x (n*m) = 1024 x 16:
    # Entry [(j,k),(i)] = T[i,j,k] = 1. All 1s, rank 1.
    # 
    # So all matrix flattenings have rank 1, which means the lower bound from flattenings is 1.
    # That's not helpful.
    # 
    # The true tensor rank might be 16 or could be lower. Let me just go with 16 for now,
    # since I have a valid construction.
    
    # Construct the rank-16 decomposition
    rank = n * m  # 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for kk in range(p):
                V[j * p + kk][c] = 1
            for kk in range(p):
                W[kk * n + i][c] = 1
    
    # Verify
    if verify_fast(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
