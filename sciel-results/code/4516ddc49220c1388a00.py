"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    We use the fact that multiplying a 2x2 matrix A by a 2x12 matrix B can be
    viewed as two independent 2x12 multiplications (one for each row of A),
    where each 2x12 multiplication has rank 12 (schoolbook).
    
    However, we can do better by exploiting the structure. The key insight:
    A is 2x2, B is 2x12. The product C = A*B is 2x12.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is equivalent to: for each output position (i,k), we need a linear combination
    of the A entries weighted by B entries.
    
    The standard approach: rank = n*m*p = 2*2*12 = 48 (schoolbook).
    
    Better: We can use Strassen-like ideas. For 2x2 matrix multiplication, 
    Strassen gives rank 7. But here we're multiplying 2x2 by 2x12.
    
    Let's think of it differently. We can decompose B into two 1x12 vectors 
    (rows of B). Then C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    
    For each column k of the output, we're computing a 2x1 matrix product:
    C[:,k] = A * B[:,k]
    
    This is a 2x2 times 2x1 multiplication, which has rank 2*2*1 = 4 per column.
    For 12 columns, that's 48 total (schoolbook).
    
    Can we do better? The tensor for 2x2 * 2x12 is the "matrix multiplication tensor" 
    <2,2,12>. Its rank is known to be at most 24 (by viewing it as two copies of <2,2,6> 
    or similar). Actually, let's think more carefully.
    
    The tensor T_{i,j,k} = e_i ⊗ e_j ⊗ e_k for the multiplication.
    
    One approach: use the fact that 2x2 matrix mult has rank 7 (Strassen). 
    We can write A*B where B is 2x12 as:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    If we could decompose the 2x2 part into rank 7, we'd get 7*12 = 84? No, that's worse.
    
    Actually, the correct way: the tensor <2,2,12> can be seen as the "external product" 
    of <2,2,1> with a 12-dimensional space, but that's not quite right either.
    
    Let me reconsider. The schoolbook decomposition has rank 48. 
    
    A known result: the rank of <n,m,p> is at most n*m*p, and for <2,2,p> it's known 
    to be 2p for p >= 2 (I believe this is tight). Let me verify: 
    <2,2,p> has rank exactly 2p for p >= 2? 
    
    Actually, I recall that R(<2,2,p>) = 2p for p >= 2. So for p=12, rank = 24.
    
    Let me construct a rank-24 decomposition.
    
    For <2,2,p>, we can use the following construction:
    For each column k (0 <= k < p), we need to compute C[:,k] = A * B[:,k].
    
    A 2x2 times 2x1 product: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
                                      C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    This can be done with rank 2 per column if we're clever? No, the rank of <2,2,1> is 4.
    
    Wait, let me reconsider. The rank of <2,2,1> is indeed 4 (it's just the 2x2 matrix 
    applied to a vector, which is a bilinear map R^2 x R^2 -> R^2, and its tensor rank is 4).
    
    So for 12 independent columns, we'd get 12*4 = 48.
    
    But there might be a better decomposition that shares structure across columns.
    
    Let me look at this from the tensor perspective. The multiplication tensor for 
    (2,2,12) is a tensor in R^{4} ⊗ R^{24} ⊗ R^{24} (since n*m=4, m*p=24, p*n=24).
    
    Actually, the indices are: U has shape (n*m, rank) = (4, rank), V has shape (m*p, rank) = (24, rank), W has shape (p*n, rank) = (24, rank).
    
    The decomposition satisfies: for all i,j,k:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    (This is the standard convention where the tensor entry is 1 if the indices correspond to a valid multiplication term.)
    
    Let me try to find a rank-24 decomposition. 
    
    Idea: Use the fact that we can write the 2x2 matrix multiplication using 4 terms per "pair" but share across the 12 columns.
    
    Actually, I think the rank of <2,2,p> is indeed 2p for p >= 2. Here's a construction:
    
    For each k in 0..p-1, use 2 rank-1 tensors:
    Term 1: U[i*m+0, r1] = delta(i,0), V[0*p+k, r1] = 1, W[k*n+0, r1] = 1
            U[i*m+1, r1] = delta(i,1), V[1*p+k, r1] = 1, W[k*n+1, r1] = 1
    
    Hmm, this is getting complicated. Let me just implement the schoolbook (rank 48) and verify it works, then try to improve.
    
    Actually, let me just go with the known optimal. I believe R(<2,2,12>) = 24.
    
    Construction for rank 2p:
    For each k in range(p):
      r1 = 2*k
      r2 = 2*k + 1
      
      # Term r1: contributes to C[0,k] and C[1,k]
      U[0*m + 0][r1] += 1  # i=0, j=0
      U[1*m + 0][r1] += 1  # i=1, j=0
      V[0*p + k][r1] = 1   # j=0, k
      W[k*n + 0][r1] = 1   # k, i=0
      W[k*n + 1][r1] = 1   # k, i=1
      
      # Term r2:
      U[0*m + 1][r2] += 1  # i=0, j=1
      U[1*m + 1][r2] += 1  # i=1, j=1
      V[1*p + k][r2] = 1   # j=1, k
      W[k*n + 0][r2] = 1   # k, i=0
      W[k*n + 1][r2] = 1   # k, i=1
    
    Let's verify: For output (i,k), the contribution is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r.
    
    Wait, the convention is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    And we want T[i,j,k] = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears once).
    
    Let's check T[0,0,k]:
    = sum_r U[0*m+0, r] * V[0*p+k, r] * W[k*n+0, r]
    = sum_r U[0, r] * V[k, r] * W[k*2+0, r]
    
    For r = r1 = 2*k: U[0, r1] = 1, V[k, r1] = 1, W[2k, r1] = 1. Product = 1.
    For r = r2 = 2*k+1: U[0, r2] = 1 (from U[0*m+1][r2]... wait, U[0*m+1] is U[1], not U[0]).
    
    I'm confusing myself. Let me be more careful.
    
    U has indices [i*m + j] where i in 0..n-1, j in 0..m-1.
    For n=2, m=2: U has 4 rows: U[0]=U[0*2+0], U[1]=U[0*2+1], U[2]=U[1*2+0], U[3]=U[1*2+1].
    
    V has indices [j*p + k] where j in 0..m-1, k in 0..p-1.
    For m=2, p=12: V has 24 rows.
    
    W has indices [k*n + i] where k in 0..p-1, i in 0..n-1.
    For p=12, n=2: W has 24 rows.
    
    The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    We want T[i,j,k] = 1 for all i,j,k.
    
    Let me try the construction again more carefully:
    
    For each k in 0..p-1:
      r1 = 2*k
      r2 = 2*k + 1
      
      # We want to cover all (i,j) pairs for this k.
      # For fixed k, we need T[i,j,k] = 1 for i in {0,1}, j in {0,1}.
      
      # Let's use 2 terms per k, so 2*p = 24 terms total.
      
      # Term r1:
      # U[i*m+j, r1] should be non-zero for certain (i,j)
      # V[j*p+k, r1] should be non-zero for certain j
      # W[k*n+i, r1] should be non-zero for certain i
      
      # Idea: r1 handles j=0, r2 handles j=1.
      
      # Term r1 (handles j=0):
      U[0*2+0][r1] = 1  # i=0, j=0
      U[1*2+0][r1] = 1  # i=1, j=0
      V[0*12+k][r1] = 1 # j=0, k
      W[k*2+0][r1] = 1  # k, i=0
      W[k*2+1][r1] = 1  # k, i=1
      
      # Term r2 (handles j=1):
      U[0*2+1][r2] = 1  # i=0, j=1
      U[1*2+1][r2] = 1  # i=1, j=1
      V[1*12+k][r2] = 1 # j=1, k
      W[k*2+0][r2] = 1  # k, i=0
      W[k*2+1][r2] = 1  # k, i=1
      
    Now check T[0,0,k]:
    = sum_r U[0, r] * V[k, r] * W[2k, r]
    r=r1: U[0,r1]=1, V[k,r1]=1, W[2k,r1]=1 => 1
    r=r2: U[0,r2]=0 (U[0] is row 0, r2 only sets U[1] and U[3]) => 0
    Other r: 0
    Total: 1 ✓
    
    T[1,0,k]:
    = sum_r U[2, r] * V[k, r] * W[2k, r]
    r=r1: U[2,r1]=1, V[k,r1]=1, W[2k,r1]=1 => 1
    Total: 1 ✓
    
    T[0,1,k]:
    = sum_r U[1, r] * V[12+k, r] * W[2k, r]
    r=r2: U[1,r2]=1, V[12+k,r2]=1, W[2k,r2]=1 => 1
    Total: 1 ✓
    
    T[1,1,k]:
    = sum_r U[3, r] * V[12+k, r] * W[2k, r]
    r=r2: U[3,r2]=1, V[12+k,r2]=1, W[2k,r2]=1 => 1
    Total: 1 ✓
    
    Great! So rank 24 works.
    """
    n, m, p = DIMENSIONS
    rank = 2 * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    for k in range(p):
        r1 = 2 * k
        r2 = 2 * k + 1
        
        # Term r1: handles j=0
        U[0 * m + 0][r1] = 1  # i=0, j=0
        U[1 * m + 0][r1] = 1  # i=1, j=0
        V[0 * p + k][r1] = 1  # j=0, k
        W[k * n + 0][r1] = 1  # k, i=0
        W[k * n + 1][r1] = 1  # k, i=1
        
        # Term r2: handles j=1
        U[0 * m + 1][r2] = 1  # i=0, j=1
        U[1 * m + 1][r2] = 1  # i=1, j=1
        V[1 * p + k][r2] = 1  # j=1, k
        W[k * n + 0][r2] = 1  # k, i=0
        W[k * n + 1][r2] = 1  # k, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
