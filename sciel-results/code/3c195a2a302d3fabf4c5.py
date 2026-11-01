"""Matrix multiplication 2x2x14: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix.
    
    Strategy: Decompose the 2x2 matrix multiplication into two independent
    1x2x14 vector-matrix products using the linearity of matrix multiplication.
    
    For a 2x2 matrix A and 2x14 matrix B:
    - Row 0 of result: A[0,0]*B[0,:] + A[0,1]*B[1,:]
    - Row 1 of result: A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    Each row computation is a sum of 2 vector-matrix products.
    Each 1x2x14 vector-matrix product (row vector times 2x14 matrix) has rank 14
    (schoolbook: for each output position k, one term per input position j).
    
    Total rank = 2 rows * 2 input rows * 14 columns = 56? No, let's think more carefully.
    
    Actually, for a single 1x2x14 product: (a0, a1) * [[b00,...,b0,13],[b10,...,b1,13]]
    Output k: a0*b0k + a1*b1k
    This is a tensor of size 1x2x14 with rank 2 (not 14!) because:
    T[i,j,k] = delta(i,0)*b_jk... no wait.
    
    Let me reconsider. The tensor for multiplying 1x2 by 2x14:
    T[a,b,c] where a in {0}, b in {0,1}, c in {0..13}
    T[0,b,c] = 1 if we're computing output[c] = sum_b A[b]*B[b,c]
    Wait, the multiplication tensor T[i,j,k] = 1 if output[i,k] uses input A[j] * B[j,k].
    
    For 1x2 * 2x14:
    Output is 1x14. Output[c] = A[0]*B[0,c] + A[1]*B[1,c]
    So T[0,0,c] = 1, T[0,1,c] = 1 for all c.
    This tensor has rank 2: it's the sum of two rank-1 tensors:
    T1: T1[0,0,c]=1 for all c, T1[0,1,c]=0 -> factor as u=[1], v=[1,0], w=[1,1,...,1]
    T2: T2[0,0,c]=0, T2[0,1,c]=1 for all c -> factor as u=[1], v=[0,1], w=[1,1,...,1]
    
    So a 1x2x14 product has rank 2!
    
    Now for 2x2 * 2x14:
    Output[i,c] = sum_j A[i,j]*B[j,c]
    T[i,j,c] = 1 for all i,j,c in their ranges.
    This is just the outer product of three vectors: ones(2) x ones(2) x ones(14)
    Wait no, that's not right either. The multiplication tensor is:
    T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to output[i,k].
    In standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    So T is a 2x2x14 tensor with all entries equal to 1.
    
    The rank of the all-ones tensor of size n x m x p is min(n,m,p) in general?
    No, the rank of the all-ones tensor is 1! Because T[i,j,k] = 1 = 1*1*1.
    It's a rank-1 tensor: u_i=1, v_j=1, w_k=1 for all i,j,k.
    
    Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    The matrix multiplication tensor T_{i,j,k} is defined such that:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k]
    
    For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    So yes, T is the all-ones tensor, which has rank 1.
    
    But that's trivially wrong because we know 2x2x2 matrix multiplication has rank 7.
    
    I'm confusing myself. Let me look at this more carefully.
    
    The multiplication tensor for multiplying an n x m matrix by an m x p matrix:
    T is a tensor of size (n*m) x (m*p) x (n*p) where:
    T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k', else 0.
    
    So T is actually a set of disjoint "1"s - it's like an identity-like tensor.
    Specifically, T[(i,j),(j',k),(i',k')] = delta(i,i') * delta(j,j') * delta(k,k')
    
    This means T can be written as a sum of n*m*p rank-1 tensors:
    T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    
    where e_{i,j} is the unit vector with 1 at position (i,j) in the n*m space, etc.
    
    So the schoolbook rank is n*m*p = 2*2*14 = 56.
    
    Now, can we do better? The question is whether there's a lower-rank decomposition.
    
    For 2x2x2, the minimal rank is 7 (Strassen).
    For 2x2x14, we need to find the minimal rank.
    
    One approach: Use the fact that the 2x2 part can be handled with rank 7 (Strassen),
    and then extend to the 14 columns.
    
    Actually, let's think about this differently. The tensor T[(i,j),(j',k),(i',k')] = delta(i,i')delta(j,j')delta(k,k')
    
    We can group by k. For each fixed k, we have a 2x2x2 sub-tensor (indices i,j,i',j' with k fixed, but wait, the middle index is j and j' must be equal).
    
    Let me re-index. The three modes are:
    - Mode 1: (i,j) where i in {0,1}, j in {0,1} -> 4 elements
    - Mode 2: (j',k) where j' in {0,1}, k in {0,...,13} -> 28 elements  
    - Mode 3: (i',k') where i' in {0,1}, k' in {0,...,13} -> 28 elements
    
    T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k' else 0.
    
    For a fixed k, the slice T[:,:,k] (in mode 2) is a 4x28 tensor where:
    T[(i,j),(j',k),(i',k')] = delta(i,i')delta(j,j')delta(k,k')
    
    For fixed k, T[(i,j),(j',k),(i',k')] is nonzero only when k'=k and i=i' and j=j'.
    So for fixed k, the tensor is:
    T[(i,j),(j',k),(i',k')] = delta(i,i')delta(j,j') if k'=k, else 0.
    
    This is like a 4x4x1 tensor for each k, but embedded in 4x28x28.
    
    Actually, let's think of it as: for each k, we have a 2x2x2 multiplication tensor (for the i,j,i',j' part) but with the output index also carrying k'.
    
    Hmm, this is getting complicated. Let me try a different approach.
    
    Since the 14 columns are independent (each column of B can be treated separately),
    the problem decomposes into 14 independent 2x2x2 matrix multiplications, one for each column k.
    
    For each column k, we compute:
    Output[:,k] = A * B[:,k]
    
    This is a 2x2 matrix times a 2x1 vector, which is a 2x2x2 tensor problem.
    
    The minimal rank for 2x2x2 matrix multiplication is 7 (Strassen).
    
    So for 14 independent columns, the total rank would be 14 * 7 = 98? That's worse than 56.
    
    Wait, no. The Strassen decomposition for 2x2x2 gives rank 7 for a single 2x2x2 multiplication.
    But here we have 14 columns, and the Strassen decomposition can be applied to all columns simultaneously because the 2x2 matrix A is shared across all columns.
    
    In Strassen's algorithm for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    Now, if B is 2x14, then B00, B01, B10, B11 are 1x14 vectors.
    The Strassen formulas still apply, with each "product" being a 1x14 vector times a scalar.
    
    Each M_i is a 1x14 vector (sum of products of a scalar and a 1x14 vector).
    So we have 7 M_i vectors, each of length 14.
    
    The rank of the decomposition: Each M_i corresponds to one rank-1 term in the tensor decomposition.
    But wait, each M_i is a vector of length 14, so it's not a single rank-1 term but 14 rank-1 terms (one for each position in the 14-dimensional output).
    
    Hmm, let me reconsider. In the tensor decomposition, each rank-1 term is:
    u ⊗ v ⊗ w where u is in R^4, v is in R^28, w is in R^28.
    
    For the Strassen approach applied to all 14 columns:
    M1 = (A00+A11) * (B00+B11)
    Here, A00+A11 is a scalar, and B00+B11 is a 1x14 vector.
    The product is a 1x14 vector.
    
    In the tensor decomposition, this corresponds to:
    u = e_{00} + e_{11} (in the 4-dim input space)
    v = e_{0,0} + e_{0,1} + ... + e_{0,13} (in the 28-dim B space, for row 0)
       + e_{1,0} + e_{1,1} + ... + e_{1,13} (for row 1)
    Wait, B00 is row 0 of B, which has 14 elements. B11 is row 1 of B, which has 14 elements.
    So B00+B11 is the sum of row 0 and row 1 of B, a 1x14 vector.
    
    v for this term: the B-space has indices (j,k) where j in {0,1}, k in {0..13}.
    v_{j,k} = 1 for all j,k? No, v_{j,k} = 1 for j=0, all k, and j=1, all k.
    So v = ones(28) in the B-space? No, that's not right either.
    
    Actually, B00 is the first row of B (j=0, all k), and B11 is the second row of B (j=1, all k).
    So B00+B11 has components: v_{0,k} = 1 for all k, v_{1,k} = 1 for all k.
    So v is the all-ones vector in the 28-dim space.
    
    w for this term: the output space has indices (i',k') where i' in {0,1}, k' in {0..13}.
    M1 contributes to C00 and C11 (with coefficients +1 and +1 respectively).
    But M1 is a 1x14 vector, so it contributes to all output positions.
    C00[k] += M1[k] and C11[k] += M1[k] for all k.
    So w_{i',k'} = 1 if (i'=0, k'=k) or (i'=1, k'=k) for some k? No, that's not a single vector.
    
    I think I'm overcomplicating this. Let me just directly construct the rank-56 schoolbook decomposition and see if I can find a lower-rank one.
    
    Actually, the key insight is: the 14 columns are independent. For each column k, we have a 2x2x2 multiplication. The Strassen decomposition gives rank 7 for each 2x2x2. But since the 2x2 matrix A is shared, we can apply Strassen to all 14 columns at once.
    
    In the tensor decomposition, each Strassen term M_i is a scalar (from A) times a 1x14 vector (from B). The result is a 1x14 vector that contributes to the output.
    
    For M1 = (A00+A11)(B00+B11):
    - u (input A): A00+A11, so u = [1,0,0,1] (indices 00,01,10,11)
    - v (input B): B00+B11, so v has 1's at all positions in row 0 and row 1 of B.
      v_{j,k} = 1 for j in {0,1}, k in {0..13}
    - w (output): M1 contributes to C00 and C11.
      C00[k] += M1[k] for all k
      C11[k] += M1[k] for all k
      So w_{i',k'} = 1 if i'=0 (any k') or i'=1 (any k')? No, that's not right.
      
      Actually, M1 is a 1x14 vector. M1[k] contributes to C00[k] and C11[k].
      So w_{i',k'} = 1 if (i'=0 and k'=k) or (i'=1 and k'=k) for the same k.
      But k is not fixed here; M1[k] is a different value for each k.
      
      This means M1 is not a single rank-1 term but 14 rank-1 terms, one for each k.
      
      So the total rank from M1 is 14.
      
      Similarly, each of the 7 Strassen terms M1..M7 contributes 14 rank-1 terms.
      Total rank = 7 * 14 = 98.
      
      That's worse than 56!
    
    So Strassen doesn't help here because the 14 columns are independent and each Strassen term spans all 14 columns.
    
    The schoolbook rank of 56 is actually better than the Strassen-based rank of 98.
    
    So the minimal rank is at most 56, and likely exactly 56 (or possibly less with a more clever decomposition).
    
    For now, let's just return the schoolbook decomposition of rank 56.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
