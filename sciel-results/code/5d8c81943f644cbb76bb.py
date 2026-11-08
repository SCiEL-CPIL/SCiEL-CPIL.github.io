"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix
    by a 4x6 matrix.
    
    Strategy:
    1. Use the structural idea from Inspiration 2: Split the middle dimension (m=4)
       into 2+2 blocks. This creates 4 independent sub-problems of size 3x2 * 2x6.
    2. For each 3x2 * 2x6 sub-problem, we can use a rank-10 decomposition based on
       Strassen's algorithm for the 2x2 core plus rank-1 terms for the remaining parts.
       However, a simpler and potentially more efficient approach for 3x2 * 2x6 is to
       recognize that this is equivalent to multiplying a 3x2 matrix by a 2x6 matrix.
       
       Let's analyze the 3x2 * 2x6 case:
       A is 3x2, B is 2x6. C is 3x6.
       We can use a decomposition of rank 10? 
       Actually, let's look at the known bounds. 
       For 3x2 * 2x6, the tensor is T_{3,2,6}.
       We can split the 6 columns of B into 3 groups of 2.
       So it's 3 * (3x2 * 2x2).
       We know that 3x2 * 2x2 has rank 6 (since 3x2 * 2x2 is equivalent to 3x2x2 tensor,
       and the rank of 3x2x2 is 6? No, wait.
       
       Let's re-evaluate.
       The standard schoolbook for 3x4*4x6 is rank 72.
       Inspiration 2 suggests splitting m=4 into 2+2.
       This gives 4 blocks of 3x2 * 2x6.
       If we can decompose 3x2 * 2x6 with rank R, total rank is 4R.
       
       What is the rank of 3x2 * 2x6?
       We can further split the 6 in 3x2 * 2x6 into 2+2+2.
       So 3 * (3x2 * 2x2).
       What is the rank of 3x2 * 2x2?
       Let A be 3x2, B be 2x2. C = A*B is 3x2.
       This is a tensor of format 3x2x2.
       The rank of 3x2x2 is 6 (it's the same as 2x2x3, which is known to be 6? 
       Actually, 2x2x2 has rank 7. 2x2x3 has rank? 
       Let's check small cases.
       2x2x2 rank is 7.
       2x2x3: We can use the fact that it's a sum of two 2x2x2 tensors? No.
       
       Let's try a different approach.
       Maybe we can use a known decomposition for 3x4 * 4x6 directly or via a different split.
       
       Alternative: Use the fact that 4 = 2+2 and 6 = 2+2+2.
       Total 3 * 2 * 2 = 12 blocks of 3x2 * 2x2.
       If rank(3x2 * 2x2) = 6, then total rank = 12 * 6 = 72. Same as schoolbook.
       If rank(3x2 * 2x2) < 6, we win.
       
       Is rank(3x2 * 2x2) < 6?
       The tensor is T_{3,2,2}.
       It is known that rank(2x2x2) = 7.
       rank(2x2x3) is not 10 (which would be 2*2*3). 
       Actually, for 2x2xn, the rank is 2n for n>=2? No.
       
       Let's look at the structure of 3x2 * 2x2.
       A = [[a, b], [c, d], [e, f]]
       B = [[g, h], [i, j]]
       C = [[ag+bh, ah+bj], [cg+dh, ch+dj], [eg+fh, eh+fj]]
       
       We can compute this using 6 multiplications?
       Let's try to find a rank 6 decomposition.
       If we just use schoolbook, it's 3*2*2 = 12.
       Wait, 3x2 * 2x2 has 12 entries in the tensor? No, the tensor dimension is 3*2*2 = 12.
       The rank is at most 12.
       Can we do better?
       
       Actually, there is a known result that rank(2x2x2) = 7.
       For 3x2x2, we can view it as 3 copies of 2x2? No.
       
       Let's try to construct a decomposition for 3x2 * 2x2 with rank 6.
       If we can't, maybe rank 7? 12*7 = 84 > 72. Bad.
       So we need rank(3x2 * 2x2) <= 6 to beat 72 with the 12-block split.
       Or use a different split.
       
       Let's try splitting m=4 into 1+1+1+1.
       4 blocks of 3x1 * 1x6.
       Rank(3x1 * 1x6) = 18? No, 3x1 * 1x6 is just 3*1*6 = 18 terms? 
       A is 3x1, B is 1x6. C is 3x6.
       C_ij = A_i * B_j.
       This is rank 18? No, it's rank 1 * 6? 
       Wait, A is a vector of length 3, B is a vector of length 6.
       C = A * B^T. This is a rank 1 matrix!
       So the tensor T_{3,1,6} has rank 1? 
       No, the tensor T is defined such that C_k = sum_r U_kr V_r W_r? 
       No, the tensor is the bilinear map A, B -> C.
       If A is 3x1 and B is 1x6, C is 3x6.
       C_{ij} = A_i B_j.
       This is a rank 1 tensor in the sense that it can be written as a single term?
       U is 3x1, V is 6x1, W is 18x1?
       U_{i, 1} = 1, V_{j, 1} = 1, W_{(i,j), 1} = 1?
       Then C_{ij} = sum_r U_{i,r} V_{j,r} W_{(i,j),r} = 1*1*1 = 1.
       But we need C_{ij} = A_i B_j.
       So U_{i,1} = A_i, V_{j,1} = B_j, W_{(i,j),1} = 1.
       Yes! Rank 1.
       So 4 blocks of rank 1 gives total rank 4? 
       Wait, if we split m=4 into 4 parts of size 1, we have 4 sub-problems.
       Each sub-problem is 3x1 * 1x6.
       Rank is 1 for each?
       Total rank = 4 * 1 = 4?
       That seems too good to be true.
       
       Let's check the indices.
       A is 3x4. B is 4x6.
       Split A into A_0, A_1, A_2, A_3 where A_k is 3x1 (column k).
       Split B into B_0, B_1, B_2, B_3 where B_k is 1x6 (row k).
       C = sum_{k=0}^3 A_k B_k.
       Each A_k B_k is a 3x6 matrix of rank 1.
       The tensor for A_k B_k has rank 1.
       So the total tensor is a sum of 4 rank-1 tensors.
       Therefore, the rank is at most 4.
       
       Is this correct?
       The tensor T is defined by C_{ij} = sum_{k} A_{ik} B_{kj}.
       If we decompose T as sum_{r} U_r \otimes V_r \otimes W_r,
       then C_{ij} = sum_r U_{ir} V_{jr} W_{ijr}? No.
       The convention is U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
       Wait, the task says:
       U shape (n*m, rank)
       V shape (m*p, rank)
       W shape (p*n, rank)
       And C is computed as sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]?
       No, that's not how matrix multiplication works with these shapes.
       
       Standard tensor decomposition for matrix multiplication:
       C_{ik} = sum_{j=1}^m A_{ij} B_{jk}.
       We want C_{ik} = sum_{r=1}^R U_{ir, j?} ...
       
       Let's look at the indices in the seed code:
       U[i * m + j][column] = 1
       V[j * p + k][column] = 1
       W[k * n + i][column] = 1
       
       This implies the term for (i,j,k) is:
       U_{(i,j), r} * V_{(j,k), r} * W_{(k,i), r}
       
       So C_{ik} = sum_{j=1}^m A_{ij} B_{jk} = sum_{j=1}^m sum_{r} U_{(i,j),r} V_{(j,k),r} W_{(k,i),r} A_{ij} B_{jk}?
       No. The decomposition is exact for the multiplication tensor.
       This means that for any A, B, the product C = AB is given by:
       C_{ik} = sum_{r=1}^R (sum_{j} U_{(i,j),r} A_{ij}) (sum_{j} V_{(j,k),r} B_{jk}) W_{(k,i),r}
       
       Wait, if we use the schoolbook:
       For each (i,j,k), we have a column r.
       U_{(i,j), r} = 1, V_{(j,k), r} = 1, W_{(k,i), r} = 1.
       Then sum_r U_{(i,j'),r} A_{ij'} V_{(j',k'),r} B_{j'k'} W_{(k',i'),r}
       = sum_{j} A_{ij} B_{jk} * 1 * 1 * 1 (where r corresponds to i,j,k)
       = sum_{j} A_{ij} B_{jk}.
       Yes, this works.
       
       Now, back to the split m=4 into 1+1+1+1.
       We have 4 sub-problems.
       Sub-problem k (k=0..3):
       A_k is column k of A (3x1).
       B_k is row k of B (1x6).
       The product is A_k B_k, which is 3x6.
       The tensor for this sub-problem has rank 1?
       Let's check.
       We need to represent the map (A_k, B_k) -> A_k B_k.
       A_k is a vector of length 3. B_k is a vector of length 6.
       The result is a 3x6 matrix.
       Entry (i, l) is A_{ik} B_{kl}.
       Using the formula:
       C_{il} = sum_r U_{(i,k),r} V_{(k,l),r} W_{(l,i),r} (A_{ik} B_{kl})?
       No, the U, V, W are fixed coefficients.
       The input is A and B.
       The term is U_{(i,j),r} A_{ij} * V_{(j,k),r} B_{jk} * W_{(k,i),r}.
       
       For the sub-problem where only j=k is non-zero (in the sense that we isolate the j-th term of the sum):
       We set U_{(i,j),r} = 1, V_{(j,l),r} = 1, W_{(l,i),r} = 1.
       Then the contribution is A_{ij} B_{jl} * 1 * 1 * 1 = A_{ij} B_{jl}.
       This is exactly the term for the j-th column/row.
       
       So, if we use 4 such terms (one for each j=0..3), we get the full sum.
       Total rank = 4.
       
       Wait, is this valid?
       The rank is the number of terms R.
       Here R=4.
       But the shapes of U, V, W must be consistent.
       U is 12x4. V is 24x4. W is 18x4.
       
       Let's verify.
       For r=0 (corresponding to j=0):
       U_{(i,0), 0} = 1 for all i. (Indices 0, 1, 2 in U)
       V_{(0,l), 0} = 1 for all l. (Indices 0, 1, 2, 3, 4, 5 in V)
       W_{(l,i), 0} = 1 for all l, i. (Indices 0..17 in W)
       
       For r=1 (corresponding to j=1):
       U_{(i,1), 1} = 1 for all i. (Indices 4, 5, 6 in U)
       V_{(1,l), 1} = 1 for all l. (Indices 6, 7, 8, 9, 10, 11 in V)
       W_{(l,i), 1} = 1 for all l, i.
       
       And so on.
       
       Let's check the product:
       C_{il} = sum_{r=0}^3 (sum_{j} U_{(i,j),r} A_{ij}) (sum_{j'} V_{(j',l),r} B_{j'l}) W_{(l,i),r}
       
       For r=0:
       sum_j U_{(i,j),0} A_{ij} = A_{i0} (since U_{(i,0),0}=1, others 0)
       sum_{j'} V_{(j',l),0} B_{j'l} = B_{0l} (since V_{(0,l),0}=1, others 0)
       W_{(l,i),0} = 1
       Term = A_{i0} B_{0l}
       
       Summing over r=0..3:
       C_{il} = sum_{r=0}^3 A_{ir} B_{rl}
       This is exactly (AB)_{il}.
       
       So the rank is 4!
       
       This is a huge improvement over 72.
       
       Let's double check if there are any constraints I missed.
       "integer/half-integer entries" - 1 is an integer.
       "exact tensor decomposition" - Yes.
       
       So the code is very simple.
       
       U: 12x4
       V: 24x4
       W: 18x4
       
       For r in 0..3:
       For i in 0..2:
           U[i*4 + r][r] = 1
       For l in 0..5:
           V[r*6 + l][r] = 1
       For l in 0..5:
           For i in 0..2:
               W[l*3 + i][r] = 1
               
       Wait, V index is j*p + k. Here j=r, k=l.
       W index is k*n + i. Here k=l, i=i.
       
       Let's write the code.
    """
    n, m, p = DIMENSIONS
    rank = m  # 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(m):
        for i in range(n):
            U[i * m + r][r] = 1
        for k in range(p):
            V[r * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
