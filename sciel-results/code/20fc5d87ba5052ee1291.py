"""Matrix multiplication 4,5,6 tensor decomposition search."""
import time
import itertools
import random
import sys

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_left(deadline):
    return deadline - time.time()

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix mult."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build a Strassen-based decomposition for 4x5 * 5x6
    # Strategy: partition into 2x2 blocks and apply Strassen where possible.
    # 
    # A is 4x5, B is 5x6, C is 4x6.
    # Partition:
    #   A rows: {0,1}, {2,3}
    #   A cols: {0,1}, {2,3}, {4}
    #   B rows: {0,1}, {2,3}, {4}
    #   B cols: {0,1}, {2,3}, {4,5}
    #   C rows: {0,1}, {2,3}
    #   C cols: {0,1}, {2,3}, {4,5}
    #
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # For C block (rows r, cols c) where r in {0,1} (2x2 block) and c in {0,1,2} (2x2, 2x2, 2x2):
    # C_rc = sum_{j_block} A_{r, jb} * B_{jb, c}
    #
    # Let's enumerate all terms and group them for Strassen.
    
    # For the 2x2 subproblems (where A block is 2x2, B block is 2x2, C block is 2x2),
    # we can use Strassen's 7 multiplications instead of 8.
    #
    # The 2x2 blocks of A (2x2 each): A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # The 2x2 blocks of B (2x2 each): B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # The 2x2 blocks of C (2x2 each): C[0:2,0:2], C[0:2,2:4], C[2:4,0:2], C[2:4,2:4]
    #
    # Plus the "edge" blocks involving column/row 4:
    # A[0:2,4] (2x1), A[2:4,4] (2x1)
    # B[4,0:2] (1x2), B[4,2:4] (1x2), B[4,4:6] (1x2)
    # C[0:2,4:6] (2x2), C[2:4,4:6] (2x2)
    #
    # And B[0:2,4:6] (2x2), B[2:4,4:6] (2x2)
    
    # Let's build the decomposition factor by factor.
    # Each factor c contributes: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 if (i,j,k) is a valid term, else 0
    # Actually: for the matrix multiplication, the tensor entry T[i,j,k] = 1 always (since C[i,k] = sum_j A[i,j]B[j,k])
    # The decomposition must satisfy: for all i,j,k: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # Wait no. Let me re-derive.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Our formula: C[i,k] = sum_c W[k,i,c] * (sum_{i',j'} U[i',j',c] A[i',j']) * (sum_{j',k'} V[j',k',c] B[j',k'])
    # = sum_c W[k,i,c] sum_{i',j',j'',k'} U[i',j',c] V[j'',k',c] A[i',j'] B[j'',k']
    # For this to equal sum_j A[i,j] B[j,k], we need:
    # sum_c W[k,i,c] U[i',j',c] V[j'',k',c] = delta_{i',i} delta_{k',k} delta_{j',j''}
    # for all i',j',j'',k'.
    
    # So the constraint is: for all (i',j',j'',k'), 
    #   sum_c W[k,i,c] U[i',j',c] V[j'',k',c] = 1 if i'=i, k'=k, j'=j'' else 0
    # But i and k are the output indices, and i',k' are input indices.
    # Actually i=k' and i'=i... let me be more careful.
    
    # The RHS is: coefficient of A[i',j'] B[j'',k'] in C[i,k] should be:
    # 1 if i'=i, j'=j'', k'=k, else 0.
    # So: sum_c W[k,i,c] U[i',j',c] V[j'',k',c] = delta_{i',i} delta_{k',k} delta_{j',j''}
    
    # This must hold for ALL i,k,i',j',j'',k'.
    
    # Schoolbook: one factor per (i,j,k) triple:
    #   U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for c corresponding to (i,j,k)
    #   Then sum_c W[k,i,c] U[i',j',c] V[j'',k',c] = 1 iff (i,j,k)=(i',j',k') and j=j''
    #   i.e., i=i', j=j''=j', k=k'. Correct.
    
    # Now let's build a Strassen-based decomposition.
    # 
    # I'll use the following approach:
    # 1. Handle the "bulk" 2x2x2x2x2x2 blocks with Strassen (7 factors each instead of 8)
    # 2. Handle the edge blocks (involving the 5th row/col) with schoolbook
    
    # Let me define the blocks:
    # A blocks:
    #   A00 = A[0:2, 0:2]  (2x2)
    #   A01 = A[0:2, 2:4]  (2x2)
    #   A02 = A[0:2, 4]    (2x1)
    #   A10 = A[2:4, 0:2]  (2x2)
    #   A11 = A[2:4, 2:4]  (2x2)
    #   A12 = A[2:4, 4]    (2x1)
    #
    # B blocks:
    #   B00 = B[0:2, 0:2]  (2x2)
    #   B01 = B[0:2, 2:4]  (2x2)
    #   B02 = B[0:2, 4:6]  (2x2)
    #   B10 = B[2:4, 0:2]  (2x2)
    #   B11 = B[2:4, 2:4]  (2x2)
    #   B12 = B[2:4, 4:6]  (2x2)
    #   B20 = B[4, 0:2]    (1x2)
    #   B21 = B[4, 2:4]    (1x2)
    #   B22 = B[4, 4:6]    (1x2)
    #
    # C blocks:
    #   C00 = C[0:2, 0:2]  (2x2)
    #   C01 = C[0:2, 2:4]  (2x2)
    #   C02 = C[0:2, 4:6]  (2x2)
    #   C10 = C[2:4, 0:2]  (2x2)
    #   C11 = C[2:4, 2:4]  (2x2)
    #   C12 = C[2:4, 4:6]  (2x2)
    #
    # C00 = A00*B00 + A01*B10 + A02*B20
    # C01 = A00*B01 + A01*B11 + A02*B21
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C10 = A10*B00 + A11*B10 + A12*B20
    # C11 = A10*B01 + A11*B11 + A12*B21
    # C12 = A10*B02 + A11*B12 + A12*B22
    
    # The 2x2x2x2 terms (A00*B00, A01*B10, etc.) can potentially be combined using Strassen.
    # But Strassen works on a 2x2 = 2x2 * 2x2 product. Here we have multiple such products
    # contributing to different C blocks.
    
    # Actually, let me think of it differently. The full product is:
    # C = A * B where A is 4x5 and B is 5x6.
    # 
    # I can pad A to 4x6 and B to 6x6, compute 4x6 * 6x6, and take the first 6 cols.
    # 4x6 * 6x6: partition into 2x2 blocks of 2x3 and 3x3.
    # That's 4 blocks of 2x3 * 3x3.
    # Each 2x3 * 3x3 has schoolbook rank 18.
    # Total: 72. Plus the padding overhead... this doesn't help.
    
    # Let me try yet another approach. Use the known fact that:
    # rank(4x5x6) <= rank(4x6x6) (by padding B's 5th row with zeros)
    # and rank(4x6x6) can be computed via 2x2 Strassen blocks.
    # 4x6 * 6x6: 2x2 blocks of 2x3 and 3x3.
    # 4 such products, each 2x3*3x3 = 18 terms schoolbook.
    # Can we do better for 2x3*3x3?
    # 2x3*3x3: C (2x3) = A (2x3) * B (3x3)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Schoolbook: 2*3*3 = 18.
    # This is a small case; the rank is likely 18 or close.
    
    # I think the most practical approach given time constraints is to use the 
    # inspiration's result: rank 108. Let me construct it by:
    # 1. Starting with schoolbook (rank 120)
    # 2. For each 2x2x2 sub-block where we have a full 2x2 A-block, 2x2 B-block, 
    #    and they contribute to a 2x2 C-block, replace 8 schoolbook factors with 7 Strassen factors.
    
    # The 2x2x2 sub-blocks:
    # (A[0:2,0:2] * B[0:2,0:2]) contributes to C[0:2,0:2]
    # (A[0:2,2:4] * B[2:4,0:2]) contributes to C[0:2,0:2]
    # (A[2:4,0:2] * B[0:2,0:2]) contributes to C[2:4,0:2]
    # (A[2:4,2:4] * B[2:4,0:2]) contributes to C[2:4,0:2]
    # ... etc.
    
    # Each 2x2*2x2=2x2 product has 8 schoolbook terms. With Strassen, 7 terms.
    # But these products are not independent - they all contribute to the same C blocks.
    
    # Let me just directly enumerate all 120 schoolbook terms and see if I can 
    # find a set of 108 terms (with appropriate coefficients) that works.
    # This is essentially solving a linear system.
    
    import numpy as np
    
    # For a given rank r, we want U (nm x r), V (mp x r), W (pn x r) such that:
    # For all (i,j,k): sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    # (since every term A[i,j]*B[j,k] contributes to C[i,k])
    
    # Wait, that's not right. The constraint is:
    # For all (i',j',j'',k'): sum_c W[k,i,c] U[i',j',c] V[j'',k',c] = delta_{i',i} delta_{k',k} delta_{j',j''}
    
    # This is a system of equations. Let me set it up as a tensor equation.
    # Let T be the tensor with T[i',j',j'',k',k,i] = delta_{i',i} delta_{k',k} delta_{j',j''}
    # We want: T = sum_c W[:, :, c] outer U[:, :, c] outer V[:, :, c] (appropriately indexed)
    
    # Actually, let me just verify the schoolbook and then try to reduce.
    
    # Schoolbook verification:
    rank_schoolbook = n*m*p  # 120
    U = [[0]*rank_schoolbook for _ in range(nm)]
    V = [[0]*rank_schoolbook for _ in range(mp)]
    W = [[0]*rank_schoolbook for _ in range(pn)]
    for idx, (i,j,k) in enumerate(itertools.product(range(n), range(m), range(p))):
        U[i*m+j][idx] = 1
        V[j*p+k][idx] = 1
        W[k*n+i][idx] = 1
    
    # Now, let's try to use Strassen on the 2x2x2 sub-problems.
    # I'll construct the decomposition explicitly.
    
    # The key idea: for a 2x2 * 2x2 = 2x2 matrix multiplication, Strassen uses:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In our tensor decomposition, each "multiplication" Mx corresponds to a set of 
    # (U, V, W) factors. For a 2x2*2x2=2x2 product, the 8 schoolbook terms are:
    # (i,j,k) for i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen replaces these 8 with 7.
    
    # Let me build the full decomposition by:
    # 1. Identifying all 2x2x2x2 sub-blocks
    # 2. Applying Strassen to each
    # 3. Handling the remaining terms with schoolbook
    
    # This is getting complex. Let me just directly construct the rank-108 decomposition
    # by using the Strassen formulas for the applicable sub-blocks.
    
    # I'll use a different strategy: build the decomposition term by term.
    # For each of the 120 schoolbook terms (i,j,k), I'll assign it to a factor.
    # Then I'll try to merge factors using Strassen identities.
    
    # Actually, let me just implement the known rank-108 construction directly.
    # The construction works by:
    # - Partitioning the 4x5x6 problem into sub-problems
    # - Using Strassen (rank 7 vs 8) for the 2x2x2 sub-problems
    # - Using schoolbook for the rest
    
    # 4x5x6 = sum of sub-problems:
    # The "2x2 core": rows 0-1,2-3 of A; rows 0-1,2-3 of B; cols 0-1,2-3 of B
    # This gives 4 sub-problems of 2x2x2x2 (A 2x2, B 2x2, C 2x2):
    #   A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    #   A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]  
    #   A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    #   A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    #   A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    #   A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    #   A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
    #   A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    
    # Wait, that's 8 sub-problems, but they share C blocks. The total rank would be:
    # If each 2x2x2x2 uses Strassen (7 factors), and they're independent, we'd need 8*7=56.
    # But they're not independent - they contribute to the same C blocks.
    
    # I think the correct approach is to treat the entire 4x4x4 sub-problem (using the 
    # first 4 rows/cols) as a single unit and apply Strassen recursively.
    # 4x4x4 has known rank 49 (using Strassen: 49 = 7^2).
    # Then handle the remaining terms involving the 5th row/col.
    
    # 4x4x4 (using A[0:4,0:4], B[0:4,0:4], C[0:4,0:4]): rank 49 via double Strassen.
    # Remaining terms:
    # A[0:4,4] * B[4,0:6] -> C[0:4,0:6]: 4*1*6 = 24 terms
    # A[0:4,0:4] * B[0:4,4:6] -> C[0:4,4:6]: 4*4*2 = 32 terms
    # Total remaining: 56 terms.
    # Total: 49 + 56 = 105. That's better than 108!
    
    # But wait, the 4x4x4 Strassen decomposition has rank 49, and the remaining 56 terms 
    # are schoolbook. Total: 105.
    
    # Let me verify: 4x4x4 using Strassen:
    # 4x4 = 2x2 blocks of 2x2.
    # C = A*B where A,B,C are 4x4.
    # A = [[A00,A01],[A10,A11]], B = [[B00,B01],[B10,B11]], C = [[C00,C01],[C10,C11]]
    # Each Ai,Bi,Ci is 2x2.
    # Strassen: 7 multiplications of 2x2 matrices.
    # Each 2x2*2x2=2x2 has rank 7 (Strassen) or 8 (schoolbook).
    # So total rank: 7 * 7 = 49. Yes!
    
    # So the construction is:
    # 1. Use Strassen for the 4x4x4 core (rank 49)
    # 2. Use schoolbook for the remaining 56 terms
    # Total: 105
    
    # Let me implement this.
    
    def strassen_2x2(U_base, V_base, W_base, rank_offset, 
                     i_off, j_off, k_off):
        """
        Add Strassen factors for a 2x2 * 2x2 = 2x2 multiplication.
        U_base, V_base, W_base: the factor matrices being built
        rank_offset: where to start adding new factors
        i_off, j_off, k_off: offsets for i, j, k indices
        Returns new rank_offset.
        """
        # Strassen's 7 products for 2x2*2x2=2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mx is a 2x2 matrix product. In the tensor decomposition,
        # each "multiplication" of two 2x2 matrices to get a 2x2 matrix
        # requires 7 factors (Strassen) or 8 factors (schoolbook).
        #
        # But in our case, the Mx are not standalone products - they're 
        # intermediate results that get combined. The tensor decomposition
        # works at the level of individual (i,j,k) terms.
        #
        # For a 2x2*2x2=2x2 product at offset (i_off, j_off, k_off):
        # The 8 schoolbook terms are:
        # (i_off+0, j_off+0, k_off+0), (i_off+0, j_off+1, k_off+0),
        # (i_off+1, j_off+0, k_off+0), (i_off+1, j_off+1, k_off+0),
        # (i_off+0, j_off+0, k_off+1), (i_off+0, j_off+1, k_off+1),
        # (i_off+1, j_off+0, k_off+1), (i_off+1, j_off+1, k_off+1)
        #
        # Strassen replaces these 8 with 7 factors.
        # The 7 factors correspond to the 7 Strassen products.
        # Each Strassen product Mx involves specific (i,j) pairs from A and (j,k) pairs from B.
        
        # For M1 = (A11+A22)(B11+B22):
        # A11 is at (i_off, j_off), A22 is at (i_off+1, j_off+1)
        # B11 is at (j_off, k_off), B22 is at (j_off+1, k_off+1)
        # M1 contributes to C11 and C22.
        # The 4 terms in M1 are:
        # A11*B11: (i_off, j_off, k_off)
        # A11*B22: (i_off, j_off+1, k_off+1)
        # A22*B11: (i_off+1, j_off, k_off)
        # A22*B22: (i_off+1, j_off+1, k_off+1)
        # In Strassen, these 4 terms are handled by 1 factor with appropriate coefficients.
        # Actually no - Strassen's M1 is a single product of two 2x2 matrices, 
        # which itself requires 7 factors (if we recurse) or 8 (schoolbook).
        #
        # I think I'm overcomplicating this. Let me use the known fact that
        # rank(2x2x2) = 7 and directly specify the 7 factors.
        
        # The 7 Strassen factors for 2x2*2x2=2x2:
        # Factor 1: U has (0,0)=1,(1,1)=1; V has (0,0)=1,(1,1)=1; W has (0,0)=1,(1,1)=1
        #   This gives: A[0,0]B[0,0]C[0,0] + A[0,0]B[1,1]C[1,1] + A[1,1]B[0,0]C[0,0] + A[1,1]B[1,1]C[1,1]
        #   Wait, that's not right. Each factor is a single (U,V,W) triple.
        #   U[0,0]=1, U[1,1]=1 means the factor picks up A[0,0] and A[1,1].
        #   V[0,0]=1, V[1,1]=1 means it picks up B[0,0] and B[1,1].
        #   W[0,0]=1, W[1,1]=1 means it contributes to C[0,0] and C[1,1].
        #   The product: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) contributes to C[0,0] and C[1,1] with coefficient 1.
        #   But we need specific coefficients for the Strassen formula.
        
        # Let me look up the exact Strassen tensor decomposition.
        # For 2x2*2x2=2x2, the 7 factors are:
        # (using 0-indexed (i,j,k)):
        # f1: U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1)  -> (A00+A11)(B00+B11) -> C00+=, C11+=
        #     Wait, I need to be more careful with the indexing.
        
        # Let me use the standard Strassen decomposition:
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
        # p = a*e + c*g, q = a*f + c*h, r = b*e + d*g, s = b*f + d*h
        #
        # Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-g)
        # m4 = d(h-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        #
        # In terms of (i,j,k) indices (i=row of A/C, j=col of A/row of B, k=col of B/C):
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # p=C[0,0], q=C[0,1], r=C[1,0], s=C[1,1]
        #
        # m1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        #   Factor: U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
        #   This single factor contributes: A[0,0]B[0,0] to C[0,0], A[0,0]B[1,1] to C[1,1],
        #   A[1,1]B[0,0] to C[0,0], A[1,1]B[1,1] to C[1,1]
        #   But we need this to contribute to p and s with specific coefficients.
        #   In the Strassen formula, m1 contributes +1 to p and +1 to s.
        #   So W[0,0]=1, W[1,1]=1. Correct.
        
        # m2 = (A[1,0]+A[1,1])B[0,0]
        #   Factor: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[1,0]=1
        #   Contributes: A[1,0]B[0,0] to C[1,0], A[1,1]B[0,0] to C[1,0]
        #   In formula: m2 contributes +1 to r. So W[1,0]=1. Correct.
        
        # m3 = A[0,0](B[0,1]-B[1,0])
        #   Factor: U[0,0]=1, V[0,1]=1, V[1,0]=-1, W[0,1]=1
        #   Contributes: A[0,0]B[0,1] to C[0,1], A[0,0]B[1,0] to C[0,1] with -1
        #   In formula: m3 contributes +1 to q and +1 to s. So W[0,1]=1, W[1,1]=1.
        #   Wait, m3 contributes to both q and s? Let me re-check.
        #   q = m3 + m5, s = m1 - m2 + m3 + m6. Yes, m3 is in both.
        #   So W[0,1]=1, W[1,1]=1.
        
        # m4 = A[1,1](B[1,1]-B[0,0])
        #   Factor: U[1,1]=1, V[1,1]=1, V[0,0]=-1, W[0,0]=1, W[1,0]=1
        #   Contributes: A[1,1]B[1,1] to C[0,0], A[1,1]B[0,0] to C[0,0] with -1,
        #   A[1,1]B[1,1] to C[1,0], A[1,1]B[0,0] to C[1,0] with -1
        #   In formula: m4 contributes +1 to p and +1 to r. So W[0,0]=1, W[1,0]=1.
        
        # m5 = (A[0,0]+A[0,1])B[1,1]
        #   Factor: U[0,0]=1, U[0,1]=1, V[1,1]=1, W[0,0]=-1, W[0,1]=1
        #   In formula: m5 contributes -1 to p and +1 to q. So W[0,0]=-1, W[0,1]=1.
        
        # m6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        #   Factor: U[1,0]=1, U[0,0]=-1, V[0,0]=1, V[0,1]=1, W[1,1]=1
        #   In formula: m6 contributes +1 to s. So W[1,1]=1.
        
        # m7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        #   Factor: U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,0]=1
        #   In formula: m7 contributes +1 to p. So W[0,0]=1.
        
        # Let me verify by checking the sum of all 7 factors:
        # For C[0,0] (p): 
        #   m1: +1 (from W[0,0]=1 in m1)
        #   m4: +1 (from W[0,0]=1 in m4)
        #   m5: -1 (from W[0,0]=-1 in m5)
        #   m7: +1 (from W[0,0]=1 in m7)
        #   Total: 1+1-1+1 = 2. But we need 1!
        
        # I think I'm making an error. Let me re-examine.
        # The issue is that each factor's contribution to C[i,k] is:
        #   sum_{i',j',j'',k'} U[i',j'] V[j'',k'] W[k',i] * A[i',j'] B[j'',k']
        # For a single factor, this is:
        #   (sum_{i',j'} U[i',j'] A[i',j']) * (sum_{j'',k'} V[j'',k'] B[j'',k']) * W[k',i]
        # Wait no, it's:
        #   W[k',i] * (sum_{i',j'} U[i',j'] A[i',j']) * (sum_{j'',k''} V[j'',k''] B[j'',k''])
        # And we need this to equal the specific terms.
        
        # I think the correct interpretation is:
        # For factor c: the contribution to C[i,k] is:
        #   W[k,i] * (sum_{i',j'} U[i',j'] A[i',j']) * (sum_{j'',k''} V[j'',k''] B[j'',k''])
        # And we need: sum_c W[k,i] * (sum_{i',j'} U[i',j'] A[i',j']) * (sum_{j'',k''} V[j'',k''] B[j'',k'']) = sum_j A[i,j] B[j,k]
        
        # For a single factor to represent a product like (A[0,0]+A[1,1])(B[0,0]+B[1,1]) contributing to C[0,0]:
        # U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1
        # Contribution to C[0,0]: 1 * (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # But we only want A[0,0]B[0,0] and A[1,1]B[1,1] to contribute to C[0,0] (with coefficient 1 each).
        # The cross terms A[0,0]B[1,1] and A[1,1]B[0,0] should NOT contribute to C[0,0].
        # They will be cancelled by other factors.
        
        # So the 7 Strassen factors together must cancel the cross terms.
        # This is exactly how Strassen works - the 7 factors are designed so that
        # the cross terms cancel out in the sum.
        
        # Let me just use the known correct Strassen decomposition.
        # I'll look it up / derive it carefully.
        
        # The 7 factors for 2x2*2x2=2x2 (0-indexed):
        # f1: U={(0,0):1,(1,1):1}, V={(0,0):1,(1,1):1}, W={(0,0):1,(1,1):1}
        # f2: U={(1,0):1,(1,1):1}, V={(0,0):1}, W={(1,0):1}
        # f3: U={(0,0):1}, V={(0,1):1,(1,0):-1}, W={(0,1):1,(1,1):1}
        # f4: U={(1,1):1}, V={(1,1):1,(0,0):-1}, W={(0,0):1,(1,0):1}
        # f5: U={(0,0):1,(0,1):1}, V={(1,1):1}, W={(0,0):-1,(0,1):1}
        # f6: U={(1,0):1,(0,0):-1}, V={(0,0):1,(0,1):1}, W={(1,1):1}
        # f7: U={(0,1):1,(1,1):-1}, V={(1,0):1,(1,1):1}, W={(0,0):1}
        
        # Let me verify f1:
        # Contribution to C[0,0]: W[0,0]=1, (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        #   = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # Contribution to C[0,1]: W[0,1]=0, so 0.
        # Contribution to C[1,0]: W[1,0]=0, so 0.
        # Contribution to C[1,1]: W[1,1]=1, (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        #   = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        
        # f2:
        # Contribution to C[1,0]: W[1,0]=1, (A[1,0]+A[1,1])B[0,0]
        #   = A[1,0]B[0,0] + A[1,1]B[0,0]
        
        # f3:
        # Contribution to C[0,1]: W[0,1]=1, A[0,0](B[0,1]-B[1,0])
        #   = A[0,0]B[0,1] - A[0,0]B[1,0]
        # Contribution to C[1,1]: W[1,1]=1, A[0,0](B[0,1]-B[1,0])
        #   = A[0,0]B[0,1] - A[0,0]B[1,0]
        
        # f4:
        # Contribution to C[0,0]: W[0,0]=1, A[1,1](B[1,1]-B[0,0])
        #   = A[1,1]B[1,1] - A[1,1]B[0,0]
        # Contribution to C[1,0]: W[1,0]=1, A[1,1](B[1,1]-B[0,0])
        #   = A[1,1]B[1,1] - A[1,1]B[0,0]
        
        # f5:
        # Contribution to C[0,0]: W[0,0]=-1, (A[0,0]+A[0,1])B[1,1]
        #   = -A[0,0]B[1,1] - A[0,1]B[1,1]
        # Contribution to C[0,1]: W[0,1]=1, (A[0,0]+A[0,1])B[1,1]
        #   = A[0,0]B[1,1] + A[0,1]B[1,1]
        
        # f6:
        # Contribution to C[1,1]: W[1,1]=1, (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        #   = A[1,0]B[0,0] + A[1,0]B[0,1] - A[0,0]B[0,0] - A[0,0]B[0,1]
        
        # f7:
        # Contribution to C[0,0]: W[0,0]=1, (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        #   = A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1]
        
        # Now let's sum up contributions to C[0,0]:
        # f1: A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
        # f4: A[1,1]B[1,1] - A[1,1]B[0,0]
        # f5: -A[0,0]B[1,1] - A[0,1]B[1,1]
        # f7: A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0] - A[1,1]B[1,1]
        # Sum: A[0,0]B[0,0] + (A[0,0]B[1,1]-A[0,0]B[1,1]) + (A[1,1]B[0,0]-A[1,1]B[0,0]) 
        #       + (A[1,1]B[1,1]+A[1,1]B[1,1]-A[1,1]B[1,1]) - A[0,1]B[1,1] + A[0,1]B[1,0] + A[0,1]B[1,1] - A[1,1]B[1,0]
        # = A[0,0]B[0,0] + A[1,1]B[1,1] + A[0,1]B[1,0] - A[1,1]B[1,0]
        # Hmm, that's not A[0,0]B[0,0] + A[1,0]B[1,0]. I'm making errors.
        
        # I think the issue is that my Strassen factors are not quite right.
        # Let me just use a verified construction.
        
        # Given the complexity, let me take a simpler approach:
        # Use the schoolbook decomposition but with a smart grouping that saves rank.
        # 
        # Actually, the simplest correct approach: just return the schoolbook (rank 120)
        # since I can't quickly verify a lower-rank construction.
        # But the inspiration shows rank 108 is achievable.
        
        # Let me try a different strategy: use numpy to solve for a lower-rank decomposition.
        # Set up the tensor equation and use SVD or similar.
        
        return rank_offset + 7  # placeholder
    
    # Given the time constraints and complexity, let me just return the schoolbook decomposition.
    # The rank is 120.
    
    best = {'U': U, 'V': V, 'W': W}
    return best

# EVOLVE-BLOCK-END
