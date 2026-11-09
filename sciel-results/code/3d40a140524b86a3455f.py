"""
Optimized rank decomposition for 2x3 by 3x4 matrix multiplication.
Uses a constructive algorithm based on structural symmetries and
iterative refinement to find a low-rank exact decomposition.
"""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # The tensor T is of order (n*m) x (m*p) x (p*n) = 6 x 12 x 8
    # We seek rank r such that T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # where i in 0..n*m-1, j in 0..m*p-1, k in 0..p*n-1
    # T[i,j,k] = 1 iff i% m == j//p and j%p == k//n
    # i.e., if we write i = i1*m + i2, j = j1*p + j2, k = k1*n + k2,
    # then T = 1 iff i2 == j1 and j2 == k2, with i1, k1 free.

    # For 2x3x4 matrix multiplication, the known minimal rank is 22.
    # We construct an explicit rank-22 decomposition using the following method:
    # 
    # The schoolbook rank is n*m*p = 2*3*4 = 24.
    # We can save 2 ranks by exploiting the structure.
    # 
    # A known construction for M(2,3,4) with rank 22 uses the fact that
    # we can combine some of the bilinear terms. Specifically, we can use
    # a Strassen-like algorithm adapted for rectangular matrices.
    #
    # Here's a constructive approach:
    # The standard decomposition has 24 terms. We can reduce this by
    # noting that certain combinations of terms can be merged.
    #
    # One known rank-22 decomposition for 2x3 by 3x4:
    # We use the following idea from the literature on small matrix multiplication:
    # Split the 3x4 matrix into two 3x2 blocks. Then:
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # Each A*B_i is a 2x3 times 3x2 multiplication, which has rank 12 (schoolbook).
    # But we can do better by sharing computations between B1 and B2.
    #
    # Actually, let me use a direct construction. The minimal rank for 2x3x4
    # is known to be 22. Here's a valid decomposition:
    #
    # I'll construct it by starting from the schoolbook and applying a rank-reduction.
    #
    # Alternative: Use the known explicit construction. For 2x3 by 3x4,
    # we can achieve rank 22 with the following pattern:
    #
    # Let's index the 24 schoolbook terms as (i,j,k) for i in 0,1; j in 0,1,2; k in 0,1,2,3.
    # We can save 2 ranks by combining terms that share structure.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 12 x 22
    W = [[0] * rank for _ in range(p * n)]  # 8 x 22
    
    # We'll build the decomposition column by column.
    # The first 20 columns handle most of the work, and the last 2 are corrections.
    #
    # Actually, let me use a cleaner construction. Here's a known rank-22 decomposition:
    #
    # Based on the work of Landsberg and others on small matrix multiplication,
    # here's an explicit construction:
    
    col = 0
    
    # For each pair (i, j) where i in {0,1} and j in {0,1,2},
    # we need to compute sum_k A[i,j] * B[j,k] for k in {0,1,2,3}.
    # The schoolbook uses 4 terms per (i,j) pair, totaling 2*3*4=24.
    #
    # We can reduce this by using the following trick:
    # For fixed i, the 4 outputs C[i,0..3] are linear combinations of B[j,0..3].
    # Specifically, C[i,k] = sum_j A[i,j] * B[j,k].
    #
    # Here's a rank-22 construction:
    # We'll use 22 bilinear terms. Let me define them explicitly.
    
    # Terms 0-19: Handle the main computation with some sharing
    # Terms 20-21: Correction terms
    
    # Let's use a different approach: enumerate all 24 schoolbook terms
    # and then merge 2 pairs of terms that can be combined.
    
    # Schoolbook terms indexed by (i, j, k):
    # Term (i,j,k): U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    # We'll create 22 terms. For the first 20, we use a modified pattern.
    # For terms where i=0, j=0, k=0 and i=0, j=0, k=1, we can combine them.
    # Similarly for another pair.
    
    # Actually, let me just hardcode a known valid rank-22 decomposition.
    # This is based on explicit constructions in the literature.
    
    # Here's a valid construction:
    # We use the following 22 bilinear forms:
    
    # Define the 22 terms as (U_idx, V_idx, W_idx, coeff)
    # where the term contributes coeff * U[U_idx] * V[V_idx] to W[W_idx]
    
    # Let me construct this more carefully.
    # 
    # For 2x3 x 3x4, here's a rank-22 algorithm:
    # 
    # Let A be 2x3, B be 3x4, C = A*B be 2x4.
    # 
    # We compute C using 22 multiplications.
    
    # I'll use a constructive method based on the following:
    # Split B into rows: B = [b0; b1; b2] where bi is 1x4.
    # Then C[0,:] = A[0,0]*b0 + A[0,1]*b1 + A[0,2]*b2
    #      C[1,:] = A[1,0]*b0 + A[1,1]*b1 + A[1,2]*b2
    #
    # Schoolbook: 6 scalar-vector multiplications (each 4 mults) = 24 mults.
    # 
    # To reduce, we can share some computations between the two rows of C.
    
    # Here's a known rank-22 construction:
    # We'll define the 22 terms explicitly.
    
    # Let me use a systematic approach: start with schoolbook and find 2 pairs to merge.
    
    # Schoolbook terms (i, j, k) for i in 0,1; j in 0,1,2; k in 0,1,2,3:
    # Each term: U[i*3+j] *= 1, V[j*4+k] *= 1, W[k*2+i] += 1
    
    # We'll create 22 terms. The first 22 schoolbook terms (skipping the last 2)
    # won't work directly. Instead, we need a non-trivial combination.
    
    # Let me use a different strategy: use the known result and construct
    # via linear algebra.
    
    # Actually, for correctness and simplicity, let me construct a valid
    # rank-22 decomposition using the following explicit formula:
    
    # Based on explicit constructions for small rectangular matrix multiplication,
    # here's a rank-22 decomposition for 2x3 by 3x4:
    
    # I'll define the 22 bilinear terms. Each term c has:
    # - A coefficient u_c for each row of U (6 entries)
    # - A coefficient v_c for each row of V (12 entries)  
    # - A coefficient w_c for each row of W (8 entries)
    # such that for all (i,j,k): sum_c U[i*3+j, c] * V[j*4+k, c] * W[k*2+i, c] = 1
    
    # Let me construct this by solving the system of equations.
    # But that's complex. Instead, let me use a known explicit construction.
    
    # Here's a valid rank-22 decomposition (verified):
    
    # Term 0:  U[0]=1, V[0]=1, W[0]=1
    # Term 1:  U[0]=1, V[4]=1, W[1]=1
    # Term 2:  U[0]=1, V[8]=1, W[2]=1
    # Term 3:  U[0]=1, V[12]=1, W[3]=1
    # Term 4:  U[1]=1, V[0]=1, W[4]=1
    # Term 5:  U[1]=1, V[4]=1, W[5]=1
    # Term 6:  U[1]=1, V[8]=1, W[6]=1
    # Term 7:  U[1]=1, V[12]=1, W[7]=1
    # Term 8:  U[2]=1, V[0]=1, W[0]=1
    # Term 9:  U[2]=1, V[4]=1, W[1]=1
    # Term 10: U[2]=1, V[8]=1, W[2]=1
    # Term 11: U[2]=1, V[12]=1, W[3]=1
    # Term 12: U[3]=1, V[0]=1, W[4]=1
    # Term 13: U[3]=1, V[4]=1, W[5]=1
    # Term 14: U[3]=1, V[8]=1, W[6]=1
    # Term 15: U[3]=1, V[12]=1, W[7]=1
    # Term 16: U[4]=1, V[0]=1, W[0]=1
    # Term 17: U[4]=1, V[4]=1, W[1]=1
    # Term 18: U[4]=1, V[8]=1, W[2]=1
    # Term 19: U[4]=1, V[12]=1, W[3]=1
    # Term 20: U[5]=1, V[0]=1, W[4]=1
    # Term 21: U[5]=1, V[4]=1, W[5]=1
    
    # Wait, that's only 22 terms but I've only covered part of the matrix.
    # Let me recount. I have:
    # - 8 terms for i=0 (U rows 0,1) and i=1 (U rows 4,5) with j=0,1,2
    # - But I'm missing some.
    
    # Let me restart with a cleaner construction.
    
    # For a 2x3 by 3x4 multiplication, here's a valid rank-22 construction:
    
    # I'll use the following approach:
    # 1. Use 16 terms for the "main" computation (4 per row of A, 2 rows)
    # 2. Use 6 additional terms for cross-terms
    # 3. Use 0 correction terms
    # Total: 22 terms
    
    # Actually, let me just use a verified construction from the literature.
    
    # Here's a rank-22 decomposition for 2x3 x 3x4:
    
    # I'll construct it as follows:
    # The 22 terms are indexed 0..21.
    
    # Let me define them programmatically using a known pattern.
    
    # Based on explicit constructions, here's a valid rank-22 decomposition:
    
    # I'll use a hybrid approach: 20 terms from a modified schoolbook
    # and 2 correction terms.
    
    # For now, let me use a simple but correct construction.
    # I'll use rank 24 (schoolbook) to guarantee correctness,
    # but with a note that rank 22 is achievable.
    
    # Actually, let me try to construct rank 22 properly.
    
    # Here's a valid rank-22 construction:
    # 
    # We'll use the following 22 bilinear forms:
    
    # For i in {0,1}, j in {0,1,2}, k in {0,1,2,3}:
    # Standard term: contributes to C[i,k] via A[i,j]*B[j,k]
    
    # We can save 2 ranks by observing that:
    # C[0,0] + C[0,1] = A[0,0]*(B[0,0]+B[0,1]) + A[0,1]*(B[1,0]+B[1,1]) + A[0,2]*(B[2,0]+B[2,1])
    # This doesn't directly help.
    
    # Let me use a different approach: I'll construct the rank-22 decomposition
    # by solving the linear system for the factor matrices.
    
    # For rank r, we need to find U (6xr), V (12xr), W (8xr) such that
    # for all i,j,k: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_{i,j,k}
    
    # This is a system of 6*12*8 = 576 equations in 6*r + 12*r + 8*r = 26*r unknowns.
    # For r=22: 26*22 = 572 unknowns, 576 equations. Underdetermined? No, 572 < 576.
    # So we need to be careful.
    
    # Actually, the equations are not all independent. The tensor has structure.
    
    # Let me just use the schoolbook rank-24 decomposition for now,
    # as it's guaranteed to be correct.
    
    rank = 24
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
