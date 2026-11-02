"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # For matrix multiplication (n x m) * (m x p) -> (n x p)
    # Tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find a rank-minimal decomposition.
    
    # The trivial decomposition has rank = n*m*p
    # But we can potentially do better.
    
    # For 2x2 * 2x11, let's think about the structure.
    # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: use the fact that 2x2 matrix multiplication has rank 7 (Strassen)
    # and then extend to 2x11.
    
    # Actually, let's think more carefully.
    # We have A: 2x2, B: 2x11, C: 2x11
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # ...
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # ...
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    
    # So we have 22 output entries, each is a sum of 2 products.
    # Total number of "basic" products = 4 * 11 = 44
    # But we can share computations.
    
    # Let's try a different approach: think of this as multiplying a 2x2 matrix
    # by 11 independent 2x1 matrices (columns of B).
    
    # For each column k of B, we compute C[:,k] = A * B[:,k]
    # This is a 2x2 times 2x1 multiplication, which has 4 basic products.
    
    # If we do this independently for each column, rank = 11 * 4 = 44.
    
    # Can we share computations across columns?
    # The key insight is that A is the same for all columns.
    
    # Let's use Strassen's algorithm for the 2x2 part.
    # Strassen computes M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # Then C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For our case, B is 2x11, so we have 11 columns.
    # Let's apply Strassen to each column independently first.
    # For column k:
    # M1_k = (A[0,0]+A[1,1])*(B[0,k]+B[1,k])
    # M2_k = (A[1,0]+A[1,1])*B[0,k]
    # M3_k = A[0,0]*(B[0,k]-B[1,k])
    # M4_k = A[1,1]*(B[1,k]-B[0,k])
    # M5_k = (A[0,0]+A[0,1])*B[1,k]
    # M6_k = (A[1,0]-A[0,0])*(B[0,k]+B[0,k])  # wait, this doesn't make sense
    # M7_k = (A[0,1]-A[1,1])*(B[1,k]+B[1,k])
    
    # Actually, let me be more careful. Strassen's algorithm is for 2x2 * 2x2.
    # For 2x2 * 2x1, we can use a simpler approach.
    
    # Let's just use the trivial decomposition for now, but try to be smarter.
    
    # Actually, let's think about this differently.
    # The tensor for 2x2 * 2x11 multiplication has:
    # - 4 input entries from A: A[0,0], A[0,1], A[1,0], A[1,1]
    # - 22 input entries from B: B[0,0],...,B[0,10], B[1,0],...,B[1,10]
    # - 22 output entries: C[0,0],...,C[0,10], C[1,0],...,C[1,10]
    
    # The tensor T has entries T_{(i,j,k)} where:
    # i indexes A (i = i_A * m + j_A, so i in [0,4))
    # j indexes B (j = j_B * p + k_B, so j in [0,22))
    # k indexes C (k = k_C * n + i_C, so k in [0,22))
    
    # T_{i,j,k} = 1 if the product A[i_A,j_A] * B[j_B,k_B] contributes to C[i_C,k_C]
    # i.e., if i_A = i_C, j_A = j_B, and k_B = k_C
    
    # So T_{i,j,k} = 1 if (i // m == k // n) and (i % m == j // p) and (j % p == k % n)
    # Wait, let me re-derive.
    
    # i = i_A * m + j_A, so i_A = i // m, j_A = i % m
    # j = j_B * p + k_B, so j_B = j // p, k_B = j % p
    # k = k_C * n + i_C, so k_C = k // n, i_C = k % n
    
    # The product A[i_A, j_A] * B[j_B, k_B] contributes to C[i_C, k_C]
    # if i_A = i_C and j_A = j_B and k_B = k_C.
    
    # So T_{i,j,k} = 1 if (i // m == k % n) and (i % m == j // p) and (j % p == k // n)
    
    # Let's verify: for i=0 (i_A=0, j_A=0), j=0 (j_B=0, k_B=0), k=0 (k_C=0, i_C=0)
    # i // m = 0, k % n = 0, so i_A = i_C ✓
    # i % m = 0, j // p = 0, so j_A = j_B ✓
    # j % p = 0, k // n = 0, so k_B = k_C ✓
    # So T_{0,0,0} = 1 ✓
    
    # For i=0, j=0, k=1 (k_C=0, i_C=1):
    # i // m = 0, k % n = 1, so i_A != i_C, T_{0,0,1} = 0 ✓
    
    # OK so the tensor is well-defined.
    
    # Now, the trivial decomposition has rank 44.
    # Can we do better?
    
    # One idea: group the computations.
    # For a fixed i_A and j_B, we have products A[i_A, j_A] * B[j_B, k_B] for all j_A, k_B.
    # Wait, that's not quite right.
    
    # Let's think of it as: for each (i_A, j_B) pair, we compute a 2x11 matrix
    # where entry (j_A, k_B) is A[i_A, j_A] * B[j_B, k_B].
    # But this is just the outer product of A[i_A, :] (a 1x2 vector) and B[:, j_B] (a 2x11 matrix).
    # Wait, A[i_A, :] is a row of A, so it's 1x2.
    # B[:, j_B] is a column of B, so it's 2x1.
    # The outer product would be 2x2, not 2x11.
    
    # I'm confusing myself. Let me restart.
    
    # The simplest approach: use the trivial decomposition.
    # Rank = 44.
    
    # But maybe we can do better by exploiting the structure.
    
    # Let's try a different approach: use bilinear complexity.
    # The bilinear complexity of (n x m) * (m x p) matrix multiplication is at most n*m*p.
    # For (2 x 2) * (2 x 11), the trivial complexity is 44.
    
    # Is there a known lower bound? The tensor rank is at least max(n*m, m*p, n*p) = max(4, 22, 22) = 22.
    # But this is a very weak bound.
    
    # Actually, the tensor rank of matrix multiplication is at least the "slice rank" or other bounds.
    # For (2 x 2) * (2 x 11), I'm not sure what the exact rank is.
    
    # Let's just try to find a good decomposition.
    
    # Approach: use the fact that we can share the computation of A's entries.
    # For each column k of B, we compute C[:,k] = A * B[:,k].
    # This requires 4 multiplications: A[0,0]*B[0,k], A[0,1]*B[1,k], A[1,0]*B[0,k], A[1,1]*B[1,k].
    
    # Total: 11 * 4 = 44 multiplications.
    
    # Can we reduce this?
    # One idea: use the fact that A is the same for all columns.
    # But A's entries are just scalars, so there's no obvious sharing.
    
    # Another idea: use a different basis.
    # For example, define new variables:
    # A' = A + A^T (or some other transformation)
    # But this might not help.
    
    # Let's just go with the trivial decomposition for now.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
