"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Build the target tensor T[i, j, k] = 1 if this triple contributes to C[i,k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid (i,j,k)
    
    # Schoolbook rank = n*m*p = 40
    # Try to find a lower rank decomposition
    
    # Known: For small matrices, we can sometimes do better.
    # Let's try a structured approach: split the m=4 dimension into two groups of 2.
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    # For each group, we have a 2x2x5 subproblem.
    # The 2x2x5 tensor: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, j in {0,1}, k in {0..4}
    # Schoolbook rank for 2x2x5 is 2*2*5 = 20.
    # But can we do better? The 2x2 part has rank 4 (schoolbook) or 3 (Strassen).
    # If we use Strassen for the 2x2 part: rank 3 for the 2x2, then 3*5 = 15 per group?
    # No, that's not how it works. The 2x2x5 tensor rank is not simply 3*5.
    
    # Actually, for 2x2xp matrix multiplication, the rank is known to be:
    # For p=1: rank 4 (or 3 with Strassen)
    # For general p: rank is at most p * rank(2x2x1) = 4p or 3p with Strassen
    # But actually the 2x2xp tensor can be decomposed more efficiently.
    
    # Let me just try to construct a decomposition and verify it.
    
    # Approach: Use the fact that we can compute C = A*B by:
    # 1. Reshape A into a 2x2x2 tensor (2 rows, 2 column-groups, 2 columns per group)
    # 2. Reshape B into a 2x2x5 tensor (2 row-groups, 2 rows per group, 5 cols)
    # 3. Use Strassen-like decomposition for the 2x2 contraction
    
    # Simpler: Just try rank values from a lower bound upward and see if we can find a valid decomposition.
    # But finding a valid decomposition is hard. Let's use a known result.
    
    # The rank of the (2,4,5) matrix multiplication tensor:
    # Lower bound: max(n*m, m*p, n*p) = max(8, 20, 10) = 20
    # Upper bound: 40 (schoolbook)
    
    # A known technique: For (n, m, p) with small n, we can use the "border rank" or known decompositions.
    
    # Let me try a concrete construction:
    # Split m=4 into two halves of size 2.
    # For each half, compute a 2x2x5 product using a rank-? decomposition.
    
    # For 2x2x5: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # We can write this as: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is rank 2*2*5 = 20 schoolbook.
    # Using Strassen for the 2x2 part: 
    #   S1 = (A00+A11)*(B00+B11)  -- but this is for 2x2x1
    # For 2x2xp, we can use:
    #   For each k, compute the 2x2 product using Strassen (3 multiplications per k)
    #   Total: 3*5 = 15 multiplications
    # So rank 15 for 2x2x5.
    
    # Therefore, for 2x4x5: 2 groups of 2x2x5, each rank 15, total rank 30.
    
    # Let's construct this.
    rank = 30
    U = [[0.0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 4*5 = 20 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 5*2 = 10 rows
    
    col = 0
    # Group 0: j in {0,1}
    # Strassen for 2x2: 
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)  -- wait, let me use standard Strassen
    # Standard Strassen for C = A*B where A, B are 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B11)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2x5, A is 2x2 (rows i=0,1; cols j=0,1), B is 2x5 (rows j=0,1; cols k=0..4)
    # C is 2x5 (rows i=0,1; cols k=0..4)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Using Strassen-like:
    # M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k]) for each k
    # M2 = (A[1,0]+A[1,1]) * B[0,k] for each k
    # M3 = A[0,0] * (B[1,k]-B[0,k]) for each k
    # M4 = A[1,1] * (B[0,k]-B[1,k]) for each k
    # M5 = (A[0,0]+A[0,1]) * B[1,k] for each k
    # M6 = (A[1,0]-A[0,0]) * (B[0,k]+B[1,k]) for each k
    # M7 = (A[0,1]-A[1,1]) * (B[0,k]+B[1,k]) for each k
    
    # C[0,k] = M1[k] + M4[k] - M5[k] + M7[k]
    # C[1,k] = M2[k] + M4[k]  ... wait, this doesn't look right for rectangular.
    
    # Let me just verify: for 2x2x1 (p=1), Strassen works. For p>1, we apply Strassen independently for each k.
    # So for 2x2x5, we do 5 independent 2x2x1 Strassen decompositions, each of rank 3.
    # Total rank = 5*3 = 15.
    
    # For each group g in {0,1} (corresponding to j offset 2*g):
    #   For each k in 0..4:
    #     M1 = (A[i,2g] + A[i,2g+1]) ... no, Strassen mixes both i and j.
    
    # Let me be more careful. For group g, the sub-matrices are:
    # A_sub[i, j'] = A[i, 2g + j'] for i in {0,1}, j' in {0,1}
    # B_sub[j', k] = B[2g + j', k] for j' in {0,1}, k in {0..4}
    # C_sub[i, k] = sum_{j'=0}^{1} A_sub[i, j'] * B_sub[j', k]
    
    # Strassen for this 2x2x5:
    # For each k:
    #   M1[k] = (A_sub[0,0] + A_sub[1,1]) * (B_sub[0,k] + B_sub[1,k])
    #   M2[k] = (A_sub[1,0] + A_sub[1,1]) * B_sub[0,k]
    #   M3[k] = A_sub[0,0] * (B_sub[1,k] - B_sub[0,k])
    #   M4[k] = A_sub[1,1] * (B_sub[0,k] - B_sub[1,k])
    #   M5[k] = (A_sub[0,0] + A_sub[0,1]) * B_sub[1,k]
    #   M6[k] = (A_sub[1,0] - A_sub[0,0]) * (B_sub[0,k] + B_sub[1,k])
    #   M7[k] = (A_sub[0,1] - A_sub[1,1]) * (B_sub[0,k] + B_sub[1,k])
    #   C_sub[0,k] = M1[k] + M4[k] - M5[k] + M7[k]
    #   C_sub[1,k] = M2[k] + M4[k]  -- This is WRONG for Strassen.
    
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # But here we have 2x2 * 2x5, so the output is 2x5, not 2x2.
    # The "C01" and "C11" don't exist in the same way.
    
    # Actually, for 2x2 * 2x5, we can't directly use 2x2 Strassen because the right matrix is not 2x2.
    # We need a different approach.
    
    # For 2x2 * 2xp, the rank is known to be at most 3p (using Strassen per column) but the decomposition is:
    # For each k, compute the 2-vector C[:,k] = A_sub * B_sub[:,k] using 3 multiplications.
    # So we need 3*p scalar multiplications total.
    
    # Let me define the 7 Strassen products for a 2x2 * 2x1, then extend to 2x2 * 2x5 by doing it for each k.
    
    # For 2x2 * 2x1 (B is 2x1, so B[0,k], B[1,k] for a single k):
    # M1 = (A00+A11)*(B0+B1)
    # M2 = (A10+A11)*B0
    # M3 = A00*(B1-B0)
    # M4 = A11*(B0-B1)
    # M5 = (A00+A01)*B1
    # M6 = (A10-A00)*(B0+B1)
    # M7 = (A01-A11)*(B0+B1)
    # C0 = M1 + M4 - M5 + M7
    # C1 = M2 + M4  -- NO, this is incorrect.
    
    # I'm confusing myself. Let me just use the schoolbook for now and try a different optimization.
    
    # Given time constraints, let me just return the schoolbook decomposition (rank 40) which is guaranteed correct.
    # But that's the same as the seed. Let me try rank 20 by a simple grouping.
    
    # Simple grouping: split m=4 into two groups of 2.
    # For each group, use schoolbook 2x2x5 = rank 20. Total = 40. No improvement.
    
    # Alternative: Use the fact that n=2 is small. 
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    # We can compute these using 2*4*5 = 40 products, or potentially fewer.
    
    # The tensor rank of (2,4,5) matmul is known to be at most 2*4*5 - something.
    # Actually, I recall that for (2,m,p) matrix multiplication, the rank is at most m*p + 2*p or similar.
    
    # Let me just go with the schoolbook to ensure correctness.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
