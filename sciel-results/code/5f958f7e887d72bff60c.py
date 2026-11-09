import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 30
    # Decomposition:
    # 1. i in [0,3), j in [0,3):
    #    - k in [0,3): Schoolbook 3x3x3 (rank 9)
    #    - k in [3,6): Schoolbook 3x3x3 (rank 9)
    # 2. i in [0,3), j in [3,5):
    #    - k in [0,3): Rank 6 decomp for 3x2x3
    #    - k in [3,6): Rank 6 decomp for 3x2x3
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j in [0, 3)
    for i in range(3):
        for j in range(3):
            # Sub-part 1a: k in [0, 3)
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            
            # Sub-part 1b: k in [3, 6)
            for k in range(3, 6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Part 2: j in [3, 5)
    # j_local = j - 3 (0 or 1)
    for i in range(3):
        for j_local in range(2):
            j = j_local + 3
            # Sub-part 2a: k in [0, 3)
            for k in range(3):
                # Term 1: A[i, j_local] * B[j_local, k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                # Term 2: A[i, j_local+1] * B[j_local+1, k]
                U[i * m + (j_local + 1 + 3)][col] = 1 # Wait, j is the column of A.
                # My U index is i*m + j.
                # The term is A[i, j_global] * B[j_global, k].
                # In the loop, I am iterating j_global.
                # The rank 6 decomp for 3x2x3 uses:
                # C[i,k] = sum_{j_l=0}^1 A[i, j_l] B[j_l, k]
                # So for each i,k, we have 2 terms.
                # Term 0: A[i, j_global] B[j_global, k]
                # Term 1: A[i, j_global+1] B[j_global+1, k]
                
                # Correction:
                # In the previous block (Part 1), I used schoolbook for 3x3x3.
                # Here, for 3x2x3, the "schoolbook" is exactly rank 6 (since 3*2*3=18 is wrong, 3*2*3=18 terms? No.
                # 3 rows, 2 cols in A, 3 cols in B.
                # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
                # There are 3*3 = 9 entries in C.
                # Each entry is a sum of 2 products.
                # Total products = 18.
                # But we can share?
                # No, the products A[i,0]B[0,k] are distinct for each i,k.
                # So rank is 18?
                
                # Let's re-evaluate R(3,2,3).
                # T[i,j,k] = 1.
                # Sum U[i,j,r] V[j,k,r] W[k,i,r] = 1.
                # If we use schoolbook, rank is 3*2*3 = 18.
                # Is there a lower rank?
                # R(1,2,3) = 2.
                # R(3,2,3) = 3 * R(1,2,3) = 6?
                # Let's check the tensor definition again.
                # T[i,j,k] = 1 for all i,j,k.
                # For R(1,2,3): i=0.
                # T[0,0,k] = 1, T[0,1,k] = 1.
                # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k].
                # We need U[0,0,r] V[0,k,r] W[k,0,r] + U[0,1,r] V[1,k,r] W[k,0,r] = 1.
                # If we set r=1: U[0,0,1]=1, V[0,k,1]=1, W[k,0,1]=1.
                # Then sum is 1 for T[0,0,k].
                # But we also need T[0,1,k] = 1.
                # If we set r=2: U[0,1,2]=1, V[1,k,2]=1, W[k,0,2]=1.
                # Then sum is 1 for T[0,1,k].
                # So for fixed i, we need 2 ranks?
                # Wait. The sum is over r.
                # For a fixed (i,k), the equation is:
                # sum_r U[i,0,r] V[0,k,r] W[k,i,r] + U[i,1,r] V[1,k,r] W[k,i,r] = 1.
                # This must hold for all k.
                # If we use r=1..6.
                # For i=0:
                # k=0: U[0,0,1]V[0,0,1]W[0,0,1] + ... = 1.
                # k=1: ... = 1.
                # k=2: ... = 1.
                
                # If we use the "row independent" argument:
                # The tensor is a direct sum of 3 tensors T_i, each of size 1x2x3.
                # Since the indices i do not interact, the rank is the sum of the ranks.
                # R(1,2,3) = 2.
                # So R(3,2,3) = 6.
                
                # So my rank 6 assumption was correct.
                # The 18 terms in schoolbook are not minimal.
                # The minimal rank is 6.
                
                # So for each i, j_local, k:
                # We need to add 2 terms? No.
                # For a fixed i, the rank is 2.
                # So for 3 i's, rank is 6.
                # The 6 terms correspond to:
                # For each i (3 of them), we have 2 terms (for j=0 and j=1).
                # But these terms depend on k.
                # Wait.
                # Term for i, j=0: U[i,0,r] V[0,k,r] W[k,i,r].
                # This term contributes to T[i,0,k].
                # It must be 1 for all k.
                # So V[0,k,r] must be 1 for all k?
                # If V[0,k,r] = 1 for all k, and U[i,0,r]=1, W[k,i,r]=1.
                # Then for T[i,0,k], the contribution is 1.
                # What about T[i,1,k]?
                # We need another term r' such that U[i,1,r']=1, V[1,k,r']=1, W[k,i,r']=1.
                
                # So for each i, we need 2 ranks.
                # Total 6 ranks.
                
                # Let's implement this.
                
                pass # Placeholder for logic below
            
            # Sub-part 2b: k in [3, 6)
            for k in range(3, 6):
                pass

    # Re-implementing Part 2 correctly with Rank 6 logic
    
    # Reset col to where Part 1 ended (18)
    col = 18
    
    for i in range(3):
        # j=3 (j_local=0)
        # We need a rank r such that U[i,3,r]=1, V[3,k,r]=1, W[k,i,r]=1 for all k in [0,3)
        # And another rank r' such that U[i,4,r']=1, V[4,k,r']=1, W[k,i,r']=1 for all k in [0,3)
        
        # Rank for j=3
        r_j3 = col
        col += 1
        for k in range(3):
            U[i * m + 3][r_j3] = 1
            V[3 * p + k][r_j3] = 1
            W[k * n + i][r_j3] = 1
            
        # Rank for j=4
        r_j4 = col
        col += 1
        for k in range(3):
            U[i * m + 4][r_j4] = 1
            V[4 * p + k][r_j4] = 1
            W[k * n + i][r_j4] = 1
            
        # Now for k in [3,6)
        # Rank for j=3
        r_j3_b = col
        col += 1
        for k in range(3, 6):
            U[i * m + 3][r_j3_b] = 1
            V[3 * p + k][r_j3_b] = 1
            W[k * n + i][r_j3_b] = 1
            
        # Rank for j=4
        r_j4_b = col
        col += 1
        for k in range(3, 6):
            U[i * m + 4][r_j4_b] = 1
            V[4 * p + k][r_j4_b] = 1
            W[k * n + i][r_j4_b] = 1

    return {"U": U, "V": V, "W": W}
