"""Program for 3x5x5 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # The tensor for matrix multiplication of (n x m) * (m x p) -> (n x p)
    # T[i, j, k] = 1 for the term A[i,j] * B[j,k] contributing to C[i,k]
    # The factor matrices U (nm x r), V (mp x r), W (pn x r) satisfy:
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other index combinations, the sum should be 0.
    
    # We use a structured approach:
    # Decompose the 5x5 space as 4x4 core + rank-1 correction.
    # 
    # Key insight: A 3x5x5 multiplication can be viewed as:
    # - Split columns of A (5 cols) into 4 + 1
    # - Split rows of B (5 rows) into 4 + 1  
    # - Split columns of B (5 cols) into 4 + 1
    #
    # This gives blocks:
    # C[3x5] = A[3x4]*B[4x5] + A[3x1]*B[1x5]
    # 
    # More precisely, let's use the approach of embedding into 4x4x4:
    # We can write the 3x5x5 multiplication using a 4x4x4 decomposition
    # with some zero-padded structure.
    
    # Actually, let's try a direct algebraic approach.
    # The known result: rank(3x5x5) <= rank(4x4x4) + extra terms
    # Strassen: rank(4x4x4) = 49 (known), but we need 3x5x5.
    
    # Let's try a different approach: use the fact that
    # 3x5x5 can be decomposed by treating it as 3x(4+1)x(4+1)
    # = 3x4x4 + 3x4x1 + 3x1x4 + 3x1x1
    # 
    # Rank of 3x4x4: schoolbook = 48
    # Rank of 3x4x1: schoolbook = 12
    # Rank of 3x1x4: schoolbook = 12
    # Rank of 3x1x1: schoolbook = 3
    # Total schoolbook: 75
    
    # But we can do better by sharing structure.
    
    # Let's try to find a decomposition with rank < 75 using a linear algebra approach.
    # The tensor has 3*5*5 = 75 "non-zero" entries in the (i,j,k) space, but
    # the constraints are that for each output (i,k), the sum over j of
    # A[i,j]*B[j,k] must be correct.
    
    # A valid decomposition requires:
    # For all i in [0,n), k in [0,p): sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # For all other (i',j',k') not corresponding to valid (i,j,k) triples:
    #   sum_r U[i'*m+j',r]*V[j'*p+k',r]*W[k'*n+i',r] = 0
    
    # This is a system of bilinear constraints. Let's try to solve it.
    
    # Approach: Use the "border rank" or "approximate rank" ideas, but we need exact.
    
    # Let's try a constructive approach based on block structure.
    # 
    # Represent A as [A1 | A2] where A1 is 3x4, A2 is 3x1
    # Represent B as [[B11, B12], [B21, B22]] where B11 is 4x4, B12 is 4x1,
    #                B21 is 1x4, B22 is 1x1
    #
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #   = C1 + C2 + C3 + C4
    # where C1 is 3x4, C2 is 3x1, C3 is 3x4, C4 is 3x1
    #
    # C = [C1 + C3 | C2 + C4]
    
    # For C1 = A1*B11 (3x4 * 4x4 -> 3x4): schoolbook rank = 48
    # For C2 = A1*B12 (3x4 * 4x1 -> 3x1): schoolbook rank = 12
    # For C3 = A2*B21 (3x1 * 1x4 -> 3x4): schoolbook rank = 12
    # For C4 = A2*B22 (3x1 * 1x1 -> 3x1): schoolbook rank = 3
    #
    # But C2 and C4 both contribute to the last column of C.
    # C3 contributes to the first 4 columns of C.
    #
    # Total if independent: 48 + 12 + 12 + 3 = 75 (same as schoolbook)
    
    # To reduce rank, we need to find shared structure.
    # 
    # Key idea: The computation of C1 + C3 (both 3x4) can share some multipliers.
    # C1[i,k] = sum_j A1[i,j] * B11[j,k]
    # C3[i,k] = A2[i] * B21[k]
    #
    # So (C1+C3)[i,k] = sum_j A1[i,j]*B11[j,k] + A2[i]*B21[k]
    #
    # This is like computing a 3x4 matrix product where one of the "columns" 
    # of the left matrix is augmented with a scalar times the right matrix's row.
    
    # Let's try a different decomposition strategy.
    # Use the fact that we can combine the rank-1 terms.
    
    # Alternative: Use a 4x4x4 decomposition for the core and handle the 
    # remaining terms separately.
    
    # Strassen's algorithm for 4x4x4 has rank 49.
    # But we need 3x5x5, not 4x4x4.
    
    # Let's try to directly construct a lower-rank decomposition.
    # 
    # We'll use the following approach:
    # 1. Start with the schoolbook decomposition (rank 75)
    # 2. Try to find linear dependencies among the 75 elementary tensors
    # 3. If we find that the tensor rank is actually less than 75, we can
    #    express some tensors as linear combinations of others.
    
    # The tensor T is in R^{(3*5) x (5*5) x (5*3)} = R^{15 x 25 x 15}
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), 0 otherwise.
    #
    # The rank of T is the minimum r such that T = sum_{r'} u_{r'} v_{r'} w_{r'}
    # where u_{r'} in R^{15}, v_{r'} in R^{25}, w_{r'} in R^{15}.
    
    # Let's try to find a rank-70 decomposition by exploiting structure.
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but with a cleaner implementation.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Check time
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding a structured decomposition.
    # 
    # Strategy: Use the "split" approach where we decompose 5 = 4 + 1.
    # 
    # Let's define:
    # A = [A_core (3x4) | A_tail (3x1)]
    # B = [B_core (4x4)  B_core2 (4x1); B_tail (1x4)  B_tail2 (1x1)]
    #
    # C = A_core * B_core + A_core * B_core2 + A_tail * B_tail + A_tail * B_tail2
    #
    # Let's compute each part with its own decomposition and then combine.
    
    # Part 1: A_core * B_core (3x4 * 4x4 -> 3x4)
    # Schoolbook rank = 48
    # Part 2: A_core * B_core2 (3x4 * 4x1 -> 3x1)
    # Schoolbook rank = 12
    # Part 3: A_tail * B_tail (3x1 * 1x4 -> 3x4)
    # Schoolbook rank = 12
    # Part 4: A_tail * B_tail2 (3x1 * 1x1 -> 3x1)
    # Schoolbook rank = 3
    #
    # Total = 75, same as schoolbook. No gain from simple splitting.
    
    # To get a gain, we need to find a non-trivial decomposition.
    # 
    # Let's try to use the known rank-49 decomposition for 4x4x4
    # and embed our problem into it.
    
    # Actually, let's try a more direct approach: solve for a rank-r decomposition
    # using iterative refinement.
    
    # For now, return the schoolbook decomposition.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
