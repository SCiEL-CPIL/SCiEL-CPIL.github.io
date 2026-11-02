"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import math
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    # 1. Schoolbook Decomposition (Rank 36)
    # This serves as the baseline and fallback.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school

    # 2. Attempt to find a lower rank decomposition via Random Search
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k.
    # We seek U, V, W of shape (n*m, r), (m*p, r), (p*n, r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    #
    # We use a randomized approach:
    # 1. Generate random sparse matrices U, V, W with small integer entries.
    # 2. Compute the resulting tensor R.
    # 3. Check if R is close to the target tensor (all 1s).
    # 4. If close, try to "snap" to exact integers if possible, or use least squares to correct.
    
    # Known lower bounds suggest rank 34 or 35 might be possible, but 36 is safe.
    # A rank 35 decomposition would be a significant improvement.
    
    # Let's try a local search / random perturbation from the schoolbook solution?
    # Or pure random search for small ranks?
    # Random search for rank < 36 is very hard without structure.
    
    # Alternative: Use the "sum of two products" idea from the prompt?
    # The prompt mentions: "Apply the 'sum of two products' algebraic technique by constructing a rank-21 decomposition as the direct sum of the optimal 3x3x3 matrix multiplication tensor (rank 21) and a trivial 3x3x1 multiplication tensor"
    # Wait, rank of 3x3x3 is 23 (Strassen-like), not 21. Schoolbook is 27.
    # The prompt says "optimal 3x3x3 ... (rank 21)". This is likely incorrect or refers to a specific variant.
    # Actually, the minimal rank of 3x3x3 is 23.
    # If we use rank 23 for the first 3 columns of B, and rank 3 for the 4th column (since 3x3x1 is rank 3? No, 3x3x1 is just 3x3 matrix mult with a vector, rank 3? No, the tensor is size 3x3x1. The multiplication C_ik = sum_j A_ij B_jk. If p=1, C_i = sum_j A_ij B_j. This is a bilinear map. The rank of the 3x3x1 tensor is 3 (just the 3 terms).
    # So direct sum of 3x3x3 (rank 23) and 3x3x1 (rank 3) gives rank 26?
    # Wait, the tensor for 3x3x4 is T_{i,j,k}.
    # If we split k into k=0,1,2 and k=3.
    # T_{i,j,0..2} is the 3x3x3 tensor. Rank 23.
    # T_{i,j,3} is the 3x3x1 tensor. Rank 3.
    # Total rank 23 + 3 = 26?
    # Is this a valid decomposition?
    # U_total = [U_23, U_3]
    # V_total = [V_23, V_3]
    # W_total = [W_23, W_3]
    # The W matrix for the first part must have zeros in the k=3 row.
    # The W matrix for the second part must have zeros in the k=0,1,2 rows.
    # Yes, this is a valid decomposition.
    # So rank 26 is achievable if we have a rank 23 decomposition for 3x3x3.
    # We do not have a rank 23 decomposition hardcoded.
    # However, we can try to find one or use a known one.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # For 3x3x3, there are known algorithms with rank 23.
    
    # Let's try to construct a rank 26 decomposition using a known rank 23 decomposition for 3x3x3.
    # If we can't find a rank 23 decomposition, we can try to search for one.
    # Or we can use the schoolbook 27 + 3 = 30? No, schoolbook for 3x3x3 is 27.
    # 27 + 3 = 30. This is better than 36.
    
    # Let's implement the direct sum of schoolbook 3x3x3 (rank 27) and schoolbook 3x3x1 (rank 3).
    # This gives rank 30.
    
    # 3x3x3 Schoolbook:
    # U1 shape (9, 27), V1 shape (9, 27), W1 shape (9, 27)
    # W1 index is k*n + i. Here k is 0..2, n=3. So W1 rows 0..8.
    # We need to embed this into W shape (12, r).
    # W1_embedded shape (12, 27). Rows 0..8 are W1, rows 9..11 are 0.
    
    # 3x3x1 Schoolbook:
    # U2 shape (9, 3), V2 shape (3, 3), W2 shape (3, 3)
    # Wait, dimensions for 3x3x1:
    # A is 3x3, B is 3x1. C is 3x1.
    # U2 shape (n*m, rank) = (9, 3).
    # V2 shape (m*p, rank) = (3*1, 3) = (3, 3).
    # W2 shape (p*n, rank) = (1*3, 3) = (3, 3).
    # Embed V2 into V shape (12, 3). Rows 0..2 are V2, rows 3..11 are 0.
    # Embed W2 into W shape (12, 3). Rows 9..11 are W2, rows 0..8 are 0.
    # Note: W index is k*n + i. For p=1, k=0. So W index is 0*3 + i = i.
    # But in the 3x3x4 context, the 4th column corresponds to k=3.
    # So W index should be 3*3 + i = 9 + i.
    # So W2_embedded rows 9..11 should be W2.
    
    # Let's construct this rank 30 decomposition.
    
    U_30 = np.zeros((n * m, 30))
    V_30 = np.zeros((m * p, 30))
    W_30 = np.zeros((p * n, 30))
    
    # Part 1: 3x3x3 (k=0,1,2)
    col1 = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U_30[i * m + j, col1] = 1
                V_30[j * p + k, col1] = 1
                W_30[k * n + i, col1] = 1
                col1 += 1
                
    # Part 2: 3x3x1 (k=3)
    # This part computes C_i3 = sum_j A_ij B_j3
    # U2[i*m+j, col] = 1
    # V2[j*p+3, col] = 1
    # W2[3*n+i, col] = 1
    col2 = col1
    for i in range(n):
        for j in range(m):
            U_30[i * m + j, col2] = 1
            V_30[j * p + 3, col2] = 1
            W_30[3 * n + i, col2] = 1
            col2 += 1
            
    # Verify rank 30
    # This is a valid decomposition.
    # Score would be 36/30 = 1.2.
    
    # Can we do better?
    # If we use Strassen for 3x3x3?
    # Strassen for 2x2x2 is rank 7.
    # To get 3x3x3, we can use 4 blocks of 2x2x2?
    # 3 = 2 + 1.
    # This is getting complicated.
    # Let's stick with rank 30 for now.
    
    # Is there a rank 29 or 28?
    # Maybe we can combine the last column of the 3x3x3 part with the 3x3x1 part?
    # The 3x3x3 part uses k=0,1,2.
    # The 3x3x1 part uses k=3.
    # They are disjoint in W and V indices.
    # So they are independent.
    
    # Let's try to search for a better decomposition using random initialization and least squares.
    # We will try ranks 29, 28, ... down to 26.
    
    def verify_decomposition(U, V, W, target_tensor):
        # target_tensor[i, j, k] = 1
        # Compute R[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a tensor product.
        # We can reshape and use einsum.
        # R[i, j, k] = np.einsum('ir,jr,kr->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))
        # But W index is k*n+i. So W.reshape(p, n, -1) is correct.
        # V index is j*p+k. So V.reshape(m, p, -1) is correct.
        # U index is i*m+j. So U.reshape(n, m, -1) is correct.
        
        R = np.einsum('ir,jr,kr->ijk', U.reshape(n, m, -1), V.reshape(m, p, -1), W.reshape(p, n, -1))
        return np.allclose(R, target_tensor, atol=1e-9)
    
    target_tensor = np.ones((n, m, p))
    
    # Verify rank 30
    if verify_decomposition(U_30, V_30, W_30, target_tensor):
        best_U = U_30
        best_V = V_30
        best_W = W_30
        best_rank = 30
        
    # Try to find rank 29, 28, etc. using random search.
    # We will use a simple random search with integer coefficients in {-1, 0, 1}.
    # This is unlikely to find a solution, but it's worth a try.
    
    # Better: Use the rank 30 solution as a starting point and try to reduce rank.
    # Or just return rank 30.
    
    # Let's check if we can find a rank 29 solution by perturbing the rank 30 solution.
    # Or just return rank 30.
    
    # Given the time limit, returning rank 30 is a solid improvement over 36.
    
    # Final check:
    # U shape (9, 30)
    # V shape (12, 30)
    # W shape (12, 30)
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
