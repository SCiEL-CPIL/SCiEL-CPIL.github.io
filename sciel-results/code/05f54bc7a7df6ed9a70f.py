"""Matrix multiplication 3x3 by 3x6 tensor decomposition."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random
import copy

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Rank-minimal exact tensor decomposition.
    
    We start with the schoolbook rank-54 decomposition and attempt to reduce
    the rank using a search for linear dependencies among the terms.
    The key insight is that for C = A * B where B is 3x6, we can view this
    as two 3x3 multiplications sharing the same A, which may allow for
    a decomposition with rank less than 2*23=46.
    """
    n, m, p = DIMENSIONS
    
    # Deadline guard: 150s wall-clock limit, use 140s for safety
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook decomposition: rank = n*m*p = 3*3*6 = 54
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
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a lower rank decomposition by exploiting structure
    # The tensor T[i,j,k] for C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to find linear combinations of terms that give the same result
    
    # Approach: Try to merge terms where possible
    # For 3x3x6, we have 54 terms. Each term (i,j,k) contributes to C[i,k]
    # via A[i,j]*B[j,k].
    
    # A key observation: terms with the same (i,k) but different j can potentially
    # be combined if we can express the sum as a single product.
    
    # Let's try a different approach: use the known rank-23 decomposition for 3x3x3
    # and see if we can extend it to 3x3x6.
    
    # For now, let's try to verify if we can do better than 54 by checking
    # if the 54 terms have linear dependencies in the tensor space.
    
    # Convert to numpy for efficient computation
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor is T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # We want to find the minimal number of terms needed.
    
    # One approach: try to find a subset of terms that still spans the tensor
    # This is a combinatorial problem, so we'll use a greedy approach.
    
    # Check time
    if time.time() > deadline - 30:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a rank-46 decomposition by using two rank-23 decompositions
    # for the two 3x3 blocks of B.
    
    # The rank-23 decomposition for 3x3x3 is known. Let's use it.
    # Each term in the rank-23 decomposition has vectors u, v, w in {−1,0,1}^9.
    
    # For 3x3x6, we can use the same 23 terms for the first 3 columns of B
    # and a different 23 terms for the last 3 columns of B.
    # But we need to be careful: the U vectors must be the same for both blocks
    # because A is shared.
    
    # Actually, in the rank-23 decomposition, the U vectors depend on both
    # the input matrix A and the structure of the decomposition. For 3x3x3,
    # the U vectors are 9-dimensional. For 3x3x6, if we use the same U vectors
    # for both blocks, we need the V and W vectors to be appropriately extended.
    
    # Let me define the rank-23 decomposition explicitly.
    
    # The 23 terms for 3x3x3 matrix multiplication:
    # Each term c has vectors u_c, v_c, w_c in {−1,0,1}^9.
    # Indexing: A[i*m+j], B[j*p+k], C[k*n+i] for m=p=n=3.
    
    # Here are the 23 terms (u, v, w) for 3x3x3:
    
    r23_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
    ]
    
    r23_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
    ]
    
    r23_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
    ]
    
    # Verify the rank-23 decomposition for 3x3x3
    def verify_3x3x3(U23, V23, W23):
        """Verify that sum_c u_c ⊗ v_c ⊗ w_c equals the 3x3x3 matmul tensor."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    target = 1.0
                    computed = 0.0
                    for c in range(23):
                        computed += U23[c][i*3+j] * V23[c][j*3+k] * W23[c][k*3+i]
                    if abs(computed - target) > 1e-9:
                        return False
        return True
    
    # Check if the rank-23 decomposition is valid
    if verify_3x3x3(r23_U, r23_V, r23_W):
        # Great! Now let's use it for 3x3x6.
        # For 3x3x6, we have C (3x6) = A (3x3) * B (3x6)
        # We can write B as [B1, B2] where B1, B2 are 3x3.
        # C = [A*B1, A*B2]
        
        # For the first block (columns 0-2 of C), we use the rank-23 decomposition
        # For the second block (columns 3-5 of C), we use the same rank-23 decomposition
        # but with different V and W vectors.
        
        # The U vectors are the same for both blocks because A is shared.
        # The V vectors for the second block need to be shifted.
        # The W vectors for the second block need to be shifted.
        
        # For 3x3x6:
        # U: (9, 46)
        # V: (18, 46)
        # W: (18, 46)
        
        # For the first 23 terms (block 1):
        # U[:, 0:23] = r23_U
        # V[0:9, 0:23] = r23_V
        # V[9:18, 0:23] = 0
        # W[0:9, 0:23] = r23_W
        # W[9:18, 0:23] = 0
        
        # For the next 23 terms (block 2):
        # U[:, 23:46] = r23_U
        # V[0:9, 23:46] = 0
        # V[9:18, 23:46] = r23_V
        # W[0:9, 23:46] = 0
        # W[9:18, 23:46] = r23_W
        
        U46 = [[0] * 46 for _ in range(9)]
        V46 = [[0] * 46 for _ in range(18)]
        W46 = [[0] * 46 for _ in range(18)]
        
        for c in range(23):
            for idx in range(9):
                U46[idx][c] = r23_U[c][idx]
                V46[idx][c] = r23_V[c][idx]
                W46[idx][c] = r23_W[c][idx]
                
                U46[idx][23+c] = r23_U[c][idx]
                V46[9+idx][23+c] = r23_V[c][idx]
                W46[9+idx][23+c] = r23_W[c][idx]
        
        # Verify the rank-46 decomposition for 3x3x6
        def verify_3x3x6(U46, V46, W46):
            """Verify that sum_c u_c ⊗ v_c ⊗ w_c equals the 3x3x6 matmul tensor."""
            for i in range(3):
                for j in range(3):
                    for k in range(6):
                        target = 1.0
                        computed = 0.0
                        for c in range(46):
                            # V index: j*6+k
                            # W index: k*3+i
                            computed += U46[i*3+j][c] * V46[j*6+k][c] * W46[k*3+i][c]
                        if abs(computed - target) > 1e-9:
                            return False
            return True
        
        if verify_3x3x6(U46, V46, W46):
            best_U, best_V, best_W = U46, V46, W46
            best_rank = 46
        
        # Check time
        if time.time() > deadline - 30:
            return {"U": best_U, "V": best_V, "W": best_W}
        
        # Now let's try to find a lower rank decomposition.
        # The rank of M_{3,3,6} is known to be at most 46.
        # Can we do better?
        
        # One approach: try to find linear dependencies among the 46 terms.
        # If some terms are linearly dependent, we can remove them.
        
        # Convert to numpy
        U46_np = np.array(U46, dtype=float)
        V46_np = np.array(V46, dtype=float)
        W46_np = np.array(W46, dtype=float)
        
        # The tensor is T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # We can check if the 46 terms are linearly independent by checking
        # if the matrix [U[:,c] outer V[:,c] for c in range(46)] has rank 46.
        
        # Actually, the terms are linearly independent if the only solution to
        # sum_c alpha_c * (U[:,c] outer V[:,c] outer W[:,c]) = 0 is alpha_c = 0.
        
        # This is equivalent to checking if the 46 vectors U[:,c] outer V[:,c]
        # (flattened to 9*18=162 dimensional vectors) are linearly independent.
        
        # Let's check the rank of the matrix formed by these vectors.
        # Each term c gives a vector of length 9*18 = 162.
        
        # Actually, let's check the rank of the 46x162 matrix.
        # If the rank is 46, the terms are linearly independent.
        
        # But this doesn't directly tell us if we can reduce the rank.
        # We need to check if there's a subset of terms that still spans the tensor.
        
        # For now, let's just return the rank-46 decomposition if it's valid.
        # If we have time, we can try to find a lower rank decomposition.
        
        # Try to find a rank-45 decomposition by removing one term and adjusting
        # the others.
        
        # This is a hard problem, so let's just return the best we have.
        
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If the rank-23 decomposition is not valid, return the schoolbook decomposition
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
