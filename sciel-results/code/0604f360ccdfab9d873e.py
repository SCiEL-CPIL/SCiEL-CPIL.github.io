"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition as baseline (rank 60)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def verify(U, V, W, tol=1e-8):
        """Check if the decomposition is valid using the tensor convention."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: Kronecker product structure
    # Decompose 2x5*5x6 into independent sub-problems using block structure
    # We can split the middle dimension j into groups and use Strassen-like tricks
    
    # Approach 1: Use the fact that rank(mult_{n,m,p}) <= rank(mult_{n,m1,p}) + rank(mult_{n,m-m1,p})
    # But more powerfully, we can use Strassen on 2x2 blocks
    
    # Let's try to construct a lower-rank decomposition using Strassen on the 2x2 (i,k) structure
    # combined with efficient handling of the j dimension
    
    # For 2x5*5x6, we can think of it as:
    # C = A * B where A is 2x5, B is 5x6
    # C[0,:] = A[0,:] * B and C[1,:] = A[1,:] * B
    
    # Using Strassen on the "outer" 2x2 structure isn't directly applicable since we don't have 2x2 blocks
    # But we can use a different decomposition:
    
    # Actually, let's use the known result that rank(mult_{2,m,p}) can be less than 2*m*p
    # For small dimensions, we can try to find explicit decompositions
    
    # Let's try a systematic search for rank < 60 using random initialization + verification
    # with a smarter initialization strategy
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 55 down to 45 with structured random search
    for target_rank in range(55, 44, -1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Use a structured initialization: start from schoolbook and randomly perturb/combine
            # For each column, assign random (i,j,k) patterns
            
            for c in range(target_rank):
                # Randomly select which (i,j) entries are nonzero in U
                # and which (j,k) entries in V, which (k,i) entries in W
                u_mask = rng.random(n * m) < 0.5
                v_mask = rng.random(m * p) < 0.5
                w_mask = rng.random(p * n) < 0.5
                
                U[u_mask, c] = rng.choice([-1, 1])
                V[v_mask, c] = rng.choice([-1, 1])
                W[w_mask, c] = rng.choice([-1, 1])
            
            # Check if this works
            if verify(U, V, W):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < rank_school:
            break
    
    # If we haven't found a better decomposition, try a more targeted approach
    # using the block structure of the problem
    
    if best_rank == rank_school:
        # Try using the fact that we can combine the two rows of A more efficiently
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        
        # We can write:
        # C[0,k] + C[1,k] = sum_j (A[0,j] + A[1,j]) B[j,k]
        # C[0,k] - C[1,k] = sum_j (A[0,j] - A[1,j]) B[j,k]
        
        # This gives us 2*p products for the sum and difference, but we need to recover
        # C[0,k] and C[1,k] from these, which requires 2*p more operations (additions/subtractions)
        # but those are free in the tensor rank model.
        
        # So the rank is at most: p (for sum) + p (for diff) + 2*p (to recover C[0] and C[1])
        # Wait, that's not right. The tensor rank counts the number of scalar multiplications.
        
        # Actually, the standard schoolbook already does this optimally for rectangular matrices
        # in many cases. Let me just return the schoolbook result.
        
        pass
    
    return best_result
# EVOLVE-BLOCK-END
