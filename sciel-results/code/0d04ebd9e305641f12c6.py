"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    import numpy as np

    # Schoolbook decomposition as fallback (rank = n*m*p = 84)
    def schoolbook():
        rank = n * m * p
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

    def verify_fast(U, V, W):
        """Vectorized verification using numpy."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        # Check all entries equal 1
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def verify_integer(U, V, W):
        """Check if entries are integer or half-integer."""
        for mat in [U, V, W]:
            for row in mat:
                for x in row:
                    # Check if x is integer or half-integer
                    doubled = 2 * x
                    if abs(doubled - round(doubled)) > 1e-9:
                        return False
        return True

    # Strategy: Use Kronecker product structure
    # 3x4x7 = (3x2x2) ⊗ (1x2x7/2)? No, that doesn't work directly.
    # Let's think differently.
    # We can view 3x4x7 as a combination of smaller multiplications.
    
    # Actually, let's try a different approach:
    # For matrix multiplication, the tensor is T[i,j,k] = A[i,j]B[j,k]
    # We need to find minimal rank decomposition.
    
    # Known result: For n x m x p, the rank is at most n*m*p.
    # But we can do better using structured decompositions.
    
    # Let's try to use the fact that 4 = 2*2 and 7 is prime.
    # Or 3 = 3, 4 = 2*2, 7 = 7.
    
    # Alternative: Use the identity that for fixed j, the contribution is
    # A[i,j] * B[j,k], which is separable in i and k.
    
    # Let's try a greedy approach:
    # 1. Start with schoolbook
    # 2. Try to eliminate terms by finding linear dependencies
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Try structured constructions
    # For 3x4x7, we can try to split into smaller blocks
    
    # Construction 1: Use 2x2x2 Strassen for parts of the matrix
    # 3x4x7 can be viewed as having 3 rows, 4 cols in A, 4 rows, 7 cols in B
    
    # Let's try a direct construction based on the structure
    
    # For each output entry C[i,k] = sum_j A[i,j] B[j,k]
    # We need to represent this sum using rank-1 terms
    
    # Key insight: If we can find vectors u_i, v_k, w_j such that
    # sum_c u_i[c] * v_k[c] * w_j[c] = delta_{i,k,j} (where delta is 1 if this term exists)
    
    # Actually, the tensor T[i,j,k] = 1 for all i,j,k (since we're multiplying
    # an identity-like structure where each output depends on all inputs)
    
    # Wait, no. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k
    
    # This means we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    # This is equivalent to finding a rank-r decomposition of the all-ones tensor
    # with the constraint that the factors have the specific structure
    
    # Let's try random search with better initialization
    
    value_pool = [0, 1, -1, 0.5, -0.5]
    
    # Try ranks from lower bound up to current best
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    for target_rank in range(28, min(best_rank, 50)):
        if time.time() > deadline:
            break
        
        # Limit attempts per rank to stay within time budget
        max_attempts = 30 if target_rank < 35 else 15
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use structured initialization
            if attempt < 10:
                pool = [0, 1, -1]
            else:
                pool = value_pool
            
            # Generate random matrices with some structure
            U = [[random.choice(pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Quick sparsity check
            if all(all(x == 0 for x in row) for row in U):
                continue
            if all(all(x == 0 for x in row) for row in V):
                continue
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            if verify_fast(U, V, W):
                if verify_integer(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    # If random search didn't find improvement, try a structured construction
    # based on known results for small matrix multiplication
    
    # For 3x4x7, let's try to use the fact that we can combine terms
    
    # Construction: Use the identity that sum_j A[i,j]B[j,k] can be written
    # as a sum of rank-1 terms in the (i,k) space for each j
    
    # For fixed j, the contribution is A[i,j] * B[j,k]
    # This is a rank-1 matrix in (i,k) space
    
    # So we need to represent the all-ones matrix J (n x p) as a sum of
    # rank-1 matrices, one for each j
    
    # But we're constrained by the j-index coupling
    
    # Let's try a different approach: use the fact that 4 = 2+2
    # and split the middle dimension
    
    # Actually, let's just return the best result found
    return best_result
# EVOLVE-BLOCK-END
