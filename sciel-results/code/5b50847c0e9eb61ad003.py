"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

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

    def verify_numpy(U, V, W):
        """Vectorized verification using numpy."""
        try:
            import numpy as np
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
            return True
        except Exception:
            return False

    # Strategy: "Slice-and-dice" construction
    # Partition the 3x4x7 multiplication tensor into sub-tensors:
    # - 3 blocks of 2x2x7 (using rows 0-1, 1-2, 2-3 of A; but actually we partition A's rows)
    #   Actually, let's partition by A's row index i:
    #   i=0: 1x4x7 subproblem
    #   i=1: 1x4x7 subproblem  
    #   i=2: 1x4x7 subproblem
    #   This gives 3 * rank(1x4x7) total rank.
    #   
    #   For 1x4x7: C[0,k] = sum_j A[0,j]*B[j,k]
    #   This is a rank-1 tensor in i (only i=0), so we need rank(1x4x7) for the j,k part.
    #   The 1x4x7 tensor T[j,k] = 1 for all j in 0..3, k in 0..6.
    #   This is just an outer product of a 4-vector and 7-vector, so rank 1!
    #   Wait, that's not right. The tensor for matrix multiplication C = A*B where A is 1x4, B is 4x7:
    #   C[0,k] = sum_j A[0,j] * B[j,k]
    #   The multiplication tensor has entries T[i,j,k] where i=0, j=0..3, k=0..6, and T[0,j,k]=1 for all j,k.
    #   This tensor is rank 1: U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1... wait no.
    #   Actually T[i,j,k] = delta_{i,0} * 1. So U[i,j,c], V[j,k,c], W[k,i,c] such that sum_c U*V*W = T.
    #   For rank 1: U[0,j,0]=1, V[j,k,0]=1, W[k,0,0]=1. Then sum = 1*1*1 = 1 = T[0,j,k]. Correct!
    #   So rank(1x4x7) = 1. Total rank = 3*1 = 3? That can't be right because the lower bound is 28.
    #   
    #   I'm confusing myself. Let me reconsider.
    #   The matrix multiplication tensor for n x m times m x p has dimensions n*m*p.
    #   For 1x4x7: n=1, m=4, p=7. The tensor T[i,j,k] with i in 0..0, j in 0..3, k in 0..6.
    #   T[i,j,k] = 1 if this is a valid term in the product, which it always is. So T[0,j,k] = 1 for all j,k.
    #   The rank of this tensor: we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k] = delta_{i,0}.
    #   For i=0: sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = 1 for all j,k.
    #   For i!=0: but i only goes to 0, so no constraint.
    #   This is equivalent to: the 4x7 matrix of all ones has rank 1, so we can write it as outer product.
    #   U[0,j,0] = 1 for all j, V[j,k,0] = 1 for all j,k, W[k,0,0] = 1 for all k.
    #   Check: sum_c U[0,j,c]*V[j,k,c]*W[k,0,c] = 1*1*1 = 1. Correct.
    #   So rank(1x4x7) = 1. But wait, that would mean total rank = 3, which violates the lower bound of 28.
    #   
    #   The issue: when we partition by i, we're not getting independent subproblems because the W factor
    #   couples across i. The W matrix has shape (p*n, rank) = (7*3, rank), and W[k*n+i, c] appears in the
    #   term for (i,j,k). If we partition by i, the W factors for different i are in different rows of W,
    #   so they ARE independent. Let me re-examine.
    #   
    #   For the full 3x4x7 problem:
    #   U has shape (3*4, rank) = (12, rank), indexed by (i,j)
    #   V has shape (4*7, rank) = (28, rank), indexed by (j,k)
    #   W has shape (7*3, rank) = (21, rank), indexed by (k,i)
    #   
    #   If we partition by i:
    #   - For i=0: U[0,j,c] for j=0..3, V[j,k,c] for j=0..3,k=0..6, W[k,0,c] for k=0..6
    #   - For i=1: U[1,j,c] for j=0..3, V[j,k,c] for j=0..3,k=0..6, W[k,1,c] for k=0..6
    #   - For i=2: U[2,j,c] for j=0..3, V[j,k,c] for j=0..3,k=0..6, W[k,2,c] for k=0..6
    #   
    #   The V factors are SHARED across i! So the subproblems are NOT independent.
    #   This is why the simple partitioning doesn't work.
    #   
    #   Alternative: partition by j instead.
    #   - For j=0: U[i,0,c] for i=0..2, V[0,k,c] for k=0..6, W[k,i,c] for k=0..6,i=0..2
    #   - For j=1: U[i,1,c] for i=0..2, V[1,k,c] for k=0..6, W[k,i,c] for k=0..6,i=0..2
    #   - etc.
    #   Again, W is shared.
    #   
    #   So the slice-and-dice approach as described in the directive doesn't directly work because
    #   the factors are shared. However, we can still try to find a structured decomposition.
    #   
    #   Let's try a different approach: use the known Strassen-like decomposition for smaller blocks.
    #   For 2x2x2, Strassen gives rank 7 instead of 8.
    #   For 2x2x7, we can use a Kronecker-like product: 2x2x7 = (2x2x2) x (1x1x3.5)? No, that doesn't work.
    #   
    #   Actually, for 2x2xp, the rank is at most 7*p/2 if we use Strassen for the 2x2 part and schoolbook for p.
    #   But that's not quite right either.
    #   
    #   Let me just do a structured random search with the value pool, focusing on finding any improvement
    #   over the schoolbook rank of 84. The key insight from the current program is that random search
    #   with the value pool [0, 1, -1, 0.5, -0.5, 2, -2] might find a lower-rank decomposition.
    #   
    #   To improve the search, let's:
    #   1. Use a smarter initialization: start from the schoolbook decomposition and try to merge columns.
    #   2. Use simulated annealing or gradient-based optimization on the discrete space.
    #   3. Try more targeted ranks based on theoretical bounds.

    best_result = schoolbook()
    best_rank = n * m * p

    # Value pool for half-integer entries
    value_pool = [0, 1, -1, 0.5, -0.5, 2, -2]
    
    # Try to find lower-rank decompositions using structured random search
    # Focus on ranks that are multiples of small numbers, as they may have more structure
    
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Try ranks from 28 up to 83
    
    # Use numpy for faster verification
    import numpy as np
    
    def verify_fast(U, V, W):
        """Fast verification using numpy."""
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            U_r = U_np.reshape(n, m, rank)
            V_r = V_np.reshape(m, p, rank)
            W_r = W_np.reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            return np.allclose(T, 1.0, atol=1e-9)
        except Exception:
            return False

    # Try a structured search with more attempts for promising ranks
    for target_rank in range(28, best_rank):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 0.5:
            break
        
        # Allocate attempts based on rank
        if target_rank < 40:
            max_attempts = min(30, int(remaining_time * 0.5))
        elif target_rank < 60:
            max_attempts = min(15, int(remaining_time * 0.3))
        else:
            max_attempts = min(5, int(remaining_time * 0.15))
        
        if max_attempts < 1:
            max_attempts = 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Use deterministic seeding for reproducibility
            random.seed(hash((target_rank, attempt)) % (2**32))
            
            # Generate random matrices with values from the pool
            U = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice(value_pool) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Verify the decomposition
            if verify_fast(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break  # Move to next rank
    
    return best_result
# EVOLVE-BLOCK-END
