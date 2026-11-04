"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    # Schoolbook rank is n*m*p = 100
    schoolbook_rank = n * m * p

    def build_schoolbook():
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

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix mult: T[i,j,k] = 1 iff the product entry (i,k) of A*B uses A[i,j]*B[j,k]
        # Actually for 4x5 * 5x5: A is 4x5, B is 5x5, C is 4x5
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] should be 1 for all valid (i,j,k) since each triple contributes to exactly one product
        # Wait - the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k
        # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) term appears with coefficient 1.
        # 
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Build the full tensor
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Let's check a sample of entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    def random_decomposition(rank, seed=42):
        """Generate a random candidate decomposition and try to refine it."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize columns
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
            if norm > 1e-12:
                scale = norm ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        return U.tolist(), V.tolist(), W.tolist()

    # Try to find a lower-rank decomposition
    # Known lower bounds and results:
    # For 4x5 * 5x5, the best known upper bound is less than 100.
    # Let's try ranks from 99 down.
    
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    # We'll use a simple approach: try to verify if a lower rank works
    # by attempting random initializations and gradient-free search
    
    # Actually, for this small problem, let's think about structure.
    # The matrix multiplication tensor for n x m * m x p has rank at most n*m*p.
    # Lower bounds: rank >= max(n*p, m*p, n*m) ... no, that's not right.
    # Strassen showed rank < n^3 for n x n * n x n.
    # For 4x5 * 5x5, let's just try to find any decomposition with rank < 100.
    
    # Given the time constraints and the difficulty of finding a provably lower-rank
    # decomposition, let's try a few random searches at ranks 99, 98, ...
    # but this is very unlikely to succeed quickly.
    
    # A more principled approach: use the fact that we can sometimes combine terms.
    # If two terms share structure, we might merge them.
    
    # For now, return the schoolbook decomposition which is guaranteed correct.
    # This gives rank = 100.
    
    # Let's try one quick random search at rank 99 to see if we can do better
    if time.time() < deadline - 30:
        for trial_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 5, 80), -1):
            if time.time() > deadline - 15:
                break
            for attempt in range(5):
                U, V, W = random_decomposition(trial_rank, seed=42 + attempt * 1000)
                if verify(U, V, W, trial_rank):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = trial_rank
                    break
            if best_rank < schoolbook_rank:
                break
    
    return best_result
# EVOLVE-BLOCK-END
