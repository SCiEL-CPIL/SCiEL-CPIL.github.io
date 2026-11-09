"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = 2, 5, 5
    N = n * m  # 10
    M = m * p  # 25
    P = p * n  # 10

    def verify(U_list, V_list, W_list):
        """Verify that the decomposition exactly represents matrix multiplication."""
        rank = len(U_list[0])
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        # Build the tensor from the decomposition
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # U shape: (N, rank), V shape: (M, rank), W shape: (P, rank)
        # For each c: outer product U[:,c] @ V[:,c] @ W[:,c]
        # U[:,c] has shape (N,), V[:,c] has shape (M,), W[:,c] has shape (P,)
        # The tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Note: W[k*n+i, c] - index is k*n+i, which maps to (i,k) in the W matrix
        
        # Check against known matrix multiplication tensor
        # For matrix mult: T[i,j,k] = 1 if i,j,k are valid (they always are for the standard tensor)
        # Actually, the matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
        # Wait, no. The tensor for matrix multiplication C = A*B where A is n×m, B is m×p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all (i,j,k) in [0,n)×[0,m)×[0,p)
        
        # Let's verify by computing T from the factors and checking it's all ones
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u_col = U[:, c]  # shape (N,)
            v_col = V[:, c]  # shape (M,)
            w_col = W[:, c]  # shape (P,)
            # u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            # We need to broadcast properly
            # Create 3D arrays
            u3d = u_col.reshape(n, m)  # shape (n, m)
            v3d = v_col.reshape(m, p)  # shape (m, p)
            w3d = w_col.reshape(p, n)  # shape (p, n)
            # T[i,j,k] += u3d[i,j] * v3d[j,k] * w3d[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u3d[i, j] * v3d[j, k] * w3d[k, i]
        
        # For matrix multiplication, T[i,j,k] should be 1 for all valid indices
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a lower-rank decomposition using the block structure
    # The key insight: for 2×5×5, we can potentially use the fact that
    # the first dimension is small (n=2).
    
    # Strategy: Use the structure of the problem to search for rank reductions.
    # One approach: try to find a decomposition where some of the rank-1
    # tensors combine to give the full multiplication.
    
    # For now, let's try a systematic search for rank < 50
    # We'll use a greedy approach: start with schoolbook and try to merge/eliminate terms
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try rank values below schoolbook
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But for matrix multiplication, the rank is typically higher.
    # For 2×5×5, the rank is known to be at most 50 (schoolbook).
    # Let's try to find if rank < 50 is possible.
    
    # Use a randomized approach to try to find lower-rank decompositions
    np.random.seed(42)
    
    # Try different rank values
    for target_rank in range(49, 24, -1):
        if time.time() > deadline - 10:
            break
        # Try to find a decomposition of this rank
        # This is a hard problem, so we'll use a heuristic approach
        
        # One heuristic: start with schoolbook and try to merge pairs
        # or use random initialization with verification
        
        # For small ranks, try random sparse matrices
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization
            U = np.zeros((N, target_rank))
            V = np.zeros((M, target_rank))
            W = np.zeros((P, target_rank))
            
            # Initialize with random ±1 entries
            for c in range(target_rank):
                # Randomly choose which entries to set
                n_nonzero_u = max(1, N // 3)
                n_nonzero_v = max(1, M // 3)
                n_nonzero_w = max(1, P // 3)
                
                u_indices = np.random.choice(N, size=min(n_nonzero_u, N), replace=False)
                v_indices = np.random.choice(M, size=min(n_nonzero_v, M), replace=False)
                w_indices = np.random.choice(P, size=min(n_nonzero_w, P), replace=False)
                
                for idx in u_indices:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in v_indices:
                    V[idx, c] = np.random.choice([-1, 1])
                for idx in w_indices:
                    W[idx, c] = np.random.choice([-1, 1])
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
