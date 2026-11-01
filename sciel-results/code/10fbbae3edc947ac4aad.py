import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 10
    rank = 10
    
    # Define the tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    # a in 0..3, b in 0..5, c in 0..5
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # We want to find U (nm x rank), V (mp x rank), W (pn x rank)
    # such that sum_r U[:,r] V[:,r] W[:,r] == T
    
    # Strategy: Local search with rounding
    # Use multiple restarts
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    start_time = time.time()
    timeout = 110 # 10s margin
    
    # Try to find rank 10
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Scale: entries should be around 1 or -1
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid overflow
        U = U / np.linalg.norm(U)
        V = V / np.linalg.norm(V)
        W = W / np.linalg.norm(W)
        
        # Gradient descent step (simple)
        # This is computationally expensive if done naively.
        # For small dimensions, we can use a more direct method or just rely on the structure.
        # Actually, let's try a simpler heuristic: 
        # Assume the solution has small integer/half-integer values.
        # We can try to solve for W given U, V.
        # T = sum U_r V_r^T W_r^T? No, T is a 3-tensor.
        # Reshape T to (nm*mp, pn)
        T_mat = T.reshape(n*m * m*p, p*n)
        
        # For a fixed U, V, we can solve for W in least squares sense?
        # T_mat = A W_mat, where A is (nm*mp, rank)
        # A[:, r] = (U[:,r] * V[:,r]).flatten()
        
        A = np.zeros((n*m * m*p, rank))
        for r in range(rank):
            A[:, r] = (U[:, r] * V[:, r]).flatten()
            
        # Solve A W_mat = T_mat
        # W_mat is (rank, pn)
        try:
            W_mat, residuals, rank_sol, sv = np.linalg.lstsq(A, T_mat, rcond=None)
            W = W_mat.T # (pn, rank)
        except:
            continue
            
        # Now we have U, V, W.
        # Check if rounding to 0.5 works.
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Verify
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
            
        if np.allclose(T_recon, T, atol=1e-9):
            # Found a valid rank 10 solution
            # Convert to lists
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
        # If not exact, try to refine U and V?
        # For simplicity, just restart.
        
    # Fallback to rank 12
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
# EVOLVE-BLOCK-END
