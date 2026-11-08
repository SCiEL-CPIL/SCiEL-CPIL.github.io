"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        for r in range(rank):
                            val += U[i*m + j, r] * A[i, j] * V[j*p + k, r] * B[j, k] * W[k*n + i, r]
                    C_comp[i, k] = val
            
            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True
    
    # Schoolbook decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies
    # Convert to numpy for easier manipulation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try random linear combinations to see if we can merge terms
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Try to reduce by 1
        target = best_rank - 1
        if target < 1:
            break
        
        # Random perturbation approach
        U_new = U_arr.copy()
        V_new = V_arr.copy()
        W_new = W_arr.copy()
        
        # Try to merge two columns
        r1 = random.randint(0, best_rank - 1)
        r2 = random.randint(0, best_rank - 1)
        if r1 == r2:
            continue
        
        alpha = random.uniform(-2, 2)
        
        # Try: new_col = col_r1 + alpha * col_r2
        # This is a heuristic
        
        # Actually, let's just do random search for lower rank
        for _ in range(10):
            U_rand = np.random.randn(n * m, target)
            V_rand = np.random.randn(m * p, target)
            W_rand = np.random.randn(p * n, target)
            
            if verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), target):
                best_rank = target
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
