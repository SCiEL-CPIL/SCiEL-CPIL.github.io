"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # For each (i,k), check: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # This should equal sum_j A[i,j]*B[j,k] for all A, B
        # Equivalent: for all i,k, the tensor T[i,k] = sum_r W[k*n+i,r] * (U_row_i_r) * (V_col_k_r) should match the identity
        # More precisely: the bilinear form must reproduce matrix multiplication
        # Check: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if this is a valid term, else 0
        # Actually, the condition is that the resulting bilinear map equals matrix multiplication
        # Let's verify by checking that for random A, B, we get C = AB
        for _ in range(10):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_actual = np.zeros((n, p))
            for r in range(rank):
                a_r = A @ U[:, r]  # shape (n,)
                b_r = V[:, r] @ B  # shape (p,)
                C_actual += W[:, r] * np.outer(a_r, b_r)
            if not np.allclose(C_actual, C_expected, atol=1e-9):
                return False
        return True
    
    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try to reduce rank by removing zero columns or finding redundancies
    # First, check if any columns are zero (they can be removed)
    for c in range(rank0 - 1, -1, -1):
        if time.time() > deadline - 30:
            break
        if np.all(U0[:, c] == 0) or np.all(V0[:, c] == 0) or np.all(W0[:, c] == 0):
            U0 = np.delete(U0, c, axis=1)
            V0 = np.delete(V0, c, axis=1)
            W0 = np.delete(W0, c, axis=1)
            best_rank = U0.shape[1]
            best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    
    # Try local search: random perturbations of a subset of entries, then round
    # This is a heuristic; may not find improvements but worth trying
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10 and attempts < 200:
        attempts += 1
        current_rank = best_U.shape[1]
        if current_rank <= 70:  # Stop if we're doing well
            break
        
        # Try to reduce rank by 1: remove a random column and see if we can compensate
        if current_rank <= 1:
            break
        
        col_to_remove = random.randint(0, current_rank - 1)
        U_trial = np.delete(best_U, col_to_remove, axis=1)
        V_trial = np.delete(best_V, col_to_remove, axis=1)
        W_trial = np.delete(best_W, col_to_remove, axis=1)
        
        # Perturb remaining entries slightly
        U_trial = U_trial + np.random.randn(*U_trial.shape) * 0.1
        V_trial = V_trial + np.random.randn(*V_trial.shape) * 0.1
        W_trial = W_trial + np.random.randn(*W_trial.shape) * 0.1
        
        # Round to half-integers
        U_trial = np.round(U_trial * 2) / 2
        V_trial = np.round(V_trial * 2) / 2
        W_trial = np.round(W_trial * 2) / 2
        
        if verify(U_trial, V_trial, W_trial):
            best_rank = current_rank - 1
            best_U, best_V, best_W = U_trial, V_trial, W_trial
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
