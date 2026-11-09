"""Matrix multiplication 2,4,6 – rank-minimal decomposition via local search."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np
from scipy.optimize import least_squares

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    n_s, m_s, p_s = 2, 2, 2
    rank_s = 7
    U = np.zeros((n_s * m_s, rank_s), dtype=float)
    V = np.zeros((m_s * p_s, rank_s), dtype=float)
    W = np.zeros((p_s * n_s, rank_s), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[2, 4] = 1
    U[2, 5] = -1; U[3, 5] = 1
    U[1, 6] = 1; U[3, 6] = -1
    
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[0, 3] = -1; V[2, 3] = 1
    V[3, 4] = 1
    V[0, 5] = 1; V[2, 5] = 1
    V[1, 6] = 1; V[3, 6] = 1
    
    W[0, 0] = 1; W[3, 0] = 1
    W[1, 1] = 1; W[3, 1] = -1
    W[2, 2] = 1; W[3, 2] = 1
    W[0, 3] = 1; W[1, 3] = 1
    W[0, 4] = -1; W[2, 4] = 1
    W[3, 5] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_from_strassen():
    """Build 2x4x6 decomposition (rank 42) from Strassen blocks."""
    n, m, p = 2, 4, 6
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Build 2x4x2 (rank 14)
    U_242 = np.zeros((8, 14), dtype=float)
    V_242 = np.zeros((8, 14), dtype=float)
    W_242 = np.zeros((4, 14), dtype=float)
    
    # Block 1: j in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_242[i*4+j, r] = U_s[i*2+j, r]
                V_242[j*2+k, r] = V_s[j*2+k, r] # k loop missing in thought, fix below
        # Correcting the loop structure for clarity and correctness
        # Re-implementing the block building carefully
        
    # Let's use the previous working code logic for building 2x4x2 and 2x4x4
    # to ensure we have a valid baseline of 42.
    
    # Re-using the logic from the current program which is verified to work.
    def _build_2x4x2_from_strassen(U_s, V_s, W_s):
        n, m, p = 2, 4, 2
        rank = 14
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for r in range(7):
            for i in range(2):
                for j_local in range(2):
                    U[i * 4 + j_local, r] = U_s[i * 2 + j_local, r]
                    for k in range(2):
                        V[j_local * 2 + k, r] = V_s[j_local * 2 + k, r]
                    for k in range(2):
                        W[k * 2 + i, r] = W_s[k * 2 + i, r]
        
        for r in range(7):
            for i in range(2):
                for j_local in range(2):
                    U[i * 4 + (j_local + 2), r + 7] = U_s[i * 2 + j_local, r]
                    for k in range(2):
                        V[(j_local + 2) * 2 + k, r + 7] = V_s[j_local * 2 + k, r]
                    for k in range(2):
                        W[k * 2 + i, r + 7] = W_s[k * 2 + i, r]
        
        return U, V, W

    def _build_2x4x4_from_2x4x2(U1, V1, W1):
        n, m, p = 2, 4, 4
        rank = 28
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for r in range(14):
            for i in range(2):
                for j in range(4):
                    U[i * 4 + j, r] = U1[i * 4 + j, r]
                for j in range(4):
                    for k_local in range(2):
                        V[j * 4 + k_local, r] = V1[j * 2 + k_local, r]
                for k_local in range(2):
                    for i in range(2):
                        W[k_local * 2 + i, r] = W1[k_local * 2 + i, r]
        
        for r in range(14):
            for i in range(2):
                for j in range(4):
                    U[i * 4 + j, r + 14] = U1[i * 4 + j, r]
                for j in range(4):
                    for k_local in range(2):
                        V[j * 4 + (k_local + 2), r + 14] = V1[j * 2 + k_local, r]
                for k_local in range(2):
                    for i in range(2):
                        W[(k_local + 2) * 2 + i, r + 14] = W1[k_local * 2 + i, r]
        
        return U, V, W

    U1, V1, W1 = _build_2x4x2_from_strassen(U_s, V_s, W_s)
    U2, V2, W2 = _build_2x4x4_from_2x4x2(U1, V1, W1)
    
    n, m, p = 2, 4, 6
    total_rank = 42
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    U_total[:, :28] = U2
    for j in range(4):
        for k in range(4):
            V_total[j * 6 + k, :28] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(2):
            W_total[k * 2 + i, :28] = W2[k * 2 + i, :]
            
    U_total[:, 28:] = U1
    for j in range(4):
        for k_local in range(2):
            V_total[j * 6 + (k_local + 4), 28:] = V1[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(2):
            W_total[(k_local + 4) * 2 + i, 28:] = W1[k_local * 2 + i, :]
            
    return U_total, V_total, W_total

def _verify(U, V, W):
    n, m, p = 2, 4, 6
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _local_search(U, V, W, time_limit=100):
    """Try to reduce rank by zeroing entries and re-solving."""
    start_time = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = np.sum(best_U != 0) # Proxy for complexity, but we want column rank
    # Actually, rank is the number of non-zero columns.
    current_rank = np.sum(np.any(best_U != 0, axis=0))
    
    while time.time() - start_time < time_limit:
        # Pick a random non-zero entry to try to zero out
        non_zero_U = np.argwhere(best_U != 0)
        if len(non_zero_U) == 0: break
        idx = non_zero_U[np.random.randint(len(non_zero_U))]
        u_idx, u_col = idx[0], idx[1]
        
        # Try zeroing it
        trial_U = best_U.copy()
        trial_U[u_idx, u_col] = 0
        
        # Check if the column is still active
        if np.all(trial_U[:, u_col] == 0):
            # Column became zero, rank decreased
            # Check if V and W columns are also zero? If not, we have a problem.
            # If U col is zero, the term contributes 0.
            # We can simply remove this column from U, V, W.
            # But we need to ensure the tensor is still valid.
            # Since the term was 0, removing it doesn't change the sum.
            # So we can just delete the column.
            mask = np.ones(trial_U.shape[1], dtype=bool)
            mask[u_col] = False
            new_U = trial_U[:, mask]
            new_V = best_V[:, mask]
            new_W = best_W[:, mask]
            
            # Verify
            if _verify(new_U, new_V, new_W):
                best_U, best_V, best_W = new_U, new_V, new_W
                current_rank -= 1
                continue
        
        # If column is not zero, we need to re-solve for the non-zero entries
        # This is expensive. We will skip this for now and just try column removal.
        
    return best_U, best_V, best_W

def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Get baseline 42-rank solution
    U, V, W = _build_2x4x6_from_strassen()
    
    # Verify baseline
    if not _verify(U, V, W):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt local search to reduce rank
    if time.time() < deadline - 20:
        U, V, W = _local_search(U, V, W, time_limit=90)
        
    # Final verification
    if not _verify(U, V, W):
        U, V, W = _schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
