"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to find a lower-rank decomposition for 1x5x5
    def find_1x5x5_decomp(target_rank=20, max_trials=100):
        """Find a decomposition for 1x5x5 matrix-vector product."""
        m_local = 5
        p_local = 5
        
        for r in range(target_rank, m_local * p_local + 1):
            if time.time() > deadline:
                return None
            
            for trial in range(max_trials):
                if time.time() > deadline:
                    return None
                
                # Random initialization
                U_local = np.random.randn(m_local, r)
                V_local = np.random.randn(m_local, p_local, r)
                W_local = np.random.randn(p_local, r)
                
                # Check if the decomposition is valid
                # Condition: sum_c W[k,c] * U[j,c] * V[j,k,c] = delta_{j,k}
                valid = True
                for j in range(m_local):
                    for k in range(p_local):
                        val = np.sum(W_local[k, :] * U_local[j, :] * V_local[j, k, :])
                        expected = 1 if j == k else 0
                        if abs(val - expected) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return U_local, V_local, W_local, r
        
        return None
    
    # Try to find a rank-20 decomposition for 1x5x5
    decomp = find_1x5x5_decomp(target_rank=20, max_trials=100)
    
    if decomp is not None:
        U_local, V_local, W_local, r_local = decomp
        # Combine three 1x5x5 decompositions to get 3x5x5
        total_rank = 3 * r_local
        U_full = np.zeros((n * m, total_rank))
        V_full = np.zeros((m * p, total_rank))
        W_full = np.zeros((p * n, total_rank))
        
        for i in range(n):
            offset = i * r_local
            U_full[i * m:(i+1) * m, offset:offset+r_local] = U_local
            V_full[:, offset:offset+r_local] = V_local
            W_full[:, offset:offset+r_local] = W_local
        
        # Verify the decomposition
        # C[i,k] = sum_c sum_j U[i*m+j, c] * A[i*m+j] * V[j*p+k, c] * B[j*p+k] * W[k*n+i, c]
        # For this to equal sum_j A[i,j] * B[j,k], we need:
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = delta_{j,k} for all i,j,k
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(W_full[k * n + i, :] * U_full[i * m + j, :] * V_full[j * p + k, :])
                    expected = 1 if j == k else 0
                    if abs(val - expected) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fall back to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
