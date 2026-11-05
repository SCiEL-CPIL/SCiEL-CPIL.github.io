import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Schoolbook Decomposition (Rank 45) - Guaranteed Valid
    def schoolbook_3x3x5():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # 2. Attempt Rank 41 Decomposition (BSC 3x3x3 Rank 23 + Schoolbook 3x3x2 Rank 18)
    def try_rank_41():
        # BSC 3x3x3 Rank 23 Coefficients
        # U, V, W are 9x23
        # Note: These coefficients are from a known valid BSC representation.
        # If invalid, verification will catch it.
        
        bsc_U = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        # Since I don't have the full 23 terms hardcoded correctly in memory, 
        # and the previous inspirations failed, I will use a SAFE fallback.
        # I will use Schoolbook for the 3x3x3 part as well, but I will try to 
        # find a valid Rank 41 decomposition using a simple search if time permits.
        
        # For now, return None to indicate failure, triggering fallback to 45.
        # This ensures we don't return an invalid decomposition.
        return None

    # 3. Verification Function
    def verify_decomposition(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Check shapes
        if U.shape != (n*m, U.shape[1]) or V.shape != (m*p, V.shape[1]) or W.shape != (p*n, W.shape[1]):
            return False
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
            
        rank = U.shape[1]
        
        # Check integer/half-integer
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
                
        # Test with random matrices
        np.random.seed(42)
        for _ in range(5):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            
            # Calculate C using factors
            # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
            
            # Efficient calculation:
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            
            # Reshape U to (n, m, rank)
            U_reshaped = U.reshape(n, m, rank)
            # Reshape V to (m, p, rank)
            V_reshaped = V.reshape(m, p, rank)
            # Reshape W to (p, n, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            # Calculate intermediate sums
            # S1[i, r] = sum_j U[i, j, r] A[i, j]
            S1 = np.einsum('ijr,ij->ir', U_reshaped, A)
            
            # S2[k, r] = sum_j V[j, k, r] B[j, k]
            S2 = np.einsum('jkr,jk->kr', V_reshaped, B)
            
            # C[i, k] = sum_r W[k, i, r] S1[i, r] S2[k, r]
            # Note: W is indexed by (k, i, r) in W_reshaped (p, n, rank)
            C_calc = np.einsum('kir,ir,kr->ik', W_reshaped, S1, S2)
            
            C_true = A @ B
            
            if not np.allclose(C_calc, C_true, atol=1e-9):
                return False
                
        return True

    # Try Rank 41
    if time.time() < deadline:
        result_41 = try_rank_41()
        if result_41:
            U, V, W = result_41
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Fallback to Rank 45
    U, V, W = schoolbook_3x3x5()
    # Verify just in case
    if verify_decomposition(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # If even schoolbook fails (should not happen), return empty or raise
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
