"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def get_schoolbook():
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

    # Verification function
    def verify(U, V, W):
        # Check if the decomposition is valid
        # C_{i,k} = sum_r W_{k*n+i, r} (sum_{i',j'} U_{i'*m+j', r} A_{i',j'}) (sum_{j',k'} V_{j'*p+k', r} B_{j',k'})
        # We test with random matrices A and B
        for _ in range(5):
            A = np.random.randint(-2, 3, (n, m))
            B = np.random.randint(-2, 3, (m, p))
            C_expected = A @ B
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            # S_r = U[:, r] dot A_flat? No.
            # U is (n*m, R). A is (n, m).
            # S_r = sum_{i,j} U[i*m+j, r] * A[i,j]
            # This is U[:, r] dot A.flatten()
            S = (U_arr * A.flatten()[:, None]).sum(axis=0) # Shape (R,)
            T = (V_arr * B.flatten()[:, None]).sum(axis=0) # Shape (R,)
            
            # C_{i,k} = sum_r W[k*n+i, r] * S_r * T_r
            # W is (p*n, R). Index k*n+i.
            # C_computed[i,k] = sum_r W[k*n+i, r] * S_r * T_r
            for i in range(n):
                for k in range(p):
                    val = 0
                    for r in range(len(S)):
                        val += W_arr[k*n+i, r] * S[r] * T[r]
                    C_computed[i, k] = val
            
            if not np.allclose(C_computed, C_expected):
                return False
        return True

    start_time = time.time()
    deadline = start_time + 10.0
    
    # Try to find a lower rank decomposition
    # We will try rank 44, 43, 42
    best_result = get_schoolbook()
    
    for target_rank in [44, 43, 42]:
        if time.time() > deadline - 2.0:
            break
            
        # Random search
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify(U, V, W):
                best_result = {"U": U, "V": V, "W": W}
                return best_result
                
    return best_result
# EVOLVE-BLOCK-END
