"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Build the target tensor
    def build_target_tensor():
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i2 in range(n):
                                if j == j2 and i == i2:
                                    T[i*m+j, j2*p+k, k2*n+i2] = 1.0
        return T
    
    T = build_target_tensor()
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of T."""
        recon = np.einsum('ac,bc,cc->abc', U, V, W, optimize=True)
        return np.allclose(recon, T, atol=1e-8)
    
    # Try to reduce rank by removing columns
    def try_reduce_rank(U, V, W):
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = current_U.shape[1]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for col in range(current_rank):
                if time.time() > deadline - 10:
                    break
                mask = np.ones(current_rank, dtype=bool)
                mask[col] = False
                U_test = current_U[:, mask]
                V_test = current_V[:, mask]
                W_test = current_W[:, mask]
                if verify(U_test, V_test, W_test):
                    current_U, current_V, current_W = U_test, V_test, W_test
                    current_rank -= 1
                    improved = True
                    break
        return current_U, current_V, current_W, current_rank
    
    U_red, V_red, W_red, rank_red = try_reduce_rank(U_sb, V_sb, W_sb)
    
    # Convert to lists
    return {
        "U": U_red.tolist(),
        "V": V_red.tolist(),
        "W": W_red.tolist()
    }
# EVOLVE-BLOCK-END

