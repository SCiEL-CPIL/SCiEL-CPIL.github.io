"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit with safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook decomposition as baseline (rank 96)
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
    
    # Verify the decomposition is correct
    def verify(U, V, W, n, m, p):
        """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 iff i,j,k correspond to a valid term."""
        # For matrix multiplication, the tensor T[i,j,k] = delta where the product A[i,j]*B[j,k] contributes to C[i,k]
        # The decomposition should satisfy: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And for all other index combinations, the sum should be 0.
        # Actually, the tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] term exists)
        # Wait, no. The tensor T is defined such that (A*B)[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] B[j,k]
        # Hmm, let me reconsider. The standard formulation:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_j sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] A[i,j] B[j,k]
        # For this to hold, we need: for each (i,j,k), sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
        # And there should be no spurious terms.
        
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
                    prod = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                    if abs(sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Check time - if we've used too much, return immediately
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
