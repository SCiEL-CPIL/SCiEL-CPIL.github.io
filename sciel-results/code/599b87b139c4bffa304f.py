"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    def schoolbook():
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

    def verify(U, V, W):
        """Verify the decomposition with random inputs."""
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Convert lists to arrays for fast multiplication
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        rank = U_arr.shape[1]
        
        # Compute C using the tensor decomposition
        # C[i, k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        # We can vectorize this:
        # Let U_reshaped be (n, m, rank)
        # Let V_reshaped be (m, p, rank)
        # Let W_reshaped be (p, n, rank)
        
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Term 1: sum_j U[i*m+j, r] A[i,j] -> (n, rank)
        # np.einsum('ijr,ij->ir', U_reshaped, A)
        term_U = np.einsum('ijr,ij->ir', U_reshaped, A)
        
        # Term 2: sum_j V[j*p+k, r] B[j,k] -> (p, rank)
        # np.einsum('jkr,jk->kr', V_reshaped, B)
        term_V = np.einsum('jkr,jk->kr', V_reshaped, B)
        
        # C[i, k] = sum_r W[k*n+i, r] * term_U[i, r] * term_V[k, r]
        # np.einsum('kir,ir,kr->ik', W_reshaped, term_U, term_V)
        C_computed = np.einsum('kir,ir,kr->ik', W_reshaped, term_U, term_V)
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Attempt Rank 41: BSC(23) for 3x3x3 + Schoolbook(18) for 3x3x2
    # Since I do not have the exact BSC coefficients hardcoded reliably, 
    # I will use the schoolbook for the first part as well to ensure validity,
    # but this results in Rank 45.
    # To potentially improve, one would need to insert valid Rank 23 coefficients here.
    # For now, we return the valid schoolbook decomposition.
    U, V, W = schoolbook()
    
    # If we had valid rank 41 coefficients, we would verify them here.
    # Since we are using schoolbook, verification is guaranteed to pass.
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
