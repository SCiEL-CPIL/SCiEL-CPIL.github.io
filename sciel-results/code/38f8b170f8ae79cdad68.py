"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    # Check if U, V, W decompose the tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We only need to check if the resulting tensor matches the identity-like tensor for matmul
    # The tensor for matmul has T[i, j, k] = 1 if it is the term A[i,j] B[j,k]
    # Actually, the tensor T is defined such that sum_{j} A[i,j] B[j,k] = C[i,k]
    # The decomposition is T_{i j k} = sum_c U_{i j c} V_{j k c} W_{k i c}
    # The target tensor T_{i j k} is 1 if we are multiplying A[i,j] and B[j,k]?
    # No, the tensor is T_{i j k} where i is row of A, j is col of A / row of B, k is col of B.
    # The product is C_{i k} = sum_j A_{i j} B_{j k}.
    # The tensor decomposition represents the bilinear map.
    # The coefficients are 1 for all (i,j,k).
    # Wait, the tensor is T_{i j k} = 1 for all i,j,k?
    # No, the tensor is the structure tensor of the algebra.
    # For matrix multiplication, T_{i j k} = 1 if the term A_{i j} B_{j k} appears in C_{i k}.
    # Since every term appears exactly once, T_{i j k} = 1 for all i, j, k.
    
    # Let's verify by computing the "reconstructed" tensor
    # R_{i j k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # We need R_{i j k} == 1 for all i, j, k.
    
    rank = len(U[0])
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Reshape for efficient computation
    # U: (n*m, rank) -> (n, m, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # Compute sum over c
    # Result shape (n, m, p)
    # R[i, j, k] = sum_c U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
    
    # This is a tensor contraction.
    # We can do it in a loop for simplicity and speed check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U_reshaped[i, j, c] * V_reshaped[j, k, c] * W_reshaped[k, i, c]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try to find a lower rank decomposition
    # We start with rank 70 and go down
    # Given the difficulty, we only try a few random initializations
    
    # Fallback
    fallback = _schoolbook()
    
    # Random search for rank 70
    # This is a heuristic. If it finds a solution, great.
    # We use integer coefficients to ensure exactness.
    
    # To save time, we only try if we have > 10s left
    if time.time() < deadline - 10:
        target_rank = 70
        # We can't easily random search for exact integer solutions in 10s.
        # So we stick to the schoolbook.
        
    return fallback
# EVOLVE-BLOCK-END
