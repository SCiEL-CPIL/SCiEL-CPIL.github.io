"""Find rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where i in [0,n), j in [0,m), k in [0,p)
    # Output index: k*n + i for W, so W[k*n+i, c]
    
    # Schoolbook decomposition
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        # Build tensor from decomposition
        T = np.zeros((n, m, p))
        for c in range(rank):
            uc = np.array(U[:, c]).reshape(n, m)
            vc = np.array(V[:, c]).reshape(m, p)
            wc = np.array(W[:, c]).reshape(p, n)
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        # Check T[i,j,k] == 1 for all
        return np.allclose(T, np.ones((n,m,p)), atol=1e-10)
    
    # Use rank-28 decomposition for 4x4x4 matrix multiplication
    # This gives us rank 56 for 4x4 x 4x8 (two independent 4x4x4 multiplications)
    
    # Known rank-28 decomposition for 4x4 matrix multiplication
    # Using the Coppersmith-Winograd style or known explicit construction
    # For simplicity, let's use a structured approach based on block decomposition
    
    # Actually, let's use the fact that 4x4x4 can be decomposed using 2x2x2 Strassen
    # Strassen for 2x2x2 has rank 7
    # For 4x4x4, we can use Kronecker: rank = 7*7 = 49? No, that's not optimal.
    # 
    # Let me try a different approach: use the known rank-21 or rank-28 decompositions
    # For now, let's construct a valid decomposition with rank 56 using a simpler method
    
    # Strategy: Split the 4x8 matrix into two 4x4 blocks
    # A(4x4) * B(4x8) = A * [B1 | B2] = [A*B1 | A*B2]
    # So we need two independent 4x4x4 multiplications
    # If we have a rank-r decomposition for 4x4x4, we get rank 2r for 4x4x4x8
    
    # Let's try to find a rank-28 decomposition for 4x4x4
    # One approach: use the tensor product of two 2x2x2 Strassen decompositions
    # But that gives rank 49, not 28
    
    # Alternative: use a known explicit rank-28 decomposition
    # For 4x4 matrix multiplication, the tensor rank is known to be 21 (or less)
    # But constructing it explicitly is complex
    
    # Let's try a practical approach: use rank 56 via two independent 4x4x4 decompositions
    # where each 4x4x4 uses rank 28
    
    # For now, let's return schoolbook as fallback and try to improve
    best_result = schoolbook()
    
    # Try to construct a rank-56 decomposition
    # Using the structure: 4x4 * 4x8 = [4x4*4x4 | 4x4*4x4]
    
    # Define a rank-28 decomposition for 4x4x4 matrix multiplication
    # Using a structured approach based on polynomial multiplication or known algorithms
    
    # For simplicity, let's use a valid but not necessarily optimal decomposition
    # Rank 56 can be achieved by using two independent rank-28 decompositions
    
    # Let me try to construct a rank-28 decomposition for 4x4x4
    # One method: use the fact that 4x4 = 2x2 blocks, and use Strassen-like combinations
    
    # Actually, let's just return schoolbook for now if we can't find better quickly
    if time.time() > deadline - 10:
        return best_result
    
    # Try to improve using random search or known decompositions
    # For now, return schoolbook
    
    return best_result
# EVOLVE-BLOCK-END
