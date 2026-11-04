"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 120
    
    def schoolbook_decomp():
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def to_list(mat):
        return [[round(float(x), 12) for x in row] for row in mat]
    
    # Verify decomposition correctness
    def verify(U, V, W):
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(result, 1.0, atol=1e-10)
    
    # Strassen's algorithm for 2x2 matrix multiplication
    def strassen_2x2():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        """
        # Strassen's 7 products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices: C11=0, C12=1, C21=2, C22=3
        
        # M1 = (A11 + A22)(B11 + B22)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0
        
        # M2 = (A21 + A22)B11
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[2, 1] = 1.0
        
        # M3 = A11(B12 - B22)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[1, 2] = 1.0
        
        # M4 = A22(B21 - B11)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[2, 3] = 1.0; W[0, 3] = 1.0
        
        # M5 = (A11 + A12)B22
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[1, 4] = 1.0
        
        # M6 = (A21 - A11)(B11 + B12)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0
        
        # M7 = (A12 - A22)(B21 + B22)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0
        
        return U, V, W
    
    # Try to use Strassen-like decomposition for 4x5x6
    # One approach: partition into 2x2 blocks where possible
    # For 4x5: rows can be split into 2+2, cols into 2+3
    # For 5x6: rows into 2+3, cols into 3+3
    # This is complex; let's try a different approach
    
    # Use the fact that we can embed 2x2 Strassen into larger matrices
    # by having overlapping 2x2 sub-blocks
    
    # Alternative: use a direct construction based on known low-rank decompositions
    # For 4x4x4, the rank is known to be 49 (or less)
    # For our case, we'll try a structured approach
    
    # Let's try a greedy approach: start with schoolbook and try to find
    # linear dependencies that allow rank reduction
    
    best_U, best_V, best_W = schoolbook_decomp()
    best_rank = schoolbook_rank
    
    # Try to find a lower rank decomposition using structured search
    # We'll use a simple approach: try to find if the tensor has a
    # decomposition with fewer terms by looking at the structure
    
    # For now, return the schoolbook decomposition which is guaranteed correct
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
