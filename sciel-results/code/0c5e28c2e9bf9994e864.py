import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    def verify(U, V, W, rank):
        """Verify that the decomposition correctly computes matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.sum(U[idx_u] * V[idx_v] * W[idx_w])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def strassen_2x2x2():
        """
        Return the Strassen decomposition for 2x2x2 matrix multiplication.
        The rank is 7. The convention is:
        U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        For 2x2x2, m=p=2, so:
        U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
        
        Strassen's 7 multiplications:
        M1 = (A00 + A11)(B00 + B11)
        M2 = (A10 + A11)B00
        M3 = A00(B01 - B11)
        M4 = A11(B10 - B00)
        M5 = (A00 + A01)B11
        M6 = (A10 - A00)(B00 + B01)
        M7 = (A01 - A11)(B10 + B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # U: (i, j) -> coefficient for A[i,j] in each M_c
        # V: (j, k) -> coefficient for B[j,k] in each M_c
        # W: (k, i) -> coefficient for C[i,k] in each M_c (note: index is k*n+i)
        
        # M1: (A00 + A11)(B00 + B11)
        # A00=(0,0), A11=(1,1); B00=(0,0), B11=(1,1)
        # C00 += 1, C11 += 1
        
        # M2: (A10 + A11)B00
        # A10=(1,0), A11=(1,1); B00=(0,0)
        # C10 += 1
        
        # M3: A00(B01 - B11)
        # A00=(0,0); B01=(0,1), B11=(1,1)
        # C01 += 1
        
        # M4: A11(B10 - B00)
        # A11=(1,1); B10=(1,0), B00=(0,0)
        # C00 += 1, C10 += 1
        
        # M5: (A00 + A01)B11
        # A00=(0,0), A01=(0,1); B11=(1,1)
        # C00 -= 1, C01 += 1
        
        # M6: (A10 - A00)(B00 + B01)
        # A10=(1,0), A00=(0,0); B00=(0,0), B01=(0,1)
        # C11 += 1
        
        # M7: (A01 - A11)(B10 + B11)
        # A01=(0,1), A11=(1,1); B10=(1,0), B11=(1,1)
        # C00 += 1, C11 -= 1
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00 (k=0, i=0)
        W[3, 0] = 1  # C11 (k=1, i=1)
        
        # M2
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10 (k=0, i=1)
        
        # M3
        U[0, 2] = 1  # A00
        V[1, 2] = 1  # B01
        V[3, 2] = -1 # B11
        W[1, 2] = 1  # C01 (k=1, i=0)
        
        # M4
        U[3, 3] = 1  # A11
        V[2, 3] = 1  # B10
        V[0, 3] = -1 # B00
        W[0, 3] = 1  # C00 (k=0, i=0)
        W[2, 3] = 1  # C10 (k=0, i=1)
        
        # M5
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00 (k=0, i=0)
        W[1, 4] = 1  # C01 (k=1, i=0)
        
        # M6
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[3, 5] = 1  # C11 (k=1, i=1)
        
        # M7
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00 (k=0, i=0)
        W[3, 6] = -1 # C11 (k=1, i=1)
        
        return U, V, W

    # Start with schoolbook as a safe baseline
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][column] = 1
                V[idx_v][column] = 1
                W[idx_w][column] = 1
                column += 1
    
    if not verify(U, V, W, rank):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank decomposition using a structured approach
    # For 2x5x5, we can try to use Strassen on 2x2 blocks and handle the rest
    # This is a complex optimization problem, so we'll use a heuristic approach
    
    # One strategy: try to find a decomposition with rank < 50
    # by exploiting the structure of the tensor
    
    # For now, let's try a simple heuristic:
    # 1. Start with schoolbook
    # 2. Try to merge terms that share the same (i, k) pair
    # 3. Use random linear combinations to find dependencies
    
    # Since this is complex, let's try a different approach:
    # Use the fact that the rank of 2x4x4 is 14 (via Strassen-like methods)
    # and try to extend this to 2x5x5
    
    # For now, return the schoolbook decomposition as a safe baseline
    # The actual rank reduction would require a more sophisticated algorithm
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
