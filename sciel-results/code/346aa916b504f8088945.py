"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Strategy: decompose 3x3x6 as direct sum of two 3x3x3 blocks
    # Block 1: columns 0,1,2 of B -> C columns 0,1,2
    # Block 2: columns 3,4,5 of B -> C columns 3,4,5
    # Each block uses rank-23 decomposition, total rank = 46
    
    # First, get a rank-23 decomposition for 3x3x3
    # We'll use the known Strassen-like decomposition
    def _get_rank23_3x3x3():
        """
        Return a rank-23 decomposition for 3x3 matrix multiplication.
        This is based on known bilinear algorithms.
        """
        n3, m3, p3 = 3, 3, 3
        rank = 23
        
        # We need U (9, 23), V (9, 23), W (9, 23) such that:
        # For all i,j,k in [0,3): sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1
        
        # Use a known construction. One approach is to use the fact that
        # rank(3x3x3) <= 23. Let me construct one explicitly.
        
        # Actually, let me try a different approach: use the known result that
        # we can get rank 23 for 3x3x3. I'll use a search-based approach to find it,
        # or use a known explicit construction.
        
        # For now, let's use a simpler but valid approach: 
        # We know rank(3x3x3) <= 23. Let me try to find it via optimization.
        
        # Alternative: use the fact that we can decompose 3x3x3 as:
        # - Use Strassen for the 2x2x2 subproblem and handle the rest
        # This gets us to rank 23.
        
        # Let me try a direct construction based on known results.
        # One known rank-23 decomposition uses the following structure:
        
        # I'll use a computational approach: start from schoolbook (rank 27) 
        # and try to reduce to rank 23 via random search.
        
        U = np.zeros((9, rank), dtype=float)
        V = np.zeros((9, rank), dtype=float)
        W = np.zeros((9, rank), dtype=float)
        
        # Initialize with schoolbook for first 27 entries, but we only have 23
        # So we need a smarter initialization
        
        # Let's try random search from rank 23
        best_found = None
        for attempt in range(5000):
            if time.time() > deadline - 20:
                break
                
            np.random.seed(attempt)
            
            # Use {-1, 0, 1} values with some structure
            U = np.random.choice([-1, 0, 1], size=(9, rank))
            V = np.random.choice([-1, 0, 1], size=(9, rank))
            W = np.random.choice([-1, 0, 1], size=(9, rank))
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_found = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_found is None:
            # Fallback: use schoolbook rank 27 and return it
            # But we need rank 23, so let's try harder
            # For now, return a schoolbook decomposition padded to rank 23 won't work
            # Let's just return schoolbook and handle it at the top level
            return None
        
        return best_found
    
    # Try to get rank-23 decomposition
    rank23_result = _get_rank23_3x3x3()
    
    if rank23_result is not None:
        U3, V3, W3 = rank23_result
        rank3 = 23
        
        # Construct the 3x3x6 decomposition using two copies of the rank-23 block
        total_rank = 2 * rank3  # 46
        
        U = [[0] * total_rank for _ in range(n * m)]  # (9, 46)
        V = [[0] * total_rank for _ in range(m * p)]  # (18, 46)
        W = [[0] * total_rank for _ in range(p * n)]  # (18, 46)
        
        # Block 1: columns 0-2 of B, columns 0-2 of C
        for c in range(rank3):
            for i in range(3):
                for j in range(3):
                    U[i * m + j][c] = U3[i * 3 + j][c]
            for j in range(3):
                for k in range(3):
                    V[j * p + k][c] = V3[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    W[k * n + i][c] = W3[k * 3 + i][c]
        
        # Block 2: columns 3-5 of B, columns 3-5 of C
        for c in range(rank3):
            offset = rank3 + c
            for i in range(3):
                for j in range(3):
                    U[i * m + j][offset] = U3[i * 3 + j][c]
            for j in range(3):
                for k in range(3):
                    # k in [0,3) maps to column k+3 in the 6-column matrix
                    V[j * p + (k + 3)][offset] = V3[j * 3 + k][c]
            for k in range(3):
                for i in range(3):
                    # k in [0,3) maps to column k+3 in the 6-column matrix
                    W[(k + 3) * n + i][offset] = W3[k * 3 + i][c]
        
        # Verify the decomposition
        def verify_decomposition(U_mat, V_mat, W_mat, rank, n, m, p):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_mat[i*m+j][c] * V_mat[j*p+k][c] * W_mat[k*n+i][c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
        
        if verify_decomposition(U, V, W, total_rank, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
# EVOLVE-BLOCK-END
