import numpy as np
import itertools

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.
    
    Strategy:
    1. Try to find a rank-16 decomposition for the M_{2,3,3} sub-problem using
       a randomized search over integer/half-integer coefficients.
    2. If found, use it for two 3x3 blocks and schoolbook for the 3x2 block.
       Total rank = 16 + 16 + 12 = 44.
    3. If not found within time limit, fall back to schoolbook for all (rank 48).
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    import time
    start_time = time.time()
    time_limit = 30  # seconds
    
    def find_rank16_m233():
        """
        Search for a rank-16 decomposition of M_{2,3,3}.
        Returns (U, V, W) if found, else None.
        """
        rn, rm, rp = 2, 3, 3
        rank = 16
        # U: 6 x 16, V: 9 x 16, W: 6 x 16
        # Target: for all i in {0,1}, j in {0,1,2}, k in {0,1,2}:
        # sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
        
        # We'll use a simple random search with local optimization.
        # Initialize with random values from {-1, 0, 1}
        # Then try to minimize the error.
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Number of attempts
        max_attempts = 20000
        attempts = 0
        
        # Use numpy for speed
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            attempts += 1
            
            # Random initialization
            U = np.random.choice([-1, 0, 1], size=(6, rank))
            V = np.random.choice([-1, 0, 1], size=(9, rank))
            W = np.random.choice([-1, 0, 1], size=(6, rank))
            
            # Compute the tensor
            # T[i, j, k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*2+i, r]
            # We want T[i, j, k] = 1 for all i, j, k
            
            # Build the tensor
            T = np.zeros((2, 3, 3))
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        T[i, j, k] = np.sum(U[i*3+j] * V[j*3+k] * W[k*2+i])
            
            error = np.sum((T - 1.0) ** 2)
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
                if error < 1e-6:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        if best_error < 1e-6:
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try to find the rank-16 decomposition
    result = find_rank16_m233()
    
    if result is not None:
        U_block, V_block, W_block = result
        # Use this for two 3x3 blocks and schoolbook for the 3x2 block
        # Block 1: columns 0-2 of the 8 columns
        # Block 2: columns 3-5 of the 8 columns  
        # Block 3: columns 6-7 of the 8 columns (schoolbook, rank 12)
        
        total_rank = 16 + 16 + 12
        U = [[0] * total_rank for _ in range(n * m)]  # 6 x 44
        V = [[0] * total_rank for _ in range(m * p)]  # 24 x 44
        W = [[0] * total_rank for _ in range(p * n)]  # 16 x 44
        
        # Place first M_{2,3,3} block (columns 0-2)
        for r in range(16):
            for i in range(2):
                for j in range(3):
                    U[i*3+j][r] = U_block[i*3+j][r]
                    # V index: j*8 + k where k is 0-2
                    for k in range(3):
                        V[j*8+k][r] = V_block[j*3+k][r]
                        # W index: k*2 + i
                        W[k*2+i][r] = W_block[k*2+i][r]
        
        # Place second M_{2,3,3} block (columns 3-5)
        for r in range(16):
            col_offset = 16
            for i in range(2):
                for j in range(3):
                    U[i*3+j][col_offset + r] = U_block[i*3+j][r]
                    # V index: j*8 + (3+k) where k is 0-2
                    for k in range(3):
                        V[j*8 + 3 + k][col_offset + r] = V_block[j*3+k][r]
                        # W index: (3+k)*2 + i
                        W[(3+k)*2+i][col_offset + r] = W_block[k*2+i][r]
        
        # Schoolbook for M_{2,3,2} (columns 6-7)
        col_offset = 32
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    U[i*3+j][col_offset + i*3*2 + j*2 + k] = 1
                    V[j*8 + 6 + k][col_offset + i*3*2 + j*2 + k] = 1
                    W[(6+k)*2 + i][col_offset + i*3*2 + j*2 + k] = 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook for the entire 2x3x8
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
