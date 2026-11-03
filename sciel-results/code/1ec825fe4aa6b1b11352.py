"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions based on block matrix multiplication
    # Split A (4x4) into 2x2 blocks, B (4x5) into 2x5 blocks
    # This gives us a 2x2 block structure where we can try to reduce rank
    
    # Strategy: Use the fact that we can decompose using tensor products
    # For 2x2 * 2x2, we know rank 7 is possible (Strassen-like)
    # For our case, let's try to exploit block structure
    
    # Try rank 64: split into 2x2 blocks, each 2x2*2x2 block needs rank ~7
    # 4 blocks * 7 = 28 for the 2x2 part, but we have 5 columns in B
    # So it's more complex. Let's try a different approach.
    
    # Use randomized search with better initialization
    np.random.seed(42)
    
    # Try various ranks with structured random initialization
    for target_rank in [72, 64, 56, 48, 40]:
        if time.time() > deadline - 15:
            break
        
        for attempt in range(100):
            if time.time() > deadline - 5:
                break
            
            # Initialize with sparse random matrices
            density = 0.3
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                # Each column has a few non-zero entries
                u_nonzero = np.random.choice(n * m, size=max(1, int(n * m * density)), replace=False)
                v_nonzero = np.random.choice(m * p, size=max(1, int(m * p * density)), replace=False)
                w_nonzero = np.random.choice(p * n, size=max(1, int(p * n * density)), replace=False)
                
                for idx in u_nonzero:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in v_nonzero:
                    V[idx, c] = np.random.choice([-1, 1])
                for idx in w_nonzero:
                    W[idx, c] = np.random.choice([-1, 1])
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
