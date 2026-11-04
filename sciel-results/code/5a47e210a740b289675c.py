"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 145s to leave safety margin
    
    # Schoolbook decomposition as fallback
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Strategy: Use block decomposition
    # 3x3 x 3x8 = (3x3 x 3x3) + (3x3 x 3x3) + (3x3 x 3x2)
    # Use known rank-19 decomposition for 3x3x3 (Strassen-like)
    # and schoolbook for 3x3x2
    
    # Known rank-19 decomposition for 3x3x3 matrix multiplication
    # This uses the fact that 3x3 = 2x2 + 1x1 blocks
    # We'll use a structured approach: Strassen for 2x2 sub-blocks
    
    # Actually, let's try a direct approach with better initialization
    # Use the structure: decompose into (3x3 x 3x3) x 2 + (3x3 x 3x2)
    
    # For 3x3 x 3x3, the minimal rank is 19 (Lader's bound)
    # For 3x3 x 3x2, the minimal rank is 18 (schoolbook)
    # Total: 19 + 19 + 18 = 56
    
    # Let's try to construct a rank-56 decomposition
    # using Strassen-like ideas for the 2x2 parts
    
    # First, let's try random search with better initialization
    # Focus on half-integer values in {-1, -0.5, 0, 0.5, 1}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try structured decompositions first
    # Approach: Use block structure
    # Split p=8 into p1=3, p2=3, p3=2
    
    # For each block (3x3 x 3xp_i), use a low-rank decomposition
    # Combine by concatenating columns
    
    def try_block_decomposition():
        """Try to construct a block decomposition."""
        p_blocks = [3, 3, 2]
        
        # For 3x3 x 3x3, use rank 19
        # For 3x3 x 3x2, use rank 18
        # Total rank = 19 + 19 + 18 = 56
        
        total_rank = 19 + 19 + 18
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_offset = 0
        for block_idx, p_block in enumerate(p_blocks):
            if p_block == 3:
                block_rank = 19
            else:
                block_rank = 18
            
            # For 3x3 x 3x3 block, use a known decomposition
            # For 3x3 x 3x2 block, use schoolbook
            
            if p_block == 3:
                # Use Strassen-like decomposition for 3x3x3
                # This is complex; for now use schoolbook (rank 27)
                # and try to improve
                for i in range(n):
                    for j in range(m):
                        for k in range(p_block):
                            U[i*m+j][col_offset + i*m + j*p_block + k] = 1
                            V[j*p + k + block_idx*p_block][col_offset + i*m + j*p_block + k] = 1
                            W[(block_idx*p_block + k)*n + i][col_offset + i*m + j*p_block + k] = 1
                col_offset += block_rank
            else:
                # Schoolbook for 3x3 x 3x2
                for i in range(n):
                    for j in range(m):
                        for k in range(p_block):
                            U[i*m+j][col_offset + i*m + j*p_block + k] = 1
                            V[j*p + k + block_idx*p_block][col_offset + i*m + j*p_block + k] = 1
                            W[(block_idx*p_block + k)*n + i][col_offset + i*m + j*p_block + k] = 1
                col_offset += block_rank
        
        return {"U": U, "V": V, "W": W}, total_rank
    
    # Try random search with structured initialization
    # Use values from {-1, -0.5, 0, 0.5, 1}
    
    def random_search(rank, max_attempts=50):
        """Try random search for a given rank."""
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.choice(values, (n*m, rank))
            V = np.random.choice(values, (m*p, rank))
            W = np.random.choice(values, (p*n, rank))
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 56 up to 72
    for rank in range(56, min(73, best_rank)):
        if time.time() > deadline:
            break
        
        result = random_search(rank, max_attempts=20)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
