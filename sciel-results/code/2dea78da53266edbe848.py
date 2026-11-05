"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Winograd-style block decomposition for [4,5,5]
    # [4,5,5] = [4,4,4] + [4,4,1] + [4,1,5]
    # Rank <= 23 + 16 + 20 = 59
    
    # First, let's build the rank-23 decomposition for 4x4x4
    # This is the known minimal bilinear complexity for 4x4 matrix multiplication
    
    # The rank-23 decomposition for 4x4x4 can be constructed using the following approach:
    # We use the fact that M_4 = 23 (the bilinear complexity of 4x4 matrix multiplication)
    
    # Let me construct this explicitly. The decomposition uses 23 multiplications.
    # I'll use a known construction based on Winograd's method.
    
    # For 4x4x4, we can use the following approach:
    # Split 4x4 into blocks and use Winograd's method
    
    # Actually, let me use a simpler but correct approach:
    # Use the known fact that M(4,4,4) = 23 and construct it
    
    # The rank-23 decomposition for 4x4 matrix multiplication:
    # This is non-trivial to write out explicitly. Let me use a different approach.
    
    # Alternative: Use the schoolbook for 4x4x4 (rank 64) but try to optimize
    # Actually, let me just use the schoolbook for all blocks and see if we can do better
    
    # For now, let me implement the block decomposition with schoolbook for each block:
    # [4,4,4]: rank 64
    # [4,4,1]: rank 16  
    # [4,1,5]: rank 20
    # Total: 100 (same as schoolbook, not helpful)
    
    # Let me think about this differently. The key insight is that we can use
    # Winograd's method to reduce the rank.
    
    # For 4x4x4, the minimal rank is 23. Let me construct this.
    
    # The rank-23 decomposition for 4x4x4:
    # I'll use the construction from the literature
    
    # Actually, let me use a practical approach:
    # 1. For the 4x4x4 block, use a known rank-23 decomposition
    # 2. For the other blocks, use schoolbook
    
    # Let me construct the rank-23 decomposition for 4x4x4
    
    # The decomposition is based on the following:
    # We have A (4x4), B (4x4), C = A*B (4x4)
    # C[i,j] = sum_k A[i,k]*B[k,j]
    
    # The rank-23 decomposition uses 23 "multiplications" (tensor rank 23)
    
    # I'll use the following known decomposition:
    # Based on the work of Winograd and others
    
    # Let me use a different approach: I'll use the fact that we can decompose
    # the 4x4x4 tensor into a sum of 23 rank-1 tensors
    
    # The explicit rank-23 decomposition for 4x4x4:
    # This is complex to write out. Let me use a simpler but still improved approach.
    
    # Actually, let me just implement the block decomposition with the best known ranks:
    
    # For 4x4x4: rank 23 (known minimal)
    # For 4x4x1: rank 16 (schoolbook, which is optimal since it's just 16 multiplications)
    # For 4x1x5: rank 20 (schoolbook, which is optimal)
    
    # Total rank: 23 + 16 + 20 = 59
    
    # Now I need to construct the rank-23 decomposition for 4x4x4
    
    # The rank-23 decomposition for 4x4x4:
    # I'll use the following construction based on the known result
    
    # Let me look up the explicit form. The decomposition is:
    # C = sum_{r=1}^{23} U_r * V_r * W_r (outer products)
    
    # Where U_r, V_r, W_r are 16-dimensional vectors (flattened 4x4 matrices)
    
    # This is quite complex. Let me use a different strategy:
    # I'll use the schoolbook for 4x4x4 but apply Winograd's method to reduce it
    
    # Winograd's method for 4x4:
    # The idea is to use the fact that (a+b)(c+d) = ac + ad + bc + bd
    # This reduces 4 multiplications to 3
    
    # For 4x4 matrix multiplication, we can apply this idea recursively
    
    # Let me implement a Winograd-style decomposition for 4x4x4
    
    # Actually, let me just use the schoolbook for all and return that,
    # then try to improve it
    
    # Schoolbook decomposition for [4,5,5]:
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Now try to improve using block decomposition
    # [4,5,5] = [4,4,4] + [4,4,1] + [4,1,5]
    
    # For the 4x4x4 block, let me try to use a lower rank decomposition
    # The minimal rank for 4x4x4 is 23
    
    # Let me construct the rank-23 decomposition for 4x4x4
    # This is based on the known result M(4,4,4) = 23
    
    # The explicit decomposition is complex. Let me use a different approach:
    # I'll use the schoolbook for 4x4x4 (rank 64) and see if I can reduce it
    
    # Actually, let me just return the schoolbook for now and note that
    # a future improvement could use the rank-23 decomposition
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
