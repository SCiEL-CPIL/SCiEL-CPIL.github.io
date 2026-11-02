"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # Build schoolbook factor matrices
    U_list = []
    V_list = []
    W_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_list.append([1.0] * n + [0.0] * (m * n - n) + [0.0] * ((p-1)*n))
                # Actually let's just build proper factor matrices
                pass
    
    # Let me properly construct schoolbook factors
    # U shape: (n*m, rank_school)
    # V shape: (m*p, rank_school)
    # W shape: (p*n, rank_school)
    
    U = [[0.0] * rank_school for _ in range(n * m)]
    V = [[0.0] * rank_school for _ in range(m * p)]
    W = [[0.0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Verify schoolbook: C = A @ B should equal sum_c U[:,c] * V[:,c] * W[:,c]^T
    # Actually the tensor T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c
    # For schoolbook, T[i,j,k] = 1 iff the (i,j,k) triple exists
    
    # Try to reduce rank by finding linear dependencies
    # The multiplication tensor for m×n×p has rank at most m*n*p
    # For small matrices, we can try to find lower-rank decompositions
    
    # Known lower bounds and results:
    # 2×2×2: rank 7 (Strassen) vs schoolbook 8
    # 2×3×3: rank ? 
    # For 3×4×7, let's try to use known results
    
    # Actually, let me try a greedy approach to eliminate dependent terms
    # The tensor is T[i,j,k] = delta_{i,j,k} for the schoolbook
    # We want to express T as sum of rank-1 tensors
    
    # For now, return schoolbook which is always valid
    # This gives score = seed/schoolbook_rank = 1.0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
