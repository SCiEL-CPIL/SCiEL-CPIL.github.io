"""Program for rank-minimal tensor decomposition of 3x4x4 matrix multiplication."""
import time
import random
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_max = n * m * p
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook decomposition (baseline)
    best_U = [[0] * rank_max for _ in range(n * m)]
    best_V = [[0] * rank_max for _ in range(m * p)]
    best_W = [[0] * rank_max for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    best_rank = rank_max
    
    # The tensor T has indices (i,j,k) with value 1 for all i in [0,n), j in [0,m), k in [0,p)
    # T_{ijk} = 1 for all valid i,j,k
    # We want to decompose T as sum_r U_{ij,r} V_{jk,r} W_{ik,r}
    # This is equivalent to finding the tensor rank of the all-ones tensor of shape (n,m,p)
    
    # For the all-ones tensor, the rank is at most max(n,m,p) using:
    # T_{ijk} = sum_r e_i^r * e_j^r * e_k^r where e_i^r = 1 for all i,r
    # Wait, that gives sum_r 1*1*1 = rank terms, but we need sum_r to equal 1
    # T_{ijk} = 1 = (1/rank) * sum_r 1 * 1 * 1 = (1/rank) * rank = 1
    # So rank = 1 works with U_{ij,0}=1, V_{jk,0}=1, W_{ik,0}=1
    # Let's verify: sum_r U_{ij,r} V_{jk,r} W_{ik,r} = 1*1*1 = 1 = T_{ijk}
    # YES! The all-ones tensor has rank 1!
    
    # But wait - the matrix multiplication tensor is NOT the all-ones tensor.
    # The matrix multiplication tensor for (n,m,p) has T_{ijk} = 1 if the corresponding
    # multiplication term exists, which is always 1 for valid i,j,k.
    # Actually, the standard matrix multiplication tensor IS the all-ones tensor.
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (in appropriate basis)
    # The tensor T_{ijk} = 1 for all i,j,k.
    
    # So rank 1 should work! Let me verify the convention.
    # The output factor follows W[k*n+i, r]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Sum over r should give the (i,j,k) entry of the tensor.
    # For matrix multiplication, T_{ijk} = 1 for all valid i,j,k.
    
    # So with rank 1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # Sum_r U*V*W = 1*1*1 = 1 ✓
    
    rank_1 = 1
    U1 = [[1] * rank_1 for _ in range(n * m)]
    V1 = [[1] * rank_1 for _ in range(m * p)]
    W1 = [[1] * rank_1 for _ in range(p * n)]
    
    best_U = U1
    best_V = V1
    best_W = W1
    best_rank = 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
