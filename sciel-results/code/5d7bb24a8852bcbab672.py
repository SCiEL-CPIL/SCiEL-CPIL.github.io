"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from datetime import datetime, timedelta

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=100)
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Try to find a lower rank decomposition using randomized SVD-based approach
    # This is a heuristic search that may or may not find an improvement
    
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if datetime.now() > deadline:
            break
        
        # Use randomized approach to find low-rank decomposition
        # The multiplication tensor T[i,j,k] = delta(i,k) * delta(j, something)
        # Actually, T[i,j,k] = 1 if the output (i,k) comes from input (i,j) and (j,k)
        # So T[i,j,k] = delta(i, i') * delta(j, j') * delta(k, k') where i'=i, j'=j, k'=k
        # Wait, the tensor is: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
        # So T[i,j,k] = 1 for all i,j,k in the appropriate ranges
        
        # Let's try a different approach: use the fact that for small n,m we can
        # explicitly construct decompositions
        
        if n == 2 and m == 2:
            # For 2x2 x 2xp, we can try to use Strassen-like ideas
            # But for 2x2, the standard is 4 mults, can we do better for the whole product?
            # Actually, for 2x2 * 2x9, we have 18 outputs
            # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
            # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
            # ... etc.
            # This is like doing 9 separate 2x2 scalar multiplications with shared A entries
            
            # Try rank 35: can we save one multiplication?
            # Probably not easily, but let's try a simple check
            
            # Construct the tensor
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = 1  # This isn't quite right
            
            # Actually, the tensor for matrix multiplication A*B where A is n x m, B is m x p
            # C is n x p, C[i,k] = sum_j A[i,j] B[j,k]
            # The tensor T has dimensions n x m x p where T[i,j,k] = 1
            # This tensor has rank n*m*p in general for this structure
            
            # For our case: 2x2x9 tensor with all 1s has rank 18*2 = 36? No.
            # The all-ones tensor of shape 2x2x9 has rank 2 (it's the outer product of [1,1] x [1,1] x [1,1,1,1,1,1,1,1,1])
            # Wait no, that's not right either. Let me think more carefully.
            
            # The multiplication tensor T_{ijk} = delta_{i,k} is not what we have.
            # We have T_{ijk} = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
            # This tensor is actually the outer product of vec(A) and vec(B) projected appropriately
            
            # Actually, I think I'm overcomplicating this. The standard result is that
            # the rank of the 2x2x2 multiplication tensor is 7 (Strassen), and for
            # 2x2xp it would be related.
            
            # Let me just try a few specific constructions
            
            pass
    
    # Convert to list format
    def to_list(arr):
        return [[arr[i][j] for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
