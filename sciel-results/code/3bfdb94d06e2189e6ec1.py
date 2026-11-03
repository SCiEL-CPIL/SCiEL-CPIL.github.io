"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W form a valid decomposition of the matrix multiplication tensor.
    Condition: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] == delta_{j,j'}
    """
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    # Efficient verification using tensor contractions
    # T_dec[i, j, j', k] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r]
    
    # We can reshape and use einsum or matrix multiplication tricks
    # But for small dimensions, a loop or vectorized numpy ops are fine.
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute the 4D tensor T_dec[i, j, j', k]
    # T_dec = sum_r U_r[i, j, r] * V_r[j', k, r] * W_r[k, i, r]
    # Note: V index is j'*p+k, so V_r[j', k, r].
    # W index is k*n+i, so W_r[k, i, r].
    
    # Using einsum:
    # 'ijr,jkr,kir->ijjk'
    T_dec = np.einsum('ijr,jkr,kir->ijjk', U_r, V_r, W_r)
    
    # Target T_target[i, j, j', k] = delta_{j,j'}
    T_target = np.zeros((n, m, m, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                T_target[i, j, j, k] = 1.0
    
    if not np.allclose(T_dec, T_target, atol=tol):
        return False
        
    return True

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # Strategy: Block decomposition
    # Partition A (3x4) into 3 row blocks of size 1x4.
    # Partition B (4x8) into 4 col blocks of size 4x2.
    # Each sub-problem is 1x4x2.
    # Rank of 1x4x2 matmul tensor is 4.
    # Total rank = 3 * 4 * 4 = 48.
    
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Map global rank index r to (i, b, s)
    # i: row block index (0..2)
    # b: col block index (0..3)
    # s: local rank index (0..3)
    
    for r in range(rank):
        i = r // 16
        b = (r // 4) % 4
        s = r % 4
        
        # 1. Fill U[i*m+j, r]
        # U_sub[j, s] = delta_{j,s}
        # Global index: i*m + j
        U[i * m + s, r] = 1.0
        
        # 2. Fill V[j*p+k, r]
        # V_sub[j, k_local] = delta_{j,s} for k_local in {0, 1}
        # Global k = b*2 + k_local
        for k_local in range(2):
            k_global = b * 2 + k_local
            # V[j, k_global] is 1 if j == s
            V[s * p + k_global, r] = 1.0
            
        # 3. Fill W[k*n+i, r]
        # W_sub[k_local] = 1
        # Global index: k_global*n + i
        for k_local in range(2):
            k_global = b * 2 + k_local
            W[k_global * n + i, r] = 1.0
            
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Optional: Verify
    # if not _verify(U_list, V_list, W_list, n, m, p):
    #     # Fallback to schoolbook if something went wrong
    #     rank = n * m * p
    #     U_sb = np.zeros((n * m, rank))
    #     V_sb = np.zeros((m * p, rank))
    #     W_sb = np.zeros((p * n, rank))
    #     idx = 0
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 U_sb[i * m + j, idx] = 1
    #                 V_sb[j * p + k, idx] = 1
    #                 W_sb[k * n + i, idx] = 1
    #                 idx += 1
    #     return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
                
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
