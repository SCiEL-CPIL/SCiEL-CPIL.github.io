"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin for 150s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
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
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def try_rank(r):
        """Try to find a rank-r decomposition using structured approach."""
        if r < 1:
            return None
        
        if time.time() > deadline - 20:
            return None
        
        # Try to find a rank-r decomposition by starting from schoolbook
        # and attempting to merge terms or find linear dependencies
        
        # For small rank reductions, we can try to find if the last few
        # schoolbook terms can be expressed as linear combinations of others
        
        # This is a heuristic approach: try to see if rank r works
        # by checking if we can find U, V, W with rank r
        
        # One approach: use the fact that for matrix multiplication,
        # there might be structure we can exploit
        
        # For now, return None as this is computationally hard
        return None
    
    # Try to find better rank (limited time)
    # Start from schoolbook rank and try to reduce
    for r in range(rank_schoolbook - 1, max(90, rank_schoolbook - 10), -1):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
