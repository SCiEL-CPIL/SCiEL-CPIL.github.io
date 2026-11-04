"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    schoolbook_rank = n * m * p  # 100
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build schoolbook as baseline
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # Strategy: For each candidate rank, try to find U, V, W with integer/half-integer entries
    # that satisfy: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # This is a hard problem. Let's try a heuristic: start from schoolbook and try to
    # eliminate one column at a time by redistributing its contribution.
    
    # For now, return schoolbook as the safe baseline.
    # The search for rank reduction is complex and may not succeed within time limits.
    
    # Convert to lists for the return format
    U_list = U_sb.tolist()
    V_list = V_sb.tolist()
    W_list = W_sb.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
