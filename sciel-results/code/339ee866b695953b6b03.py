"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _check_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 second deadline for safety
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 24
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not _check_decomposition(U_sb, V_sb, W_sb, n, m, p):
        # Should never happen
        pass
    
    # Try to find a lower-rank decomposition using a structured approach
    # For M_{2,3,4}, we can try rank 18 by decomposing into blocks
    
    # Strategy: Split B into two 3x2 blocks
    # M_{2,3,4} = M_{2,3,2} (first 2 cols) + M_{2,3,2} (last 2 cols)
    # If we can do M_{2,3,2} in rank 9, total rank = 18
    
    # Let's try to construct a rank-9 decomposition for M_{2,3,2}
    # A is 2x3, B is 3x2, C is 2x2
    # A = [[a0, a1, a2], [a3, a4, a5]]
    # B = [[b0, b1], [b2, b3], [b4, b5]]
    # C[0,0] = a0*b0 + a1*b2 + a2*b4
    # C[0,1] = a0*b1 + a1*b3 + a2*b5
    # C[1,0] = a3*b0 + a4*b2 + a5*b4
    # C[1,1] = a3*b1 + a4*b3 + a5*b5
    
    # We need 9 products to compute these 4 outputs
    # This is a tensor of size 6 x 6 x 4 = 144 entries, rank 9 means 9 terms
    
    # Let's use a known construction or try to find one
    # For now, let's just return schoolbook which is guaranteed correct
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
