import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen U and V matrices
    # U rows: 0:a, 1:b, 2:c, 3:d
    # V rows: 0:e, 1:f, 2:g, 3:h
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0]
    ])
    
    V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ])
    
    # Solve for W
    # W shape: (4, 7)
    # Rows of W correspond to k*n + i
    # 0: k=0, i=0 (C00)
    # 1: k=0, i=1 (C10)
    # 2: k=1, i=0 (C01)
    # 3: k=1, i=1 (C11)
    
    W = np.zeros((4, rank))
    
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            # Build system A * w = b
            # Unknowns are W[w_idx, 0..6]
            # Equations for each i', j, j', k'
            A_mat = []
            b_vec = []
            for ip in range(n):
                for j in range(m):
                    for jp in range(m):
                        for kp in range(p):
                            # Coefficient of A_{ip, j} B_{jp, kp} in C_{i, k}
                            # Sum_r U[ip*m+j, r] * V[jp*p+kp, r] * W[w_idx, r]
                            row = np.zeros(rank)
                            for r in range(rank):
                                row[r] = U[ip * m + j, r] * V[jp * p + kp, r]
                            A_mat.append(row)
                            
                            # RHS is 1 if ip==i, kp==k, j==jp
                            if ip == i and kp == k and j == jp:
                                b_vec.append(1)
                            else:
                                b_vec.append(0)
            
            A_np = np.array(A_mat)
            b_np = np.array(b_vec)
            
            # Solve A_np * W[w_idx] = b_np
            # Since it's an exact decomposition, we can use lstsq or pinv
            w_sol, residuals, rank_sol, s = np.linalg.lstsq(A_np, b_np, rcond=None)
            W[w_idx] = w_sol
            
    # Convert to lists of integers if possible
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
