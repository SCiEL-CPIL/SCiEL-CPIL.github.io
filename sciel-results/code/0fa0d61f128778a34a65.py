import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U: (n*m, R) = (8, R)
    # V: (m*p, R) = (28, R)
    # W: (p*n, R) = (14, R)
    
    # We expect R = 50
    R = 50
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    col = 0
    
    # Define Strassen terms for 2x2x2
    # u: 4 elements, v: 4 elements, w: 4 elements
    strassen_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    # Define Schoolbook terms for 2x2x1
    # u: 4 elements, v: 2 elements, w: 2 elements
    schoolbook_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    # Iterate over j-blocks (2 blocks of size 2)
    for j_block in range(2):
        # Iterate over k-blocks (3 blocks of size 2, 1 block of size 1)
        # k ranges 0..6
        k_blocks = [
            (0, 2, 'strassen'), # k=0,1
            (2, 4, 'strassen'), # k=2,3
            (4, 6, 'strassen'), # k=4,5
            (6, 7, 'schoolbook') # k=6
        ]
        
        for k_start, k_end, method in k_blocks:
            if method == 'strassen':
                terms = strassen_terms
                k_size = 2
            else:
                terms = schoolbook_terms
                k_size = 1
                
            for u_local, v_local, w_local in terms:
                # u_local is for A (2x2)
                # v_local is for B (2xk_size)
                # w_local is for C (2xk_size)
                
                for i in range(2):
                    for j_loc in range(2):
                        val_u = u_local[i*2 + j_loc]
                        if val_u != 0:
                            # Global A index
                            ga = i * m + (j_block * 2 + j_loc)
                            U[ga, col] += val_u
                            
                    for j_loc in range(2):
                        for k_loc in range(k_size):
                            val_v = v_local[j_loc * k_size + k_loc]
                            if val_v != 0:
                                # Global B index
                                gb = (j_block * 2 + j_loc) * p + (k_start + k_loc)
                                V[gb, col] += val_v
                                
                    for k_loc in range(k_size):
                        for i in range(2):
                            val_w = w_local[k_loc * 2 + i]
                            if val_w != 0:
                                # Global C index
                                # W[k*n+i]
                                gc = (k_start + k_loc) * n + i
                                W[gc, col] += val_w
                                
                col += 1
                
    return {"U": U, "V": V, "W": W}
