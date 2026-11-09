import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Define the 8 terms for the (2,3,2) sub-problem
    # Each term is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # U indices are (i, j) for A
    # V indices are (j, k_local) for B
    # W indices are (k_local, i) for C
    
    terms = []
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    terms.append(([ (0,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1]))
    
    # Term 2: M2 = (A10+A11)B01
    # U: (1,0)=1, (1,1)=1
    # V: (0,1)=1
    # W: (0,1)=1, (1,1)=-1
    terms.append(([ (1,0), (1,1) ], [1, 1], [ (0,1) ], [1], [ (0,1), (1,1) ], [1, -1]))
    
    # Term 3: M3 = A00(B10-B11)
    # U: (0,0)=1
    # V: (1,0)=1, (1,1)=-1
    # W: (1,0)=1, (1,1)=1
    terms.append(([ (0,0) ], [1], [ (1,0), (1,1) ], [1, -1], [ (1,0), (1,1) ], [1, 1]))
    
    # Term 4: M4 = A11(B00-B01)
    # U: (1,1)=1
    # V: (0,0)=1, (0,1)=-1
    # W: (0,0)=1, (0,1)=1
    terms.append(([ (1,1) ], [1], [ (0,0), (0,1) ], [1, -1], [ (0,0), (0,1) ], [1, 1]))
    
    # Term 5: M5 = (A00+A01)B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (0,0)=-1, (1,0)=1
    terms.append(([ (0,0), (0,1) ], [1, 1], [ (1,1) ], [1], [ (0,0), (1,0) ], [-1, 1]))
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1
    terms.append(([ (1,0), (0,0) ], [1, -1], [ (0,0), (0,1) ], [1, 1], [ (1,1) ], [1]))
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    terms.append(([ (0,1), (1,1) ], [1, -1], [ (1,0), (1,1) ], [1, 1], [ (0,0), (1,1) ], [1, 1]))
    
    # Term 8: J2 = A_i2 B_2k
    # U: (0,2)=1, (1,2)=1
    # V: (2,0)=1, (2,1)=1
    # W: (0,0)=1, (1,0)=1, (0,1)=1, (1,1)=1
    terms.append(([ (0,2), (1,2) ], [1, 1], [ (2,0), (2,1) ], [1, 1], [ (0,0), (1,0), (0,1), (1,1) ], [1, 1, 1, 1]))

    for b in range(3):
        k_offset = 2 * b
        for u_idx_list, u_vals, v_idx_list, v_vals, w_idx_list, w_vals in terms:
            # Add to U
            for (i, j), val in zip(u_idx_list, u_vals):
                U[i * m + j][col] += val
            
            # Add to V
            for (j, k_local), val in zip(v_idx_list, v_vals):
                V[j * p + (k_offset + k_local)][col] += val
            
            # Add to W
            for (k_local, i), val in zip(w_idx_list, w_vals):
                W[(k_offset + k_local) * n + i][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
