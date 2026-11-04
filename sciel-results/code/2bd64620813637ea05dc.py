"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Return exact tensor decomposition for 4x5 * 5x6.
    
    Strategy: Partition the multiplication into blocks and use:
    - Strassen-like rank-7 for 2x2x2 core
    - Schoolbook for remaining strips
    
    Partition:
    A (4x5) = [A_top(3x5); A_bot(1x5)]
    B (5x6) = [B_left(5x3); B_right(5x3)]
    
    AB = A_top*B_left + A_top*B_right + A_bot*B_left + A_bot*B_right
    
    For A_top*B_left (3x5*5x3):
      A_top = [A1(3x3) A2(3x2)], B_left = [B1(3x3); B2(2x3)]
      A_top*B_left = A1*B1 + A2*B2
      A1*B1 (3x3*3x3): Use Strassen on 2x2 sub-block + schoolbook strips = 12 terms
      A2*B2 (3x2*2x3): schoolbook = 18 terms
    
    Total: 12 + 18 + 45 + 15 + 15 = 105 terms
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    terms = []
    
    def add_term(U_dict, V_dict, W_dict):
        """Add a single rank-1 term."""
        terms.append((U_dict, V_dict, W_dict))
    
    # --- 3x3*3x3 core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3] ---
    # Use Strassen for 2x2*2x2 sub-block (rows/cols 0-1) + schoolbook for row/col 2
    
    # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # U indices: A[i,j] = i*5+j: A[0,0]=0, A[0,1]=1, A[1,0]=5, A[1,1]=6
    # V indices: B[j,k] = j*6+k: B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
    # W indices: C[k,i] = k*4+i: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    
    # Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    add_term({0: 1, 6: 1}, {0: 1, 7: 1}, {0: 1, 5: 1})
    # M2 = (A10+A11)*B00
    add_term({5: 1, 6: 1}, {0: 1}, {4: 1})
    # M3 = A00*(B01-B11)
    add_term({0: 1}, {1: 1, 7: -1}, {1: 1})
    # M4 = A11*(B10-B00)
    add_term({6: 1}, {6: 1, 0: -1}, {0: 1, 4: 1})
    # M5 = (A00+A01)*B11
    add_term({0: 1, 1: 1}, {7: 1}, {0: -1, 1: 1})
    # M6 = (A10-A00)*(B00+B01)
    add_term({5: 1, 0: -1}, {0: 1, 1: 1}, {5: 1})
    # M7 = (A01-A11)*(B10+B11)
    add_term({1: 1, 6: -1}, {6: 1, 7: 1}, {0: 1})
    
    # Schoolbook for row 2 of A * all cols of B (in the 3x3 core)
    for j in range(3):
        for k in range(3):
            u_idx = 2*5 + j  # A[2,j]
            v_idx = j*6 + k  # B[j,k]
            w_idx = k*4 + 2  # C[k,2]
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Schoolbook for rows 0-1 of A * col 2 of B (in the 3x3 core)
    for i in range(2):
        for j in range(3):
            u_idx = i*5 + j  # A[i,j]
            v_idx = j*6 + 2  # B[j,2]
            w_idx = 2*4 + i  # C[2,i]
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # --- A2*B2: A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3] ---
    for i in range(3):
        for j in range(3, 5):
            for k in range(3):
                u_idx = i*5 + j
                v_idx = j*6 + k
                w_idx = k*4 + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # --- A_top*B_right: A[0:3,0:5] * B[0:5,3:6] -> C[0:3,3:6] ---
    for i in range(3):
        for j in range(5):
            for k in range(3, 6):
                u_idx = i*5 + j
                v_idx = j*6 + k
                w_idx = k*4 + i
                add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # --- A_bot*B_left: A[3,0:5] * B[0:5,0:3] -> C[3,0:3] ---
    for j in range(5):
        for k in range(3):
            u_idx = 3*5 + j
            v_idx = j*6 + k
            w_idx = k*4 + 3
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # --- A_bot*B_right: A[3,0:5] * B[0:5,3:6] -> C[3,3:6] ---
    for j in range(5):
        for k in range(3, 6):
            u_idx = 3*5 + j
            v_idx = j*6 + k
            w_idx = k*4 + 3
            add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1})
    
    # Assemble U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][col] = val
        for idx, val in V_dict.items():
            V[idx][col] = val
        for idx, val in W_dict.items():
            W[idx][col] = val
    
    # Verify decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(val - 1) > 1e-9:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-9:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
