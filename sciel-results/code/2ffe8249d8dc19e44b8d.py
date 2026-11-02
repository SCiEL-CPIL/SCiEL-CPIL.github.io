import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Build the structured Strassen/Schoolbook decomposition (rank 39)
    # A is 2x3, B is 3x7, C is 2x7
    # Split A = [A_left (2x2) | A_right (2x1)]
    # Split B = [B_top (2x7); B_bottom (1x7)]
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # For A_left @ B_top (2x2 * 2x7):
    #   Split B_top into columns: cols 0-1 (2x2), col 2 (2x1), cols 3-4 (2x2), cols 5-6 (2x2)
    #   Use Strassen for each 2x2 block, schoolbook for 2x1
    #
    # For A_right @ B_bottom (2x1 * 1x7): schoolbook (14 terms)
    
    terms = []
    
    def u_idx(i, j):
        return i * m + j
    
    def v_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    def add_term(u_dict, v_dict, w_dict):
        u_list = [0.0] * (n * m)
        v_list = [0.0] * (m * p)
        w_list = [0.0] * (p * n)
        for idx, val in u_dict.items():
            u_list[idx] = val
        for idx, val in v_dict.items():
            v_list[idx] = val
        for idx, val in w_dict.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
    
    def strassen_block(col_offset):
        """Apply Strassen for A_left (2x2) @ B_top[:, col_offset:col_offset+2] (2x2)"""
        # A_left = [[a00, a01], [a10, a11]]
        # B_block = [[b00, b01], [b10, b11]] where b00=B[0,col_offset], etc.
        
        # M1 = (a00 + a11)(b00 + b11)
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1, w_idx(col_offset+1,1): 1}
        )
        
        # M2 = (a10 + a11)b00
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1},
            {w_idx(col_offset,1): 1, w_idx(col_offset+1,1): -1}
        )
        
        # M3 = a00(b01 - b11)
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,col_offset+1): 1, v_idx(1,col_offset+1): -1},
            {w_idx(col_offset+1,0): 1, w_idx(col_offset+1,1): 1}
        )
        
        # M4 = a11(b10 - b00)
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,col_offset): 1, v_idx(0,col_offset): -1},
            {w_idx(col_offset,0): 1, w_idx(col_offset,1): 1}
        )
        
        # M5 = (a00 + a01)b11
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): -1, w_idx(col_offset+1,0): 1}
        )
        
        # M6 = (a10 - a00)(b00 + b01)
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,col_offset): 1, v_idx(0,col_offset+1): 1},
            {w_idx(col_offset+1,1): 1}
        )
        
        # M7 = (a01 - a11)(b10 + b11)
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1}
        )
    
    def schoolbook_2x1(col_k):
        """A_left (2x2) @ B_top[:, col_k] (2x1) using schoolbook (4 terms)"""
        # C0 = a00*b0k + a01*b1k
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,col_k): 1},
            {w_idx(col_k,0): 1}
        )
        add_term(
            {u_idx(0,1): 1},
            {v_idx(1,col_k): 1},
            {w_idx(col_k,0): 1}
        )
        # C1 = a10*b0k + a11*b1k
        add_term(
            {u_idx(1,0): 1},
            {v_idx(0,col_k): 1},
            {w_idx(col_k,1): 1}
        )
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,col_k): 1},
            {w_idx(col_k,1): 1}
        )
    
    # Part 1: A_left @ B_top[:,0:2] (2x2 * 2x2) - Strassen, 7 terms
    strassen_block(0)
    
    # Part 2: A_left @ B_top[:,2] (2x2 * 2x1) - schoolbook, 4 terms
    schoolbook_2x1(2)
    
    # Part 3: A_left @ B_top[:,3:5] (2x2 * 2x2) - Strassen, 7 terms
    strassen_block(3)
    
    # Part 4: A_left @ B_top[:,5:7] (2x2 * 2x2) - Strassen, 7 terms
    strassen_block(5)
    
    # Part 5: A_right @ B_bottom (2x1 * 1x7) - schoolbook, 14 terms
    for col_k in range(7):
        # C0 += a20 * B[2, col_k]
        add_term(
            {u_idx(0,2): 1},
            {v_idx(2,col_k): 1},
            {w_idx(col_k,0): 1}
        )
        # C1 += a21 * B[2, col_k]
        add_term(
            {u_idx(1,2): 1},
            {v_idx(2,col_k): 1},
            {w_idx(col_k,1): 1}
        )
    
    rank = len(terms)
    
    # Build U, V, W from terms
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (u_list, v_list, w_list) in enumerate(terms):
        for r in range(n * m):
            U[r][t] = u_list[r]
        for r in range(m * p):
            V[r][t] = v_list[r]
        for r in range(p * n):
            W[r][t] = w_list[r]
    
    # Verify the decomposition
    def check_decomposition(U_r, V_r, W_r, rank):
        U_arr = np.array(U_r, dtype=np.float64)
        V_arr = np.array(V_r, dtype=np.float64)
        W_arr = np.array(W_r, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx_val = i * m + j
                    v_idx_val = j * p + k
                    w_idx_val = k * n + i
                    prod = U_arr[u_idx_val] * V_arr[v_idx_val] * W_arr[w_idx_val]
                    if abs(np.sum(prod) - 1.0) > 1e-9:
                        return False
        return True
    
    if not check_decomposition(U, V, W, rank):
        # Fallback to trivial decomposition
        trivial_rank = n * m * p
        U = [[0] * trivial_rank for _ in range(n * m)]
        V = [[0] * trivial_rank for _ in range(m * p)]
        W = [[0] * trivial_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        rank = trivial_rank
    
    best_rank = rank
    best_result = {"U": [row[:] for row in U], "V": [row[:] for row in V], "W": [row[:] for row in W]}
    
    # Try to find even lower rank using random search from the structured solution
    if time.time() < deadline - 20:
        def try_lower_rank(target_rank, max_attempts=200):
            if time.time() > deadline - 10:
                return None
            
            # Start from the structured decomposition and try to merge terms
            # or use random search with the structured solution as initialization
            
            # Strategy: Try random perturbations of the structured solution
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                
                U_r = [[0] * target_rank for _ in range(n * m)]
                V_r = [[0] * target_rank for _ in range(m * p)]
                W_r = [[0] * target_rank for _ in range(p * n)]
                
                # Initialize with random values in a small range
                val_range = [-2, -1, 0, 1, 2]
                for r in range(n * m):
                    for t in range(target_rank):
                        U_r[r][t] = random.choice(val_range)
                for r in range(m * p):
                    for t in range(target_rank):
                        V_r[r][t] = random.choice(val_range)
                for r in range(p * n):
                    for t in range(target_rank):
                        W_r[r][t] = random.choice(val_range)
                
                if check_decomposition(U_r, V_r, W_r, target_rank):
                    return {"U": U_r, "V": V_r, "W": W_r}
            
            return None
        
        # Try ranks just below the structured rank
        for target_rank in range(rank - 1, max(30, rank - 5), -1):
            if time.time() > deadline - 10:
                break
            
            result = try_lower_rank(target_rank, max_attempts=100)
            if result is not None:
                best_rank = target_rank
                best_result = result
                break
    
    return best_result
