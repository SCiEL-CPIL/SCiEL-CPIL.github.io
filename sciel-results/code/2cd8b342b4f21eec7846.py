import time
import random
import itertools
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Build the structured Strassen/Schoolbook decomposition (rank 39)
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
        add_term(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0,col_offset): 1},
            {w_idx(col_offset,1): 1, w_idx(col_offset+1,1): -1}
        )
        add_term(
            {u_idx(0,0): 1},
            {v_idx(0,col_offset+1): 1, v_idx(1,col_offset+1): -1},
            {w_idx(col_offset+1,0): 1, w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(1,1): 1},
            {v_idx(1,col_offset): 1, v_idx(0,col_offset): -1},
            {w_idx(col_offset,0): 1, w_idx(col_offset,1): 1}
        )
        add_term(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): -1, w_idx(col_offset+1,0): 1}
        )
        add_term(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0,col_offset): 1, v_idx(0,col_offset+1): 1},
            {w_idx(col_offset+1,1): 1}
        )
        add_term(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1,col_offset): 1, v_idx(1,col_offset+1): 1},
            {w_idx(col_offset,0): 1}
        )
    
    def schoolbook_2x1(col_k):
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
    
    strassen_block(0)
    schoolbook_2x1(2)
    strassen_block(3)
    strassen_block(5)
    
    for col_k in range(7):
        add_term(
            {u_idx(0,2): 1},
            {v_idx(2,col_k): 1},
            {w_idx(col_k,0): 1}
        )
        add_term(
            {u_idx(1,2): 1},
            {v_idx(2,col_k): 1},
            {w_idx(col_k,1): 1}
        )
    
    rank = len(terms)
    
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
    
    # Try to improve the 2x2x7 part
    if time.time() < deadline - 20:
        # Subproblem: 2x2x7
        # n_sub=2, m_sub=2, p_sub=7
        n_sub, m_sub, p_sub = 2, 2, 7
        U_size = n_sub * m_sub # 4
        V_size = m_sub * p_sub # 14
        W_size = p_sub * n_sub # 14
        
        def solve_subproblem(target_rank, max_iters=50):
            # Random init
            U_sub = np.random.randn(U_size, target_rank)
            V_sub = np.random.randn(V_size, target_rank)
            W_sub = np.random.randn(W_size, target_rank)
            
            # Target tensor T
            # T[i,j,k] = 1
            # Flatten T to vector of length 28
            T_flat = np.ones(28)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Fix W, solve for U and V
                # This is a linear system for each rank component? No, it's coupled.
                # We can use ALS.
                # Minimize || T - sum_r U_r V_r^T W_r ||
                
                # Update U:
                # For each i,j, the equation is sum_r U[i,j,r] (V_r * W_r)_{j,k}?
                # Actually, C[i,k] = sum_j A[i,j] B[j,k].
                # In the decomposition, the contribution to C[i,k] is sum_r sum_j U[i*m+j,r] A[i,j] V[j*p+k,r] ...
                # Wait, the tensor T is defined such that sum_r U[u,r] V[v,r] W[w,r] = T[u,v,w].
                # Here u=(i,j), v=(j,k), w=(k,i).
                # Note that j is shared between u and v.
                # This is not a standard CP decomposition because the indices are not independent in the same way?
                # No, u, v, w are just indices into the matrices.
                # The constraint is that the middle index j must match.
                # But in the flattened vectors U, V, W, this is handled by the specific indices we pick.
                # The tensor T has entries T[i, j, k] = 1.
                # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1.
                
                # This is a linear system in U if V, W are fixed.
                # Variables: U[4, R].
                # Equations: 28.
                # Matrix A of size 28 x (4*R).
                # For each equation (i,j,k) = 1:
                # sum_r U[i, j, r] * (V[j, k, r] * W[k, i, r]) = 1.
                
                # Let's build the linear system for U.
                # Unknowns: U_flat of length 4*R.
                # Row for (i,j,k):
                # Coeff for U[i,j,r] is V[j,k,r] * W[k,i,r].
                # Other U entries are 0.
                
                A_U = np.zeros((28, U_size * target_rank))
                b_U = np.ones(28)
                
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            # Equation for T[i,j,k]
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = V_sub[v_idx_val, r] * W_sub[w_idx_val, r]
                                A_U[idx, u_idx_val * target_rank + r] = coeff
                            idx += 1
                
                # Solve least squares
                U_sub, residuals, rank_u, s = np.linalg.lstsq(A_U, b_U, rcond=None)
                U_sub = U_sub.reshape(U_size, target_rank)
                
                # Update V
                A_V = np.zeros((28, V_size * target_rank))
                b_V = np.ones(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = U_sub[u_idx_val, r] * W_sub[w_idx_val, r]
                                A_V[idx, v_idx_val * target_rank + r] = coeff
                            idx += 1
                V_sub, residuals, rank_v, s = np.linalg.lstsq(A_V, b_V, rcond=None)
                V_sub = V_sub.reshape(V_size, target_rank)
                
                # Update W
                A_W = np.zeros((28, W_size * target_rank))
                b_W = np.ones(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            for r in range(target_rank):
                                u_idx_val = i * m_sub + j
                                v_idx_val = j * p_sub + k
                                w_idx_val = k * n_sub + i
                                coeff = U_sub[u_idx_val, r] * V_sub[v_idx_val, r]
                                A_W[idx, w_idx_val * target_rank + r] = coeff
                            idx += 1
                W_sub, residuals, rank_w, s = np.linalg.lstsq(A_W, b_W, rcond=None)
                W_sub = W_sub.reshape(W_size, target_rank)
                
                # Check error
                # Reconstruct T
                T_rec = np.zeros(28)
                idx = 0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            u_idx_val = i * m_sub + j
                            v_idx_val = j * p_sub + k
                            w_idx_val = k * n_sub + i
                            T_rec[idx] = np.sum(U_sub[u_idx_val] * V_sub[v_idx_val] * W_sub[w_idx_val])
                            idx += 1
                
                error = np.linalg.norm(T_rec - T_flat)
                if error < 1e-9:
                    return U_sub, V_sub, W_sub
            
            return None
        
        # Try ranks 24, 23, 22...
        for target_rank in range(24, 13, -1):
            if time.time() > deadline - 10:
                break
            
            # Try a few restarts
            found = False
            for _ in range(5):
                res = solve_subproblem(target_rank, max_iters=20)
                if res is not None:
                    U_sub, V_sub, W_sub = res
                    # Construct full decomposition
                    # Top part: 2x2x7 (uses A cols 0,1 and B rows 0,1)
                    # Bottom part: 2x1x7 (uses A col 2 and B row 2)
                    
                    new_rank = target_rank + 14
                    U_new = [[0] * new_rank for _ in range(n * m)]
                    V_new = [[0] * new_rank for _ in range(m * p)]
                    W_new = [[0] * new_rank for _ in range(p * n)]
                    
                    # Fill top part
                    for r in range(target_rank):
                        for i in range(n_sub):
                            for j in range(m_sub):
                                U_new[i * m + j][r] = U_sub[i * m_sub + j, r]
                        for j in range(m_sub):
                            for k in range(p_sub):
                                V_new[j * p + k][r] = V_sub[j * p_sub + k, r]
                        for k in range(p_sub):
                            for i in range(n_sub):
                                W_new[k * n + i][r] = W_sub[k * n_sub + i, r]
                    
                    # Fill bottom part (Schoolbook 2x1x7)
                    offset = target_rank
                    for col_k in range(7):
                        # C0 += a02 * B2k
                        U_new[0 * m + 2][offset] = 1
                        V_new[2 * p + col_k][offset] = 1
                        W_new[col_k * n + 0][offset] = 1
                        offset += 1
                        # C1 += a12 * B2k
                        U_new[1 * m + 2][offset] = 1
                        V_new[2 * p + col_k][offset] = 1
                        W_new[col_k * n + 1][offset] = 1
                        offset += 1
                    
                    if check_decomposition(U_new, V_new, W_new, new_rank):
                        best_rank = new_rank
                        best_result = {"U": U_new, "V": V_new, "W": W_new}
                        found = True
                        break
            if found:
                break
                
    return best_result
