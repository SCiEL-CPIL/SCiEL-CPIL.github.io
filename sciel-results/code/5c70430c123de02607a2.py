"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Try rank-35: 19 terms for 4x4x4 + 16 terms for 4x4x1
    # Verify the rank-19 decomposition, fall back to rank-65 if invalid
    
    def build_rank35():
        rank = 35
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # 19 terms for 4x4x4 decomposition
        # Each term: (u_dict, v_dict, w_dict)
        # U index: i*4+j, V index: j*4+k, W index: k*4+i
        # Coefficients must be integers or half-integers
        
        terms_19 = [
            ({(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}, {(0,0):1,(2,2):1}),
            ({(1,0):1,(2,1):1}, {(0,0):1}, {(0,1):1,(2,1):-1}),
            ({(0,1):1,(1,0):-1}, {(0,0):1}, {(1,0):-1,(1,1):1}),
            ({(0,0):1,(2,2):-1}, {(0,1):1,(2,0):-1}, {(0,0):1,(2,2):-1}),
            ({(1,1):1,(3,0):1}, {(1,1):1,(3,0):1}, {(1,1):1,(3,0):1}),
            ({(0,1):1,(1,0):-1}, {(1,1):1}, {(0,1):-1,(0,3):1}),
            ({(1,0):1,(3,1):-1}, {(1,1):1}, {(1,0):1,(3,1):-1}),
            ({(2,0):1,(3,1):1}, {(3,0):1}, {(2,0):1,(3,1):1}),
            ({(0,2):1,(1,3):1}, {(0,2):1,(1,3):1}, {(0,2):1,(1,3):1}),
            ({(2,2):1,(3,3):-1}, {(0,2):1}, {(2,2):1,(3,3):-1}),
            ({(0,3):1,(1,2):-1}, {(0,2):1}, {(0,3):-1,(0,2):1}),
            ({(2,3):1,(3,2):-1}, {(1,3):1}, {(2,3):1,(3,2):-1}),
            ({(0,0):1,(1,1):-1}, {(2,0):1}, {(0,0):1,(1,1):-1}),
            ({(2,0):1,(3,1):-1}, {(0,0):1}, {(2,0):1,(3,1):-1}),
            ({(0,1):1,(1,0):-1}, {(2,0):1}, {(0,1):1,(1,0):-1}),
            ({(0,2):1,(1,3):-1}, {(2,2):1}, {(0,2):1,(1,3):-1}),
            ({(2,2):1,(3,3):-1}, {(0,2):1}, {(2,2):1,(3,3):-1}),
            ({(0,3):1,(1,2):-1}, {(2,2):1}, {(0,3):1,(1,2):-1}),
            ({(2,3):1,(3,2):-1}, {(1,3):1}, {(2,3):1,(3,2):-1}),
        ]
        
        col = 0
        for u_dict, v_dict, w_dict in terms_19:
            for (i, j), c in u_dict.items():
                U[i * m + j][col] += c
            for (j, k), c in v_dict.items():
                V[j * p + k][col] += c
            for (k, i), c in w_dict.items():
                W[k * n + i][col] += c
            col += 1
        
        # 16 schoolbook terms for 4x4 * 4x1 (5th column)
        for i in range(4):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        return U, V, W, rank
    
    def build_rank65():
        rank = 65
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(col, u_list, v_list, w_list):
            for i, j, c in u_list:
                if c != 0:
                    U[i * m + j][col] += c
            for j, k, c in v_list:
                if c != 0:
                    V[j * p + k][col] += c
            for k, i, c in w_list:
                if c != 0:
                    W[k * n + i][col] += c
        
        def get_strassen_4x4(super_idx, inner_idx):
            u_list = []
            v_list = []
            w_list = []
            
            p_blocks = [0, 0, 0, 0]
            q_blocks = [0, 0, 0, 0]
            
            if super_idx == 0:
                p_blocks = [1, 0, 0, 1]
                q_blocks = [1, 0, 0, 1]
            elif super_idx == 1:
                p_blocks = [0, 0, 1, 1]
                q_blocks = [1, 0, 0, 0]
            elif super_idx == 2:
                p_blocks = [1, 0, 0, 0]
                q_blocks = [0, 1, 0, -1]
            elif super_idx == 3:
                p_blocks = [0, 0, 0, 1]
                q_blocks = [-1, 0, 1, 0]
            elif super_idx == 4:
                p_blocks = [1, 1, 0, 0]
                q_blocks = [0, 0, 0, 1]
            elif super_idx == 5:
                p_blocks = [-1, 0, 1, 0]
                q_blocks = [1, 1, 0, 0]
            elif super_idx == 6:
                p_blocks = [0, 1, 0, -1]
                q_blocks = [0, 0, 1, 1]
            
            if inner_idx == 0:
                p_elems = [(0, 1), (3, 1)]
                q_elems = [(0, 1), (3, 1)]
            elif inner_idx == 1:
                p_elems = [(2, 1), (3, 1)]
                q_elems = [(0, 1)]
            elif inner_idx == 2:
                p_elems = [(0, 1)]
                q_elems = [(1, 1), (3, -1)]
            elif inner_idx == 3:
                p_elems = [(3, 1)]
                q_elems = [(2, 1), (0, -1)]
            elif inner_idx == 4:
                p_elems = [(0, 1), (1, 1)]
                q_elems = [(3, 1)]
            elif inner_idx == 5:
                p_elems = [(2, 1), (0, -1)]
                q_elems = [(0, 1), (1, 1)]
            elif inner_idx == 6:
                p_elems = [(1, 1), (3, -1)]
                q_elems = [(2, 1), (3, 1)]
            
            def get_p_elem(p_elem_idx):
                res = []
                r = p_elem_idx // 2
                c = p_elem_idx % 2
                for b_idx, coeff in enumerate(p_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gi, gj = r, c
                    elif b_idx == 1:
                        gi, gj = r, 2 + c
                    elif b_idx == 2:
                        gi, gj = 2 + r, c
                    elif b_idx == 3:
                        gi, gj = 2 + r, 2 + c
                    res.append((gi, gj, coeff))
                return res
            
            def get_q_elem(q_elem_idx):
                res = []
                r = q_elem_idx // 2
                c = q_elem_idx % 2
                for b_idx, coeff in enumerate(q_blocks):
                    if coeff == 0:
                        continue
                    if b_idx == 0:
                        gj, gk = r, c
                    elif b_idx == 1:
                        gj, gk = r, 2 + c
                    elif b_idx == 2:
                        gj, gk = 2 + r, c
                    elif b_idx == 3:
                        gj, gk = 2 + r, 2 + c
                    res.append((gj, gk, coeff))
                return res
            
            for p_idx, p_coeff in p_elems:
                for i, j, c in get_p_elem(p_idx):
                    u_list.append((i, j, c * p_coeff))
            
            for q_idx, q_coeff in q_elems:
                for j, k, c in get_q_elem(q_idx):
                    v_list.append((j, k, c * q_coeff))
            
            if inner_idx == 0:
                s_contrib = [1, 0, 0, 1]
            elif inner_idx == 1:
                s_contrib = [0, 0, 1, -1]
            elif inner_idx == 2:
                s_contrib = [0, 1, 0, 1]
            elif inner_idx == 3:
                s_contrib = [1, 0, 1, 0]
            elif inner_idx == 4:
                s_contrib = [-1, 1, 0, 0]
            elif inner_idx == 5:
                s_contrib = [0, 0, 0, 1]
            elif inner_idx == 6:
                s_contrib = [1, 0, 0, 0]
            
            if super_idx == 0:
                c_block_contrib = [1, 0, 0, 1]
            elif super_idx == 1:
                c_block_contrib = [0, 0, 1, -1]
            elif super_idx == 2:
                c_block_contrib = [0, 1, 0, 1]
            elif super_idx == 3:
                c_block_contrib = [1, 0, 1, 0]
            elif super_idx == 4:
                c_block_contrib = [-1, 1, 0, 0]
            elif super_idx == 5:
                c_block_contrib = [0, 0, 0, 1]
            elif super_idx == 6:
                c_block_contrib = [1, 0, 0, 0]
            
            for s_pos in range(4):
                s_coeff = s_contrib[s_pos]
                if s_coeff == 0:
                    continue
                s_r = s_pos // 2
                s_c = s_pos % 2
                for c_b in range(4):
                    c_coeff = c_block_contrib[c_b]
                    if c_coeff == 0:
                        continue
                    total_coeff = s_coeff * c_coeff
                    if total_coeff == 0:
                        continue
                    if c_b == 0:
                        gi, gk = s_r, s_c
                    elif c_b == 1:
                        gi, gk = s_r, 2 + s_c
                    elif c_b == 2:
                        gi, gk = 2 + s_r, s_c
                    elif c_b == 3:
                        gi, gk = 2 + s_r, 2 + s_c
                    w_list.append((gk, gi, total_coeff))
            
            return u_list, v_list, w_list
        
        col = 0
        for s in range(7):
            for t in range(7):
                u_list, v_list, w_list = get_strassen_4x4(s, t)
                add_term(col, u_list, v_list, w_list)
                col += 1
        
        for i in range(4):
            for j in range(4):
                u_list = [(i, j, 1)]
                v_list = [(j, 4, 1)]
                w_list = [(4, i, 1)]
                add_term(col, u_list, v_list, w_list)
                col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Verify the decomposition against random matrices"""
        np.random.seed(42)
        for trial in range(10):
            A = np.random.randint(-5, 6, (n, m))
            B = np.random.randint(-5, 6, (m, p))
            C_expected = A @ B
            
            # Compute C from decomposition
            # C[i,k] = sum over rank of U[i*m+j] * V[j*p+k] * W[k*n+i]
            # But this is O(n*m*p*rank) which is fine for small dims
            C_computed = np.zeros((n, p), dtype=float)
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            for i in range(n):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        for j in range(m):
                            total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    C_computed[i, k] = total
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True
    
    # Try rank-35 first
    U35, V35, W35, rank35 = build_rank35()
    if verify(U35, V35, W35, rank35):
        return {"U": U35, "V": V35, "W": W35}
    
    # Fall back to rank-65
    U65, V65, W65, rank65 = build_rank65()
    return {"U": U65, "V": V65, "W": W65}
# EVOLVE-BLOCK-END
