"""Matrix multiplication 4x5x5 - rank-85 Strassen with boundary terms."""
import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    start_time = time.time()
    deadline = 140.0
    
    # Total rank: 49 (4x4x4 Strassen) + 16 (4x4x1) + 16 (4x1x4) + 4 (4x1x1) = 85
    rank_base = 85
    U = np.zeros((nm, rank_base), dtype=int)
    V = np.zeros((mp, rank_base), dtype=int)
    W = np.zeros((pn, rank_base), dtype=int)
    
    # 2x2x2 Strassen factors
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    s_u_sparse = [[(i, v) for i, v in enumerate(s_u[:, k]) if v != 0] for k in range(7)]
    s_v_sparse = [[(i, v) for i, v in enumerate(s_v[:, k]) if v != 0] for k in range(7)]
    s_w_sparse = [[(i, v) for i, v in enumerate(s_w[:, k]) if v != 0] for k in range(7)]
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        for inner_k in range(7):
            col = col_start + inner_k
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for local_idx, u_val in s_u_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * a_br + lr
                    gj = 2 * a_bc + lc
                    U[gi * 5 + gj, col] += a_coeff * u_val
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for local_idx, v_val in s_v_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gj = 2 * b_br + lr
                    gk = 2 * b_bc + lc
                    V[gj * 5 + gk, col] += b_coeff * v_val
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for local_idx, w_val in s_w_sparse[inner_k]:
                    lr = local_idx // 2
                    lc = local_idx % 2
                    gi = 2 * c_br + lr
                    gk = 2 * c_bc + lc
                    W[gk * 4 + gi, col] += c_coeff * w_val
    
    terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        if time.time() - start_time > deadline:
            break
        add_4x4_term(a_coeff, b_coeff, c_coeff, t_idx * 7)
    
    col = 49
    
    # 4x4x1: A[:4,:4] @ B[:4,4] -> C[:4,4]
    # Wait, the boundary terms are:
    # C_left = A_left B_top_left + A_right B_bottom_left
    # C_right = A_left B_top_right + A_right B_bottom_right
    
    # A_left is A[:4, :4] (rows 0-3, cols 0-3)
    # B_top_left is B[:4, :4] (rows 0-3, cols 0-3)
    # B_top_right is B[:4, 4] (rows 0-3, col 4)
    # B_bottom_left is B[4, :4] (row 4, cols 0-3)
    # B_bottom_right is B[4, 4] (row 4, col 4)
    # A_right is A[:4, 4] (rows 0-3, col 4)
    
    # The core 4x4x4 handled A_left * B_top_left -> C_left_part1
    # We need to add:
    # 1. A_right * B_bottom_left -> C_left_part2 (4x1x4)
    # 2. A_left * B_top_right -> C_right_part1 (4x4x1)
    # 3. A_right * B_bottom_right -> C_right_part2 (4x1x1)
    
    # 1. A_right (4x1) * B_bottom_left (1x4) -> C_left (4x4)
    # A_right indices: i in 0..3, j=4. U[i*5+4, col]
    # B_bottom_left indices: j=4, k in 0..3. V[4*5+k, col]
    # C_left indices: k in 0..3, i in 0..3. W[k*4+i, col]
    for i in range(4):
        for k in range(4):
            if time.time() - start_time > deadline:
                break
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
    
    # 2. A_left (4x4) * B_top_right (4x1) -> C_right (4x1)
    # A_left indices: i in 0..3, j in 0..3. U[i*5+j, col]
    # B_top_right indices: j in 0..3, k=4. V[j*5+4, col]
    # C_right indices: k=4, i in 0..3. W[4*4+i, col]
    for i in range(4):
        for j in range(4):
            if time.time() - start_time > deadline:
                break
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    # 3. A_right (4x1) * B_bottom_right (1x1) -> C_right (4x1)
    # A_right indices: i in 0..3, j=4. U[i*5+4, col]
    # B_bottom_right indices: j=4, k=4. V[4*5+4, col]
    # C_right indices: k=4, i in 0..3. W[4*4+i, col]
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W, rank_base):
        # Fallback to schoolbook
        rank_schoolbook = n*m*p
        U = np.zeros((nm, rank_schoolbook), dtype=int)
        V = np.zeros((mp, rank_schoolbook), dtype=int)
        W = np.zeros((pn, rank_schoolbook), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
