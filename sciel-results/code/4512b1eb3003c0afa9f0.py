"""Matrix multiplication 4x5x5 - rank-85 Strassen-core decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n  # 20, 25, 20
    
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Build the rank-85 decomposition
    # Core: 4x4x4 via Strassen (2x2x2 blocks) = 7 * 7 = 49 terms
    # Boundary 1: A[:, 0:4] * B[4, :] -> C[:, 0:4] = 4 * 4 = 16 terms
    # Boundary 2: A[:, 4] * B[0:4, :] -> C[:, 0:4] = 4 * 4 = 16 terms
    # Boundary 3: A[:, 0:4] * B[:, 4] -> C[:, 4] = 4 * 4 = 16 terms
    # Boundary 4: A[:, 4] * B[4, 4] -> C[:, 4] = 4 terms
    # Total: 49 + 16 + 16 + 16 + 4 = 101? No wait...
    # Let me recount:
    # Core 4x4x4: 49
    # A rows 0-3, col 4, B row 4, cols 0-3: C rows 0-3, cols 0-3 (4*4=16)
    # A rows 0-3, col 4, B row 4, col 4: C rows 0-3, col 4 (4*1=4)
    # A rows 0-3, cols 0-3, B row 4, col 4: C rows 0-3, col 4 (4*4=16)
    # Wait, let me be more careful.
    
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, k in 0..4
    # = sum_{j=0}^3 A[i,j] * B[j,k] + A[i,4] * B[4,k]
    
    # For k in 0..3:
    #   C[i,k] = sum_{j=0}^3 A[i,j] * B[j,k] + A[i,4] * B[4,k]
    #   First term: 4x4x4 core (rank 49 via Strassen)
    #   Second term: A[i,4] * B[4,k] for i in 0..3, k in 0..3 (rank 16)
    
    # For k = 4:
    #   C[i,4] = sum_{j=0}^3 A[i,j] * B[j,4] + A[i,4] * B[4,4]
    #   First term: A[i,j] * B[j,4] for i in 0..3, j in 0..3 (rank 16)
    #   Second term: A[i,4] * B[4,4] for i in 0..3 (rank 4)
    
    # Total rank: 49 + 16 + 16 + 4 = 85. Yes, that's correct.
    
    rank_base = 85
    U = np.zeros((nm, rank_base), dtype=int)
    V = np.zeros((mp, rank_base), dtype=int)
    W = np.zeros((pn, rank_base), dtype=int)
    
    # 2x2x2 Strassen factors
    s_u = np.zeros((4, 7), dtype=int)
    s_v = np.zeros((4, 7), dtype=int)
    s_w = np.zeros((4, 7), dtype=int)
    
    # Term 0: (a00+a11)(b00+b11)
    s_u[0,0]=1; s_u[3,0]=1
    s_v[0,0]=1; s_v[3,0]=1
    s_w[0,0]=1; s_w[3,0]=1
    
    # Term 1: (a10+a11)b00
    s_u[2,1]=1; s_u[3,1]=1
    s_v[0,1]=1
    s_w[2,1]=1; s_w[3,1]=-1
    
    # Term 2: a00(b01-b11)
    s_u[0,2]=1
    s_v[1,2]=1; s_v[3,2]=-1
    s_w[1,2]=1; s_w[3,2]=1
    
    # Term 3: a11(b10-b00)
    s_u[3,3]=1
    s_v[2,3]=1; s_v[0,3]=-1
    s_w[0,3]=1; s_w[2,3]=1
    
    # Term 4: (a00+a01)b11
    s_u[0,4]=1; s_u[1,4]=1
    s_v[3,4]=1
    s_w[0,4]=-1; s_w[1,4]=1
    
    # Term 5: (a10-a00)(b00+b01)
    s_u[0,5]=-1; s_u[2,5]=1
    s_v[0,5]=1; s_v[1,5]=1
    s_w[3,5]=1
    
    # Term 6: (a01-a11)(b10+b11)
    s_u[1,6]=1; s_u[3,6]=-1
    s_v[2,6]=1; s_v[3,6]=1
    s_w[0,6]=1
    
    def add_4x4_term(a_blocks, b_blocks, c_blocks, col_start):
        """
        Add a 4x4x4 Strassen term using 2x2x2 Strassen sub-factors.
        """
        for inner_k in range(7):
            col = col_start + inner_k
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            for (a_br, a_bc), a_coeff in a_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            gi = 2 * a_br + lr
                            gj = 2 * a_bc + lc
                            U[gi * 5 + gj, col] += a_coeff * u_val
            
            for (b_br, b_bc), b_coeff in b_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            gj = 2 * b_br + lr
                            gk = 2 * b_bc + lc
                            V[gj * 5 + gk, col] += b_coeff * v_val
            
            for (c_br, c_bc), c_coeff in c_blocks.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            gi = 2 * c_br + lr
                            gk = 2 * c_bc + lc
                            W[gk * 4 + gi, col] += c_coeff * w_val
    
    # 7 Strassen 4x4 terms for the 4x4x4 core
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
    
    # Boundary 1: A[i,4] * B[4,k] -> C[i,k] for i in 0..3, k in 0..3 (16 terms)
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
    
    # Boundary 2: A[i,j] * B[j,4] -> C[i,4] for i in 0..3, j in 0..3 (16 terms)
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
    
    # Boundary 3: A[i,4] * B[4,4] -> C[i,4] for i in 0..3 (4 terms)
    for i in range(4):
        if time.time() - start_time > deadline:
            break
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
    
    # Verify the decomposition
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(U_mat.shape[1]):
                        total += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if total != 1:
                        return False
        return True
    
    if not verify(U, V, W):
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
# EVOLVE-BLOCK-END
