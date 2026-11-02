import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p, tol=1e-6):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_m221_decomposition():
    """Find rank-3 decomposition of M_{2,2,1} tensor."""
    # Tensor T[i,j,k] where i in {0,1,2,3} (a,b,c,d), j in {0,1} (x,y), k in {0,1} (o1,o2)
    # T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1, rest 0
    # We want U (4x3), V (2x3), W (2x3) such that sum_c U[i,c]*V[j,c]*W[k,c] = T[i,j,k]
    
    # Set up as a linear system. Unknowns: U[i,c]*V[j,c]*W[k,c] for all i,j,k,c
    # This is nonlinear. Instead, fix V and solve for U,W.
    
    # Try fixing V to specific values and solving linear system for U and W.
    # Actually, let's use a different approach: parameterize and solve.
    
    # Let's try: V = [[1, 0, 1], [0, 1, 1]] (a common choice)
    V = np.array([[1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0]])
    
    # For each c, the contribution to T[i,j,k] is U[i,c]*V[j,c]*W[k,c]
    # Sum over c: T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    
    # Fix W and solve for U, or use iterative approach.
    # Let's fix W = [[1,1,0],[0,1,1]] and solve for U.
    # Then T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    # For fixed j,k: T[:,j,k] = V[:,j,c] * W[k,c] * U[:,c] summed over c
    # This is: T[:,j,k] = sum_c (V[j,c]*W[k,c]) * U[:,c]
    # So for each (j,k), we have a linear system in U[:,c].
    
    # Actually, let me just set up the full linear system.
    # Unknowns: U[0,0],U[0,1],U[0,2],U[1,0],...,U[3,2], W[0,0],W[0,1],W[0,2],W[1,0],W[1,1],W[1,2]
    # That's 12+6 = 18 unknowns, but the equations are bilinear in U and W.
    
    # Alternative: fix W and solve for U (linear), then check.
    # Let's try several W choices.
    
    for W_trial in [
        np.array([[1.0,1.0,0.0],[0.0,1.0,1.0]]),
        np.array([[1.0,0.0,1.0],[0.0,1.0,1.0]]),
        np.array([[1.0,1.0,1.0],[0.0,1.0,1.0]]),
    ]:
        # For each (j,k): T[:,j,k] = sum_c U[:,c] * (V[j,c]*W[k,c])
        # This is: T[:,j,k] = U @ diag-like... 
        # More precisely: T[:,j,k][i] = sum_c U[i,c] * V[j,c] * W[k,c]
        # So T[:,j,k] = U * (V[j,:]*W[k,:])  (element-wise product, then matrix-vector)
        # T[:,j,k] = U @ (V[j,:]*W[k,:])
        
        # Stack all (j,k) pairs: we have 4 equations for each of 4 (j,k) combos = 16 equations
        # Unknowns: U (4x3) = 12 unknowns
        # A (16x12) x = b (16)
        
        A_mat = np.zeros((16, 12))
        b_vec = np.zeros(16)
        
        for j in range(2):
            for k in range(2):
                row_base = j*2 + k
                # T[:,j,k] = U @ (V[j,:]*W_trial[k,:])
                coeff = V[j,:] * W_trial[k,:]  # shape (3,)
                for i in range(4):
                    row = i*4 + j*2 + k  # 16 rows: i*4 + j*2 + k
                    # Actually let me index properly: 4 i's, 2 j's, 2 k's = 16
                    row = i*4 + j*2 + k
                    for c in range(3):
                        A_mat[row, i*3+c] = coeff[c]
                    # T[i,j,k]
                    if (i,j,k) in [(0,0,0),(1,1,0),(2,0,1),(3,1,1)]:
                        b_vec[row] = 1.0
                    else:
                        b_vec[row] = 0.0
        
        # Solve A_mat @ u_flat = b_vec
        try:
            u_flat, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U_sol = u_flat.reshape(4,3)
            
            # Verify
            max_err = 0
            for i in range(4):
                for j in range(2):
                    for k in range(2):
                        val = sum(U_sol[i,c]*V[j,c]*W_trial[k,c] for c in range(3))
                        expected = 1.0 if (i,j,k) in [(0,0,0),(1,1,0),(2,0,1),(3,1,1)] else 0.0
                        max_err = max(max_err, abs(val - expected))
            
            if max_err < 1e-10:
                return U_sol.tolist(), V.tolist(), W_trial.tolist()
        except:
            continue
    
    return None

def _m229_decomposition():
    """Get decomposition for M_{2,2,9} using Winograd per column."""
    result = _find_m221_decomposition()
    if result is None:
        return None
    
    U221, V221, W221 = result
    # U221: 4x3 (for a,b,c,d)
    # V221: 2x3 (for x,y)
    # W221: 2x3 (for o1,o2)
    
    # For M_{2,2,9}: A is 2x2 (4 vars), B is 2x9 (18 vars), C is 2x9 (18 vars)
    # For each column j (0..8), we have a sub-problem M_{2,2,1}
    # B variables: x_j (top row, col j) = index j, y_j (bottom row, col j) = index 9+j
    # C variables: o1_j (top row, col j) = index j, o2_j (bottom row, col j) = index 9+j
    
    rank = 3 * 9  # 27
    U = [[0]*rank for _ in range(4)]  # 4 A-variables
    V = [[0]*rank for _ in range(18)]  # 18 B-variables
    W = [[0]*rank for _ in range(18)]  # 18 C-variables
    
    for j in range(9):
        base = j * 3
        for c in range(3):
            # A-variables: same for all columns
            for i in range(4):
                U[i][base+c] = U221[i][c]
            # B-variables: x_j is at index j, y_j is at index 9+j
            V[j][base+c] = V221[0][c]   # x_j
            V[9+j][base+c] = V221[1][c]  # y_j
            # C-variables: o1_j is at index j, o2_j is at index 9+j
            W[j][base+c] = W221[0][c]    # o1_j
            W[9+j][base+c] = W221[1][c]  # o2_j
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try block decomposition: split 4x4 A into 2x2 blocks, 4x9 B into 2x9 blocks
    # C = A*B where A = [[A11,A12],[A21,A22]], B = [[B1],[B2]]
    # C_top = A11*B1 + A12*B2 (2x9)
    # C_bot = A21*B1 + A22*B2 (2x9)
    # 4 sub-problems of M_{2,2,9}
    
    m229 = _m229_decomposition()
    if m229 is not None:
        # Combine 4 copies
        sub_rank = len(m229["U"][0])  # should be 27
        total_rank = 4 * sub_rank
        
        # Full U: 16 rows (4x4 A), total_rank cols
        # Full V: 36 rows (4x9 B), total_rank cols
        # Full W: 36 rows (9x4 C), total_rank cols
        
        U_full = [[0]*total_rank for _ in range(16)]
        V_full = [[0]*total_rank for _ in range(36)]
        W_full = [[0]*total_rank for _ in range(36)]
        
        # A block (bi, bk) where bi in {0,1} is row-block, bk in {0,1} is col-block
        # A index: bi*2 + bk_row, bk*2 + bk_col → A index = (bi*2 + r) * 4 + (bk*2 + c)
        # Wait, A is 4x4. A[i,k] where i in 0..3, k in 0..3.
        # Block A11: i in {0,1}, k in {0,1} → A indices 0,1,4,5
        # Block A12: i in {0,1}, k in {2,3} → A indices 2,3,6,7
        # Block A21: i in {2,3}, k in {0,1} → A indices 8,9,12,13
        # Block A22: i in {2,3}, k in {2,3} → A indices 10,11,14,15
        
        # B is 4x9. B[k,j] where k in 0..3, j in 0..8.
        # B1 (top 2 rows): k in {0,1}, j in 0..8 → B indices 0..17
        # B2 (bottom 2 rows): k in {2,3}, j in 0..8 → B indices 18..35
        
        # C is 4x9. C[i,j] where i in 0..3, j in 0..8.
        # W index: j*n + i = j*4 + i
        # C_top: i in {0,1}, j in 0..8 → W indices: j*4+0, j*4+1 for j=0..8
        # C_bot: i in {2,3}, j in 0..8 → W indices: j*4+2, j*4+3 for j=0..8
        
        # Sub-problem 1: A11 * B1 → contributes to C_top
        # A11 local indices: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3 in the 2x2 block
        # A11 global A indices: i*4+k for i in {0,1}, k in {0,1} → 0,1,4,5
        #   local (0,0)→global 0, (0,1)→global 1, (1,0)→global 4, (1,1)→global 5
        # B1 local indices: (0,j)→j, (1,j)→9+j for j in 0..8
        # B1 global B indices: k*9+j for k in {0,1}, j in 0..8 → 0..8, 9..17
        #   local (0,j)→global j, (1,j)→global 9+j
        # C_top local indices: (0,j)→j, (1,j)→9+j
        # C_top global W indices: j*4+i for i in {0,1}, j in 0..8
        #   local (0,j)→global j*4+0, (1,j)→global j*4+1
        
        blocks = [
            # (A_block_local_to_global, B_block_local_to_global, C_block_local_to_global)
            # A11 * B1 -> C_top
            ([0,1,4,5], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], 
             [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33]),
            # A12 * B2 -> C_top
            ([2,3,6,7], [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
             [0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,32,33]),
            # A21 * B1 -> C_bot
            ([8,9,12,13], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
             [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35]),
            # A22 * B2 -> C_bot
            ([10,11,14,15], [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
             [2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,34,35]),
        ]
        
        for block_idx, (a_map, b_map, w_map) in enumerate(blocks):
            base = block_idx * sub_rank
            for local_i, global_i in enumerate(a_map):
                for c in range(sub_rank):
                    U_full[global_i][base+c] = m229["U"][local_i][c]
            for local_j, global_j in enumerate(b_map):
                for c in range(sub_rank):
                    V_full[global_j][base+c] = m229["V"][local_j][c]
            for local_k, global_k in enumerate(w_map):
                for c in range(sub_rank):
                    W_full[global_k][base+c] = m229["W"][local_k][c]
        
        result = {"U": U_full, "V": V_full, "W": W_full}
        
        if _verify(U_full, V_full, W_full, n, m, p):
            return result
    
    # Fallback to schoolbook
    best = _schoolbook(n, m, p)
    assert _verify(best["U"], best["V"], best["W"], n, m, p), "Schoolbook failed"
    return best
