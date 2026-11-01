import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 2, 4, 5
    
    # Rank 36: 4 Strassen blocks (7 each) + 2 Schoolbook blocks (4 each)
    rank_initial = 36
    
    U = np.zeros((n * m, rank_initial), dtype=float)
    V = np.zeros((m * p, rank_initial), dtype=float)
    W = np.zeros((p * n, rank_initial), dtype=float)
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11)
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00)
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # 4 Strassen blocks
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # 2x2x1 blocks (Schoolbook)
    # Block 5: A_L * B_col4 (indices 0,1 of A, 0,1 of B_col4 -> 4,5 of C? No, B is 4x5. Col 4 is index 4)
    # A rows 0,1. B col 4.
    # A[0,0]B[0,4] -> C[0,4]
    # A[0,1]B[1,4] -> C[0,4]
    # A[1,0]B[2,4] -> C[1,4]
    # A[1,1]B[3,4] -> C[1,4]
    # Wait, A is 2x4. B is 4x5.
    # A_L is A[0:2, 0:2]. B_L5 is B[0:2, 4].
    # A[0,0] * B[0,4] + A[0,1] * B[1,4] -> C[0,4]
    # A[1,0] * B[0,4] + A[1,1] * B[1,4] -> C[1,4]
    # This is for the first part of the last column.
    
    # The seed code used:
    # U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1;
    # u_idx(0,0) is A[0,0]. v_idx(0,4) is B[0,4]. w_idx(4,0) is C[4,0] i.e. C[0,4] (k=4, i=0).
    # Correct.
    
    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    
    # Block 6: A_R * B_col4
    # A_R is A[0:2, 2:4]. B_R5 is B[2:4, 4].
    # A[0,2] * B[2,4] + A[0,3] * B[3,4] -> C[0,4]
    # A[1,2] * B[2,4] + A[1,3] * B[3,4] -> C[1,4]
    
    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    
    def check_decomposition(U_mat, V_mat, W_mat, rank):
        if rank > 36:
            return False
        try:
            D = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(D[i*m+j, j*p+k, k*n+i] - 1.0) > 1e-9:
                            return False
        except Exception:
            return False
        return True
    
    best_rank = rank_initial
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try removing single terms
    for r in range(rank_initial):
        if time.time() > deadline:
            break
        idx = [x for x in range(rank_initial) if x != r]
        U_t = U[:, idx]; V_t = V[:, idx]; W_t = W[:, idx]
        if check_decomposition(U_t, V_t, W_t, rank_initial - 1):
            best_rank = rank_initial - 1
            best_U, best_V, best_W = U_t, V_t, W_t
            break
    
    # Try removing pairs
    if best_rank == rank_initial and time.time() < deadline:
        import itertools
        for r1, r2 in itertools.combinations(range(rank_initial), 2):
            if time.time() > deadline:
                break
            idx = [x for x in range(rank_initial) if x != r1 and x != r2]
            U_t = U[:, idx]; V_t = V[:, idx]; W_t = W[:, idx]
            if check_decomposition(U_t, V_t, W_t, rank_initial - 2):
                best_rank = rank_initial - 2
                best_U, best_V, best_W = U_t, V_t, W_t
                break
    
    U, V, W = best_U, best_V, best_W
    
    U_list = [[int(val) if val == int(val) else val for val in row] for row in U.tolist()]
    V_list = [[int(val) if val == int(val) else val for val in row] for row in V.tolist()]
    W_list = [[int(val) if val == int(val) else val for val in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
