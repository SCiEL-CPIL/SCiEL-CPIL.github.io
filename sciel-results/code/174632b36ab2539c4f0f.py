"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # Part 1: 1x4x6 (Rank 24)
    # Part 2: 2x4x6 (Rank 42) split into two 2x2x6 (Rank 21 each)
    # 2x2x6 split into three 2x2x2 (Rank 7 each) using Strassen
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Top row (i=0)
    # C[0, k] = sum_j A[0, j] B[j, k]
    # U index: 0*m + j = j
    # V index: j*p + k
    # W index: k*n + 0 = 3k
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1
            
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(u_idx, v_idx, w_idx, start_col):
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        c = start_col
        
        # M1 = (u00+u11)(v00+v11) -> w00+=1, w11+=1
        U[u_idx[0]][c] += 1
        U[u_idx[2]][c] += 1
        V[v_idx[0]][c] += 1
        V[v_idx[2]][c] += 1
        W[w_idx[0]][c] += 1
        W[w_idx[3]][c] += 1
        c += 1
        
        # M2 = (u10+u11)v00 -> w10+=1, w11-=1
        U[u_idx[2]][c] += 1
        U[u_idx[3]][c] += 1
        V[v_idx[0]][c] += 1
        W[w_idx[2]][c] += 1
        W[w_idx[3]][c] -= 1
        c += 1
        
        # M3 = u00(v01-v11) -> w01+=1, w11+=1
        U[u_idx[0]][c] += 1
        V[v_idx[1]][c] += 1
        V[v_idx[3]][c] -= 1
        W[w_idx[1]][c] += 1
        W[w_idx[3]][c] += 1
        c += 1
        
        # M4 = u11(v10-v00) -> w00+=1, w10+=1
        U[u_idx[3]][c] += 1
        V[v_idx[2]][c] += 1
        V[v_idx[0]][c] -= 1
        W[w_idx[0]][c] += 1
        W[w_idx[2]][c] += 1
        c += 1
        
        # M5 = (u00+u01)v11 -> w00-=1, w01+=1
        U[u_idx[0]][c] += 1
        U[u_idx[1]][c] += 1
        V[v_idx[3]][c] += 1
        W[w_idx[0]][c] -= 1
        W[w_idx[1]][c] += 1
        c += 1
        
        # M6 = (u10-u00)(v00+v01) -> w11+=1
        U[u_idx[2]][c] += 1
        U[u_idx[0]][c] -= 1
        V[v_idx[0]][c] += 1
        V[v_idx[1]][c] += 1
        W[w_idx[3]][c] += 1
        c += 1
        
        # M7 = (u01-u11)(v10+v11) -> w00+=1, w11-=1
        U[u_idx[1]][c] += 1
        U[u_idx[3]][c] -= 1
        V[v_idx[2]][c] += 1
        V[v_idx[3]][c] += 1
        W[w_idx[0]][c] += 1
        W[w_idx[3]][c] -= 1
        c += 1
        
        return c

    # Part 2: Bottom rows (i=1,2)
    # Split m=4 into two 2s.
    # Block 2a: A_L (rows 1,2; cols 0,1) * B_L (rows 0,1; cols 0..5)
    # A_L indices in U:
    # u00: i=1, j=0 -> 1*4+0=4
    # u01: i=1, j=1 -> 1*4+1=5
    # u10: i=2, j=0 -> 2*4+0=8
    # u11: i=2, j=1 -> 2*4+1=9
    u_L = [4, 5, 8, 9]
    
    # B_L is 2x6. Split p=6 into three 2x2 blocks.
    # For s in 0,1,2:
    # k_start = 2*s
    # v00: i=0, k=k_start -> 0*6+k_start
    # v01: i=0, k=k_start+1 -> 0*6+k_start+1
    # v10: i=1, k=k_start -> 1*6+k_start
    # v11: i=1, k=k_start+1 -> 1*6+k_start+1
    # W indices for C cols 1,2, rows k_start, k_start+1
    # w00: k=k_start, i=1 -> k_start*3+1
    # w01: k=k_start, i=2 -> k_start*3+2
    # w10: k=k_start+1, i=1 -> (k_start+1)*3+1
    # w11: k=k_start+1, i=2 -> (k_start+1)*3+2
    
    for s in range(3):
        k0 = 2 * s
        k1 = 2 * s + 1
        v_L = [k0, k1, 6 + k0, 6 + k1]
        w_L = [k0 * 3 + 1, k0 * 3 + 2, k1 * 3 + 1, k1 * 3 + 2]
        col = add_strassen(u_L, v_L, w_L, col)
        
    # Block 2b: A_R (rows 1,2; cols 2,3) * B_R (rows 2,3; cols 0..5)
    # A_R indices in U:
    # u00: i=1, j=2 -> 1*4+2=6
    # u01: i=1, j=3 -> 1*4+3=7
    # u10: i=2, j=2 -> 2*4+2=10
    # u11: i=2, j=3 -> 2*4+3=11
    u_R = [6, 7, 10, 11]
    
    # B_R is 2x6. Split p=6 into three 2x2 blocks.
    # For s in 0,1,2:
    # k_start = 2*s
    # v00: i=2, k=k_start -> 2*6+k_start
    # v01: i=2, k=k_start+1 -> 2*6+k_start+1
    # v10: i=3, k=k_start -> 3*6+k_start
    # v11: i=3, k=k_start+1 -> 3*6+k_start+1
    # W indices are the same as Block 2a because they add to the same C entries.
    
    for s in range(3):
        k0 = 2 * s
        k1 = 2 * s + 1
        v_R = [12 + k0, 12 + k1, 18 + k0, 18 + k1]
        w_R = [k0 * 3 + 1, k0 * 3 + 2, k1 * 3 + 1, k1 * 3 + 2]
        col = add_strassen(u_R, v_R, w_R, col)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
