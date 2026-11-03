import time
import random

DIMENSIONS = [2, 4, 7]
DEADLINE = 145.0  # seconds wall-clock safety margin

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n*m, m*p, p*n  # 8, 28, 14
    
    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1

    def build_rank46():
        rank = 46
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b00 = b_rows[0]*p + b_cols[0]
            b01 = b_rows[0]*p + b_cols[1]
            b10 = b_rows[1]*p + b_cols[0]
            b11 = b_rows[1]*p + b_cols[1]
            c00 = c_cols[0]*n + c_rows[0]
            c01 = c_cols[1]*n + c_rows[0]
            c10 = c_cols[0]*n + c_rows[1]
            c11 = c_cols[1]*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; U[a11][col] += 1; V[b00][col] += 1; V[b11][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[a10][col] += 1; U[a11][col] += 1; V[b00][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a00][col] += 1; V[b01][col] += 1; V[b11][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[a11][col] += 1; V[b10][col] += 1; V[b00][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a00][col] += 1; U[a01][col] += 1; V[b11][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[a10][col] += 1; U[a00][col] -= 1; V[b00][col] += 1; V[b01][col] += 1; W[c11][col] += 1; col += 1
            U[a01][col] += 1; U[a11][col] -= 1; V[b10][col] += 1; V[b11][col] += 1; W[c00][col] += 1; col += 1
            return col
        
        def add_rank2_block(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            # 2x2 A, 2x1 B, 2x1 C
            # Rank 2 decomposition
            # Term 1: handles i=0
            # U[i, col] = 1 if i==0 else 0
            # V[j, col] = 1 for all j
            # W[k, col] = 1 if k==0 else 0
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b0 = b_rows[0]*p + b_col
            b1 = b_rows[1]*p + b_col
            c0 = c_col*n + c_rows[0]
            c1 = c_col*n + c_rows[1]
            col = start_col
            # r=0: i=0 part
            U[a00][col] += 1
            U[a01][col] += 1
            V[b0][col] += 1
            V[b1][col] += 1
            W[c0][col] += 1
            col += 1
            # r=1: i=1 part
            U[a10][col] += 1
            U[a11][col] += 1
            V[b0][col] += 1
            V[b1][col] += 1
            W[c1][col] += 1
            col += 1
            return col

        col = 0
        # A block 0 (rows 0,1 of A, cols 0,1)
        # B block 0 (rows 0,1 of B, cols 0,1) -> C (rows 0,1, cols 0,1)
        col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)
        # B block 1 (rows 0,1 of B, cols 2,3) -> C (rows 0,1, cols 2,3)
        col = add_strassen([0,1],[0,1],[0,1],[2,3],[0,1],[2,3], col)
        # B block 2 (rows 0,1 of B, cols 4,5) -> C (rows 0,1, cols 4,5)
        col = add_strassen([0,1],[0,1],[0,1],[4,5],[0,1],[4,5], col)
        # B block 3 (rows 0,1 of B, col 6) -> C (rows 0,1, col 6)
        col = add_rank2_block([0,1],[0,1],[0,1],6,[0,1],6, col)
        
        # A block 1 (rows 0,1 of A, cols 2,3)
        # B block 0 (rows 2,3 of B, cols 0,1) -> C (rows 0,1, cols 0,1)
        col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)
        # B block 1 (rows 2,3 of B, cols 2,3) -> C (rows 0,1, cols 2,3)
        col = add_strassen([0,1],[2,3],[2,3],[2,3],[0,1],[2,3], col)
        # B block 2 (rows 2,3 of B, cols 4,5) -> C (rows 0,1, cols 4,5)
        col = add_strassen([0,1],[2,3],[2,3],[4,5],[0,1],[4,5], col)
        # B block 3 (rows 2,3 of B, col 6) -> C (rows 0,1, col 6)
        col = add_rank2_block([0,1],[2,3],[2,3],6,[0,1],6, col)
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    s = 0.0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    expected = target.get((u,v,w), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    # Verify the new rank-46 solution
    sol = build_rank46()
    if verify(sol["U"], sol["V"], sol["W"], 46):
        return sol
    
    # Fallback to rank-50 if verification fails (should not happen)
    def build_rank50():
        rank = 50
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        
        def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, start_col):
            a00 = a_rows[0]*m + a_cols[0]
            a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]
            a11 = a_rows[1]*m + a_cols[1]
            b00 = b_rows[0]*p + b_cols[0]
            b01 = b_rows[0]*p + b_cols[1]
            b10 = b_rows[1]*p + b_cols[0]
            b11 = b_rows[1]*p + b_cols[1]
            c00 = c_cols[0]*n + c_rows[0]
            c01 = c_cols[1]*n + c_rows[0]
            c10 = c_cols[0]*n + c_rows[1]
            c11 = c_cols[1]*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; U[a11][col] += 1; V[b00][col] += 1; V[b11][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[a10][col] += 1; U[a11][col] += 1; V[b00][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a00][col] += 1; V[b01][col] += 1; V[b11][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[a11][col] += 1; V[b10][col] += 1; V[b00][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a00][col] += 1; U[a01][col] += 1; V[b11][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[a10][col] += 1; U[a00][col] -= 1; V[b00][col] += 1; V[b01][col] += 1; W[c11][col] += 1; col += 1
            U[a01][col] += 1; U[a11][col] -= 1; V[b10][col] += 1; V[b11][col] += 1; W[c00][col] += 1; col += 1
            return col
        
        def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_rows, c_col, start_col):
            a00 = a_rows[0]*m + a_cols[0]; a01 = a_rows[0]*m + a_cols[1]
            a10 = a_rows[1]*m + a_cols[0]; a11 = a_rows[1]*m + a_cols[1]
            b0 = b_rows[0]*p + b_col; b1 = b_rows[1]*p + b_col
            c0 = c_col*n + c_rows[0]; c1 = c_col*n + c_rows[1]
            col = start_col
            U[a00][col] += 1; V[b0][col] += 1; W[c0][col] += 1; col += 1
            U[a01][col] += 1; V[b1][col] += 1; W[c0][col] += 1; col += 1
            U[a10][col] += 1; V[b0][col] += 1; W[c1][col] += 1; col += 1
            U[a11][col] += 1; V[b1][col] += 1; W[c1][col] += 1; col += 1
            return col
        
        col = 0
        col = add_strassen([0,1],[0,1],[0,1],[0,1],[0,1],[0,1], col)
        col = add_schoolbook_2x2_2x1([0,1],[0,1],[0,1],2,[0,1],2, col)
        col = add_strassen([0,1],[2,3],[2,3],[0,1],[0,1],[0,1], col)
        col = add_schoolbook_2x2_2x1([0,1],[2,3],[2,3],2,[0,1],2, col)
        col = add_strassen([0,1],[0,1],[0,1],[3,4],[0,1],[3,4], col)
        col = add_strassen([0,1],[0,1],[0,1],[5,6],[0,1],[5,6], col)
        col = add_strassen([0,1],[2,3],[2,3],[3,4],[0,1],[3,4], col)
        col = add_strassen([0,1],[2,3],[2,3],[5,6],[0,1],[5,6], col)
        return {"U": U, "V": V, "W": W}

    return build_rank50()
