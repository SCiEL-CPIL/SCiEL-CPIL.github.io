import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 7
    Na, Nb, Nc = n*m, m*p, p*n  # 8, 28, 14
    
    # Build target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[(u_idx, v_idx, w_idx)] = 1
    
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
    
    def verify(U, V, W, rank):
        for u in range(Na):
            for v in range(Nb):
                for w in range(Nc):
                    s = 0
                    for r in range(rank):
                        s += U[u][r] * V[v][r] * W[w][r]
                    expected = target.get((u,v,w), 0)
                    if s != expected:
                        return False
        return True
    
    def build_eval_matrix(U, V, W, rank):
        M = np.zeros((Na * Nb * Nc, rank), dtype=float)
        for r in range(rank):
            vec = np.outer(np.outer(U[:, r], V[:, r]).flatten(), W[:, r]).flatten()
            M[:, r] = vec
        return M
    
    def try_rank_reduction():
        result50 = build_rank50()
        U = np.array(result50["U"], dtype=float)
        V = np.array(result50["V"], dtype=float)
        W = np.array(result50["W"], dtype=float)
        
        M = build_eval_matrix(U, V, W, 50)
        
        try:
            _, s, vh = np.linalg.svd(M, full_matrices=False)
            rank_M = int(np.sum(s > 1e-10))
        except:
            return None
        
        if rank_M >= 50:
            return None
        
        # Try dropping single terms
        for r_to_drop in range(50):
            if time.time() - start_time > 130:
                break
            keep = [i for i in range(50) if i != r_to_drop]
            new_U = U[:, keep]
            new_V = V[:, keep]
            new_W = W[:, keep]
            if verify(new_U.tolist(), new_V.tolist(), new_W.tolist(), 49):
                return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
        
        # Try dropping pairs of terms
        for r1 in range(50):
            if time.time() - start_time > 140:
                break
            for r2 in range(r1+1, 50):
                keep = [i for i in range(50) if i != r1 and i != r2]
                new_U = U[:, keep]
                new_V = V[:, keep]
                new_W = W[:, keep]
                if verify(new_U.tolist(), new_V.tolist(), new_W.tolist(), 48):
                    return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
        
        return None
    
    result = build_rank50()
    
    if verify(result["U"], result["V"], result["W"], 50):
        reduced = try_rank_reduction()
        if reduced is not None:
            return reduced
        return result
    else:
        return result
