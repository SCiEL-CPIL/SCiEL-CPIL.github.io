import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Try rank 21 first with a verified construction
    rank = 21
    result = _try_rank_21(n, m, p)
    if result is not None:
        return result
    
    # Fallback to rank 22
    rank = 22
    result = _try_rank_22(n, m, p)
    if result is not None:
        return result
    
    # Final fallback: rank 24 (trivial)
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for r in range(n):
        for c in range(m):
            for d in range(p):
                U[r * m + c, idx] = 1
                V[c * p + d, idx] = 1
                W[r * n + d, idx] = 1
                idx += 1
    
    return _to_int_lists(U, V, W, rank)


def _try_rank_21(n, m, p):
    """Try to construct a rank-21 decomposition."""
    U = np.zeros((n * m, 21))
    V = np.zeros((m * p, 21))
    W = np.zeros((p * n, 21))
    
    def u_idx(r, c): return r * m + c
    def v_idx(c, d): return c * p + d
    def w_idx(r, d): return r * n + d
    
    def add(u_ent, v_ent, w_ent, col):
        for i, v in u_ent: U[i, col] += v
        for i, v in v_ent: V[i, col] += v
        for i, v in w_ent: W[i, col] += v
    
    # Based on known constructions, here's a rank-21 attempt:
    # Use the structure that for each of the 3 intermediate columns,
    # we can use 7 terms, but share some across columns.
    
    # I'll use a construction that exploits the 2-row structure.
    
    # Terms 0-6: Handle output cols 0-1 for all rows and intermediate cols
    add([(u_idx(0,0),1),(u_idx(1,0),1)], [(v_idx(0,0),1),(v_idx(0,1),1)], [(w_idx(0,0),1),(w_idx(0,1),1)], 0)
    add([(u_idx(0,0),1),(u_idx(1,0),-1)], [(v_idx(1,0),1),(v_idx(1,1),-1)], [(w_idx(0,1),1),(w_idx(0,0),-1)], 1)
    add([(u_idx(0,1),1)], [(v_idx(1,1),1),(v_idx(0,1),-1)], [(w_idx(0,2),1),(w_idx(0,1),1)], 2)
    add([(u_idx(1,1),1)], [(v_idx(1,2),1),(v_idx(0,2),-1)], [(w_idx(0,3),1),(w_idx(0,2),1)], 3)
    add([(u_idx(0,2),1),(u_idx(1,2),1)], [(v_idx(2,0),1),(v_idx(2,1),1)], [(w_idx(1,0),1),(w_idx(1,1),1)], 4)
    add([(u_idx(0,2),1),(u_idx(1,2),-1)], [(v_idx(2,2),1),(v_idx(2,3),-1)], [(w_idx(1,2),1),(w_idx(1,3),-1)], 5)
    add([(u_idx(0,0),1),(u_idx(0,1),1)], [(v_idx(0,2),1)], [(w_idx(0,2),-1),(w_idx(0,3),1)], 6)
    
    # Terms 7-13: Handle output cols 2-3
    add([(u_idx(1,0),1),(u_idx(1,1),1)], [(v_idx(1,2),1)], [(w_idx(1,2),-1),(w_idx(1,3),1)], 7)
    add([(u_idx(0,1),1),(u_idx(1,1),1)], [(v_idx(0,3),1)], [(w_idx(0,3),1)], 8)
    add([(u_idx(0,1),1)], [(v_idx(1,3),1)], [(w_idx(0,0),1)], 9)
    add([(u_idx(1,1),1)], [(v_idx(1,3),1)], [(w_idx(1,0),1)], 10)
    add([(u_idx(0,2),1)], [(v_idx(2,3),1)], [(w_idx(0,3),1)], 11)
    add([(u_idx(1,2),1)], [(v_idx(2,3),1)], [(w_idx(1,3),1)], 12)
    add([(u_idx(0,0),1)], [(v_idx(0,0),1)], [(w_idx(0,0),1)], 13)
    
    # Terms 14-20: Fill in remaining
    add([(u_idx(1,0),1)], [(v_idx(1,0),1)], [(w_idx(1,0),1)], 14)
    add([(u_idx(0,1),1)], [(v_idx(1,1),1)], [(w_idx(0,1),1)], 15)
    add([(u_idx(1,1),1)], [(v_idx(1,1),1)], [(w_idx(1,1),1)], 16)
    add([(u_idx(0,2),1)], [(v_idx(2,0),1)], [(w_idx(0,0),1)], 17)
    add([(u_idx(1,2),1)], [(v_idx(2,1),1)], [(w_idx(1,1),1)], 18)
    add([(u_idx(0,2),1)], [(v_idx(2,2),1)], [(w_idx(0,2),1)], 19)
    add([(u_idx(1,2),1)], [(v_idx(2,2),1)], [(w_idx(1,2),1)], 20)
    
    if _verify(U, V, W, n, m, p):
        return _to_int_lists(U, V, W, 21)
    return None


def _try_rank_22(n, m, p):
    """Construct the known rank-22 decomposition."""
    U = np.zeros((n * m, 22))
    V = np.zeros((m * p, 22))
    W = np.zeros((p * n, 22))
    
    def add(u_ent, v_ent, w_ent, col):
        for i, v in u_ent: U[i, col] += v
        for i, v in v_ent: V[i, col] += v
        for i, v in w_ent: W[i, col] += v
    
    # Block 1: Strassen for A[0:2, 0:2] * B[0:3, 0:2]
    add([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
    add([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
    add([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
    add([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
    add([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
    add([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
    add([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
    
    # Block 2: Strassen for A[0:2, 0:2] * B[0:3, 2:4]
    add([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
    add([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
    add([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
    add([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
    add([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
    add([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
    add([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
    
    # Cross terms
    add([(2, 1)], [(8, 1)], [(0, 1)], 14)
    add([(2, 1)], [(9, 1)], [(2, 1)], 15)
    add([(2, 1)], [(10, 1)], [(4, 1)], 16)
    add([(2, 1)], [(11, 1)], [(6, 1)], 17)
    add([(5, 1)], [(8, 1)], [(1, 1)], 18)
    add([(5, 1)], [(9, 1)], [(3, 1)], 19)
    add([(5, 1)], [(10, 1)], [(5, 1)], 20)
    add([(5, 1)], [(11, 1)], [(7, 1)], 21)
    
    if _verify(U, V, W, n, m, p):
        return _to_int_lists(U, V, W, 22)
    return None


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                r_i, c_i = divmod(i, m)
                c_j, d_j = divmod(j, p)
                r_k, d_k = divmod(k, n)
                expected = 1.0 if (c_i == c_j and r_i == r_k and d_j == d_k) else 0.0
                val = sum(U[i, r] * V[j, r] * W[k, r] for r in range(rank))
                if abs(val - expected) > 1e-9:
                    return False
    return True


def _to_int_lists(U, V, W, rank):
    """Convert numpy arrays to lists with integer/half-integer values."""
    U_list = [[0] * rank for _ in range(U.shape[0])]
    V_list = [[0] * rank for _ in range(V.shape[0])]
    W_list = [[0] * rank for _ in range(W.shape[0])]
    
    for i in range(U.shape[0]):
        for r in range(rank):
            val = U[i, r]
            val_rounded = round(val * 2) / 2
            U_list[i][r] = val_rounded
    
    for i in range(V.shape[0]):
        for r in range(rank):
            val = V[i, r]
            val_rounded = round(val * 2) / 2
            V_list[i][r] = val_rounded
    
    for i in range(W.shape[0]):
        for r in range(rank):
            val = W[i, r]
            val_rounded = round(val * 2) / 2
            W_list[i][r] = val_rounded
    
    # Verify half-integer constraint
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for val in row:
                if not (val == int(val) or abs(val * 2 - int(val * 2)) < 1e-9):
                    raise ValueError("Non-half-integer entry")
    
    return {"U": U_list, "V": V_list, "W": W_list}
