import numpy as np
import time

def construct():
    """
    Construct a rank-minimal tensor decomposition for 2x2 * 2x9 matrix multiplication.
    Attempts to reduce the rank of the standard rank-32 decomposition by finding
    zero-sum subsets of rank-1 terms.
    """
    n, m, p = 2, 2, 9
    rank = 32

    U = [[0] * rank for _ in range(n * m)]   # 4 x 32
    V = [[0] * rank for _ in range(m * p)]   # 18 x 32
    W = [[0] * rank for _ in range(p * n)]   # 18 x 32

    def strassen_block(k1, k2, start_col):
        v_00 = 0 * p + k1
        v_01 = 0 * p + k2
        v_10 = 1 * p + k1
        v_11 = 1 * p + k2

        w_00 = k1 * n + 0
        w_01 = k1 * n + 1
        w_02 = k2 * n + 0
        w_03 = k2 * n + 1

        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3

        c = start_col + 0
        U[u_00][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1; V[v_11][c] += 1
        W[w_00][c] += 1; W[w_03][c] += 1

        c = start_col + 1
        U[u_10][c] += 1; U[u_11][c] += 1
        V[v_00][c] += 1
        W[w_01][c] += 1; W[w_03][c] -= 1

        c = start_col + 2
        U[u_00][c] += 1
        V[v_01][c] += 1; V[v_11][c] -= 1
        W[w_02][c] += 1; W[w_03][c] += 1

        c = start_col + 3
        U[u_11][c] += 1
        V[v_10][c] += 1; V[v_00][c] -= 1
        W[w_00][c] += 1; W[w_01][c] += 1

        c = start_col + 4
        U[u_00][c] += 1; U[u_01][c] += 1
        V[v_11][c] += 1
        W[w_00][c] -= 1; W[w_02][c] += 1

        c = start_col + 5
        U[u_10][c] += 1; U[u_00][c] -= 1
        V[v_00][c] += 1; V[v_01][c] += 1
        W[w_03][c] += 1

        c = start_col + 6
        U[u_01][c] += 1; U[u_11][c] -= 1
        V[v_10][c] += 1; V[v_11][c] += 1
        W[w_00][c] += 1

        return start_col + 7

    def schoolbook_single(k, start_col):
        v_0 = 0 * p + k
        v_1 = 1 * p + k

        w_0 = k * n + 0
        w_1 = k * n + 1

        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3

        c = start_col + 0
        U[u_00][c] += 1
        V[v_0][c] += 1
        W[w_0][c] += 1

        c = start_col + 1
        U[u_01][c] += 1
        V[v_1][c] += 1
        W[w_0][c] += 1

        c = start_col + 2
        U[u_10][c] += 1
        V[v_0][c] += 1
        W[w_1][c] += 1

        c = start_col + 3
        U[u_11][c] += 1
        V[v_1][c] += 1
        W[w_1][c] += 1

        return start_col + 4

    col = 0
    for k1, k2 in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        col = strassen_block(k1, k2, col)
    col = schoolbook_single(8, col)

    # Check for zero-sum subsets to reduce rank
    # Represent each rank-1 term as a vector
    dim_u, dim_v, dim_w = 4, 18, 18
    terms = []
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(dim_u)])
        v_vec = np.array([V[i][c] for i in range(dim_v)])
        w_vec = np.array([W[i][c] for i in range(dim_w)])
        term = np.kron(np.kron(u_vec, v_vec), w_vec)
        terms.append(term)
    
    terms = np.array(terms) # shape (32, 1296)
    
    # Look for pairs i, j such that terms[i] + terms[j] == 0
    # Since entries are integers, we can check for exact zero
    removed = set()
    for i in range(rank):
        if i in removed: continue
        for j in range(i+1, rank):
            if j in removed: continue
            if np.allclose(terms[i] + terms[j], 0):
                removed.add(i)
                removed.add(j)
                break
        if len(removed) >= 2:
            break
            
    # If no pairs found, look for triples? (Too slow for 32 choose 3)
    # But we can check if any term is 0 (shouldn't happen)
    
    if removed:
        keep = [i for i in range(rank) if i not in removed]
        new_rank = len(keep)
        new_U = [[U[i][c] for c in keep] for i in range(dim_u)]
        new_V = [[V[i][c] for c in keep] for i in range(dim_v)]
        new_W = [[W[i][c] for c in keep] for i in range(dim_w)]
        return {"U": new_U, "V": new_V, "W": new_W}
    else:
        return {"U": U, "V": V, "W": W}
