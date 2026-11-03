import time
import numpy as np
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def compute_contrib(U, V, W, rank):
        """Compute contribution of each term: contrib[r,i,j,k] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        contrib = np.zeros((rank, n, m, p))
        for r in range(rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            for i in range(n):
                for j in range(m):
                    uv = u_col[i*m + j]
                    if uv == 0:
                        continue
                    for k in range(p):
                        contrib[r, i, j, k] = uv * v_col[j*p + k] * w_col[k*n + i]
        return contrib

    def verify_contrib(contrib, tol=1e-9):
        return np.allclose(contrib.sum(axis=0), 1.0, atol=tol)

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]
        
        num_blocks = len(j_pairs) * len(k_pairs)
        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1
        
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1
        
        return U, V, W, rank

    def try_eliminate(U, V, W, rank):
        """Eliminate redundant terms using precomputed contributions."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        contrib = compute_contrib(U_arr, V_arr, W_arr, rank)
        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        cur_rank = rank

        for _ in range(30):
            if time_left() < 5:
                break
            progress = False
            for t in range(cur_rank - 1, -1, -1):
                if time_left() < 5:
                    break
                T = contrib[:cur_rank].sum(axis=0)
                T_no = T - contrib[t]
                if np.allclose(T_no, 1.0, atol=1e-9):
                    keep = [j for j in range(cur_rank) if j != t]
                    U_list = [[row[j] for j in keep] for row in U_list]
                    V_list = [[row[j] for j in keep] for row in V_list]
                    W_list = [[row[j] for j in keep] for row in W_list]
                    cur_rank -= 1
                    contrib = np.delete(contrib, t, axis=0)
                    progress = True
            if not progress:
                break
        return U_list, V_list, W_list, cur_rank

    def try_pairwise(U_list, V_list, W_list, rank):
        """Eliminate pairs of redundant terms."""
        if time_left() < 10:
            return U_list, V_list, W_list, rank
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        contrib = compute_contrib(U_arr, V_arr, W_arr, rank)
        cur_rank = rank

        for _ in range(10):
            if time_left() < 10:
                break
            progress = False
            for i1 in range(cur_rank):
                if time_left() < 10:
                    break
                for i2 in range(i1 + 1, cur_rank):
                    if time_left() < 10:
                        break
                    T = contrib[:cur_rank].sum(axis=0)
                    T_no = T - contrib[i1] - contrib[i2]
                    if np.allclose(T_no, 1.0, atol=1e-9):
                        keep = [j for j in range(cur_rank) if j != i1 and j != i2]
                        U_list = [[row[j] for j in keep] for row in U_list]
                        V_list = [[row[j] for j in keep] for row in V_list]
                        W_list = [[row[j] for j in keep] for row in W_list]
                        cur_rank -= 2
                        U_arr = np.array(U_list, dtype=float)
                        V_arr = np.array(V_list, dtype=float)
                        W_arr = np.array(W_list, dtype=float)
                        contrib = compute_contrib(U_arr, V_arr, W_arr, cur_rank)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break
        return U_list, V_list, W_list, cur_rank

    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,2),(1,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(2,5),(3,4)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(3,4),(2,5)]),
        ([(0,1),(2,3)], [(1,2),(3,4),(0,5)]),
        ([(0,1),(2,3)], [(0,3),(1,2),(4,5)]),
        ([(0,1),(2,3)], [(0,4),(2,3),(1,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,2),(1,5),(3,4)]),
        ([(0,1),(2,3)], [(0,5),(1,4),(2,3)]),
        ([(0,1),(2,3)], [(1,4),(2,5),(0,3)]),
        ([(0,1),(2,3)], [(0,1),(3,5),(2,4)]),
    ]

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        contrib = compute_contrib(U_arr, V_arr, W_arr, rank)
        if not verify_contrib(contrib):
            continue

        U_list, V_list, W_list, new_rank = try_eliminate(U, V, W, rank)

        if time_left() > 15:
            U_list, V_list, W_list, new_rank = try_pairwise(U_list, V_list, W_list, new_rank)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
