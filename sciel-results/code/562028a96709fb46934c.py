import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 5

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, 1.0, atol=1e-8)

    def get_schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build rank-65 decomposition
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def strassen_2x2():
        return [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]

    inner_contrib = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,-1)],
        [(0,1,1),(1,1,1)],
        [(0,0,1),(1,0,1)],
        [(0,0,-1),(0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]

    outer_block_map = [
        [((0,0),1),((2,2),1)],
        [((2,0),1),((2,2),-1)],
        [((0,2),1),((2,2),1)],
        [((0,0),1),((2,0),1)],
        [((0,0),-1),((0,2),1)],
        [((2,2),1)],
        [((0,0),1)],
    ]

    outer_terms = [
        ([{(r,c):1,(r+2,c+2):1} for r in range(2) for c in range(2)],
         [{(r,c):1,(r+2,c+2):1} for r in range(2) for c in range(2)]),
        ([{(r+2,c):1,(r+2,c+2):1} for r in range(2) for c in range(2)],
         [{(r,c):1} for r in range(2) for c in range(2)]),
        ([{(r,c):1} for r in range(2) for c in range(2)],
         [{(r,c+2):1,(r+2,c+2):-1} for r in range(2) for c in range(2)]),
        ([{(r+2,c+2):1} for r in range(2) for c in range(2)],
         [{(r+2,c):1,(r,c):-1} for r in range(2) for c in range(2)]),
        ([{(r,c):1,(r,c+2):1} for r in range(2) for c in range(2)],
         [{(r+2,c+2):1} for r in range(2) for c in range(2)]),
        ([{(r+2,c):1,(r,c):-1} for r in range(2) for c in range(2)],
         [{(r,c):1,(r,c+2):1} for r in range(2) for c in range(2)]),
        ([{(r,c+2):1,(r+2,c+2):-1} for r in range(2) for c in range(2)],
         [{(r+2,c):1,(r+2,c+2):1} for r in range(2) for c in range(2)]),
    ]

    col = 0
    smults = strassen_2x2()
    for j in range(7):
        U_blks, V_blks = outer_terms[j]
        U_mat = [[U_blks[r*2+c] for c in range(2)] for r in range(2)]
        V_mat = [[V_blks[r*2+c] for c in range(2)] for r in range(2)]
        for t in range(7):
            u_d, v_d = smults[t]
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), cu in u_d.items():
                for (ai, aj), cai in U_mat[ur][uc].items():
                    u_scalar[(ai, aj)] = u_scalar.get((ai, aj), 0) + cu * cai
            for (vr, vc), cv in v_d.items():
                for (bi, bj), cbj in V_mat[vr][vc].items():
                    v_scalar[(bi, bj)] = v_scalar.get((bi, bj), 0) + cv * cbj
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}

            w_dict = {}
            for (ro, co), bcoeff in outer_block_map[j]:
                for mr, mc, icoeff in inner_contrib[t]:
                    cr, cc = ro + mr, co + mc
                    w_idx = cc * n + cr
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + bcoeff * icoeff

            for (ai, aj), c in u_scalar.items():
                U[ai * m + aj, col] += c
            for (bi, bj), c in v_scalar.items():
                V[bi * p + bj, col] += c
            for w_idx, c in w_dict.items():
                W[w_idx, col] += c
            col += 1

    # 16 schoolbook terms for column k=4
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1

    # Try to reduce rank by checking if any term is redundant
    # Check if we can remove one of the last 16 terms
    # by adjusting another term
    
    # Try rank 64: remove one scalar term and see if we can compensate
    for remove_col in range(49, 65):
        if time.time() > deadline - 10:
            break
        U_test = U.copy()
        V_test = V.copy()
        W_test = W.copy()
        U_test[:, remove_col] = 0
        V_test[:, remove_col] = 0
        W_test[:, remove_col] = 0
        
        # Try to find a correction in another column
        # This is complex, so let's just verify if the current one works
        if verify(U_test, V_test, W_test):
            # Remove the column
            U_final = np.delete(U_test, remove_col, axis=1)
            V_final = np.delete(V_test, remove_col, axis=1)
            W_final = np.delete(W_test, remove_col, axis=1)
            if verify(U_final, V_final, W_final):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

    if time.time() > deadline - 5:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
