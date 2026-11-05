"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    deadline = time.time() + 130

    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_factors = (U_school, V_school, W_school)
    best_rank = rank_school

    def strassen_2x2x2():
        """7 Strassen terms for 2x2x2. Each term: (U_dict, V_dict, W_dict)
        U_dict: {(i,j): val}, V_dict: {(j,k): val}, W_dict: {(k,i): val}
        """
        terms = []
        # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        terms.append({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
        # M2 = (a10+a11)b00 -> c10 += 1
        terms.append({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1})
        # M3 = a00(b01-b11) -> c01 += 1
        terms.append({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1})
        # M4 = a11(b10-b00) -> c10 += 1, c11 -= 1
        terms.append({(1,1):1}, {(1,0):1,(0,0):-1}, {(1,0):1,(1,1):-1})
        # M5 = (a00+a01)b11 -> c00 -= 1, c11 += 1
        terms.append({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(1,1):1})
        # M6 = (a10-a00)(b00+b01) -> c11 += 1
        terms.append({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1})
        # M7 = (a01-a11)(b10+b11) -> c00 += 1
        terms.append({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1})
        return terms

    def _estimate_cost(bn, bm, bp):
        if bn == 1 and bm == 1 and bp == 1:
            return 1
        if bn == 2 and bm == 2 and bp == 2:
            return 7
        cost = bn * bm * bp
        for dim in range(3):
            size = [bn, bm, bp][dim]
            if size < 2:
                continue
            for split in range(1, size):
                if dim == 0:
                    c = _estimate_cost(split, bm, bp) + _estimate_cost(bn-split, bm, bp)
                elif dim == 1:
                    c = _estimate_cost(bn, split, bp) + _estimate_cost(bn, bm-split, bp)
                else:
                    c = _estimate_cost(bn, bm, split) + _estimate_cost(bn, bm, bp-split)
                if c < cost:
                    cost = c
        return cost

    def build_recursive(U, V, W, i0, j0, k0, bn, bm, bp, col_start, m, p, n):
        col = col_start
        if bn == 1 and bm == 1 and bp == 1:
            U[i0*m + j0, col] = 1
            V[j0*p + k0, col] = 1
            W[k0*n + i0, col] = 1
            return col + 1
        if bn == 2 and bm == 2 and bp == 2:
            terms = strassen_2x2x2()
            for term_u, term_v, term_w in terms:
                for (li, lj), val in term_u.items():
                    U[(i0+li)*m + (j0+lj), col] = val
                for (lj, lk), val in term_v.items():
                    V[(j0+lj)*p + (k0+lk), col] = val
                for (lk, li), val in term_w.items():
                    W[(k0+lk)*n + (i0+li), col] = val
                col += 1
            return col
        
        # Find best split
        best_split = None
        best_cost = bn * bm * bp
        for dim in range(3):
            size = [bn, bm, bp][dim]
            if size < 2:
                continue
            for split in range(1, size):
                if dim == 0:
                    c = _estimate_cost(split, bm, bp) + _estimate_cost(bn-split, bm, bp)
                elif dim == 1:
                    c = _estimate_cost(bn, split, bp) + _estimate_cost(bn, bm-split, bp)
                else:
                    c = _estimate_cost(bn, bm, split) + _estimate_cost(bn, bm, bp-split)
                if c < best_cost:
                    best_cost = c
                    best_split = (dim, split)
        
        if best_split is None:
            for li in range(bn):
                for lj in range(bm):
                    for lk in range(bp):
                        U[(i0+li)*m + (j0+lj), col] = 1
                        V[(j0+lj)*p + (k0+lk), col] = 1
                        W[(k0+lk)*n + (i0+li), col] = 1
                        col += 1
            return col
        else:
            dim, split = best_split
            if dim == 0:
                col = build_recursive(U, V, W, i0, j0, k0, split, bm, bp, col, m, p, n)
                col = build_recursive(U, V, W, i0+split, j0, k0, bn-split, bm, bp, col, m, p, n)
            elif dim == 1:
                col = build_recursive(U, V, W, i0, j0, k0, bn, split, bp, col, m, p, n)
                col = build_recursive(U, V, W, i0, j0+split, k0, bn, bm-split, bp, col, m, p, n)
            else:
                col = build_recursive(U, V, W, i0, j0, k0, bn, bm, split, col, m, p, n)
                col = build_recursive(U, V, W, i0, j0, k0+split, bn, bm, bp-split, col, m, p, n)
            return col

    est_rank = _estimate_cost(n, m, p)
    U = np.zeros((n*m, est_rank))
    V = np.zeros((m*p, est_rank))
    W = np.zeros((p*n, est_rank))
    final_col = build_recursive(U, V, W, 0, 0, 0, n, m, p, 0, m, p, n)
    U = U[:, :final_col]
    V = V[:, :final_col]
    W = W[:, :final_col]

    if verify(U, V, W, n, m, p):
        best_factors = (U, V, W)
        best_rank = final_col
    else:
        best_factors = (U_school, V_school, W_school)
        best_rank = rank_school

    U, V, W = best_factors
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
