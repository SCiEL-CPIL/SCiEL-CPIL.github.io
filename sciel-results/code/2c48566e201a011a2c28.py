"""Combined program: block Strassen + support search + optimization + polish."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

# EVOLVE-BLOCK-START

def _verify_einsum(U, V, W, n=N, m=M, p=P, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def _verify_strict(U, V, W, n=N, m=M, p=P, tol=1e-12):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def schoolbook(na, mb, pb):
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _add_sub(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start, n, m, p):
    r = len(U_sub[0])
    for c in range(r):
        col = col_start + c
        for i_prime in range(na):
            for j_prime in range(mb):
                u_val = U_sub[i_prime*mb + j_prime][c]
                if u_val != 0:
                    U[(i_off + i_prime)*m + (j_off + j_prime)][col] += u_val
        for j_prime in range(mb):
            for k_prime in range(pb):
                v_val = V_sub[j_prime*pb + k_prime][c]
                if v_val != 0:
                    V[(j_off + j_prime)*p + (k_off + k_prime)][col] += v_val
        for k_prime in range(pb):
            for i_prime in range(na):
                w_val = W_sub[k_prime*na + i_prime][c]
                if w_val != 0:
                    W[(k_off + k_prime)*n + (i_off + i_prime)][col] += w_val


def _build_2plus1_split():
    """Split rows as 2+1, columns as 2+2."""
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 2, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
    return U, V, W


def _build_1plus2_split():
    """Split rows as 1+2, columns as 2+2."""
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            # 1x2x2 block (schoolbook)
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
            # 2x2x2 block (Strassen)
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 1, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
    return U, V, W


def _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=15):
    """Try to reduce rank by removing columns that don't affect the result."""
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    if r > 48:
        return None
    best_rank = r
    best_U, best_V, best_W = U, V, W
    
    # Try removing each column one at a time
    for col_idx in range(r - 1, -1, -1):
        if time.time() - start > max_time:
            break
        U_test = np.delete(U, col_idx, axis=1)
        V_test = np.delete(V, col_idx, axis=1)
        W_test = np.delete(W, col_idx, axis=1)
        if _verify_einsum(U_test, V_test, W_test, n, m, p, tol=1e-10):
            U, V, W = U_test, V_test, W_test
            best_rank = W.shape[1]
            best_U, best_V, best_W = U, V, W
    
    if best_rank < r:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None


def _optimize_rank(U_init, V_init, W_init, n, m, p, r, max_iter=300, ftol=1e-12, gtol=1e-9):
    """Optimize a given rank-r decomposition starting from initial guess."""
    from scipy.optimize import minimize
    
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    U_flat = np.array(U_init, dtype=np.float64).flatten()
    V_flat = np.array(V_init, dtype=np.float64).flatten()
    W_flat = np.array(W_init, dtype=np.float64).flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def loss_func(x):
        U_f = x[:nm*r]
        V_f = x[nm*r:nm*r+mp*r]
        W_f = x[nm*r+mp*r:]
        
        U = U_f.reshape(nm, r)
        V = V_f.reshape(mp, r)
        W = W_f.reshape(pn, r)
        
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        loss = np.sum((recon - target) ** 2)
        return loss
    
    try:
        res = minimize(loss_func, x0, method='L-BFGS-B', 
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        return res
    except Exception:
        return None


def _try_optimization_for_rank(r, n, m, p, deadline, n_restarts=10):
    """Try to find a valid rank-r decomposition via optimization."""
    nm, mp, pn = n*m, m*p, p*n
    target = np.ones((n, m, p))
    
    best_loss = np.inf
    best_x = None
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initialization with various scales
        scale = np.random.choice([0.01, 0.1, 0.5, 1.0])
        U_init = np.random.randn(nm, r) * scale
        V_init = np.random.randn(mp, r) * scale
        W_init = np.random.randn(pn, r) * scale
        
        res = _optimize_rank(U_init, V_init, W_init, n, m, p, r, max_iter=200)
        
        if res is not None and res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x
            
            if best_loss < 1e-10:
                U = best_x[:nm*r].reshape(nm, r)
                V = best_x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = best_x[nm*r+mp*r:].reshape(pn, r)
                if _verify_strict(U, V, W, n, m, p, tol=1e-9):
                    return U.tolist(), V.tolist(), W.tolist(), r
    
    return None


def _polish_solution(U, V, W, n, m, p, deadline, max_iter=500):
    """High-precision polish pass on a near-solution."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Try several polish iterations with decreasing tolerance
    for tol in [1e-8, 1e-10, 1e-12]:
        if time.time() > deadline - 5:
            break
        
        res = _optimize_rank(U_arr, V_arr, W_arr, n, m, p, r, 
                           max_iter=max_iter, ftol=tol*0.1, gtol=tol*0.01)
        
        if res is not None and res.fun < tol:
            U_arr = res.x[:nm*r].reshape(nm, r)
            V_arr = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
            W_arr = res.x[nm*r+mp*r:].reshape(pn, r)
            
            if _verify_strict(U_arr, V_arr, W_arr, n, m, p, tol=tol):
                return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()
    
    return None


def _try_support_based_search(n, m, p, deadline, max_rank=35):
    """Try to find a low-rank decomposition using support-based exact cover."""
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    nm, mp, pn = n*m, m*p, p*n
    
    # Generate candidate columns with small supports
    candidates = []
    
    # Singleton supports (schoolbook)
    for (i, j, k) in triples:
        s_u = {i*m+j}
        s_v = {j*p+k}
        s_w = {k*n+i}
        covered = {(i, j, k)}
        candidates.append((frozenset(s_u), frozenset(s_v), frozenset(s_w), frozenset(covered)))
    
    # Pair supports
    for idx in range(len(triples)):
        for idx2 in range(idx+1, len(triples)):
            i1, j1, k1 = triples[idx]
            i2, j2, k2 = triples[idx2]
            s_u = {i1*m+j1, i2*m+j2}
            s_v = {j1*p+k1, j2*p+k2}
            s_w = {k1*n+i1, k2*n+i2}
            
            covered = set()
            for (i3, j3, k3) in triples:
                if i3*m+j3 in s_u and j3*p+k3 in s_v and k3*n+i3 in s_w:
                    covered.add((i3, j3, k3))
            
            if len(covered) == 2 and {(i1,j1,k1), (i2,j2,k2)}.issubset(covered):
                candidates.append((frozenset(s_u), frozenset(s_v), frozenset(s_w), frozenset(covered)))
            
            if time.time() > deadline - 5:
                break
    
    if not candidates:
        return None
    
    # Try to find an exact cover
    triples_set = frozenset(triples)
    candidates_sorted = sorted(candidates, key=lambda c: -len(c[3]))
    
    # Index candidates by which triples they cover
    triple_to_cands = {t: [] for t in triples}
    for ci, (su, sv, sw, covered) in enumerate(candidates_sorted):
        for t in covered:
            triple_to_cands[t].append(ci)
    
    best_cover = None
    best_rank = max_rank + 1
    
    def backtrack(remaining, chosen, start_idx):
        nonlocal best_cover, best_rank
        if time.time() > deadline - 3:
            return
        if not remaining:
            rank = len(chosen)
            if rank < best_rank:
                best_rank = rank
                best_cover = list(chosen)
            return
        if len(chosen) >= best_rank:
            return
        
        # Pick the triple with fewest options
        target = min(remaining, key=lambda t: len([ci for ci in triple_to_cands[t] if ci >= start_idx]))
        
        for ci in triple_to_cands[target]:
            if ci < start_idx:
                continue
            su, sv, sw, covered = candidates_sorted[ci]
            if not covered.issubset(remaining):
                continue
            new_remaining = remaining - covered
            backtrack(new_remaining, chosen + [ci], ci + 1)
    
    backtrack(triples_set, [], 0)
    
    if best_cover is None:
        return None
    
    # Lift supports to coefficients
    r = len(best_cover)
    A = [[Fraction(0)] * r for _ in range(len(triples))]
    for ci, c_idx in enumerate(best_cover):
        su, sv, sw, covered = candidates_sorted[c_idx]
        for (i, j, k) in covered:
            tri_idx = i * m * p + j * p + k
            A[tri_idx][ci] = Fraction(1)
    
    b = [Fraction(1)] * len(triples)
    
    # Solve linear system
    def solve_linear(A_mat, b_vec):
        n_eq = len(A_mat)
        n_var = len(A_mat[0]) if n_eq > 0 else 0
        M = [row[:] + [b_vec[i]] for i, row in enumerate(A_mat)]
        n_rows = len(M)
        n_cols = len(M[0]) - 1
        rank = 0
        
        for col in range(n_cols):
            if time.time() > deadline - 2:
                return None
            pivot = None
            for row in range(rank, n_rows):
                if M[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            M[rank], M[pivot] = M[pivot], M[rank]
            pivot_val = M[rank][col]
            for row in range(n_rows):
                if row != rank and M[row][col] != 0:
                    factor = M[row][col] / pivot_val
                    for c2 in range(col, n_cols + 1):
                        M[row][c2] -= factor * M[rank][c2]
            rank += 1
        
        for row in range(rank, n_rows):
            if M[row][n_cols] != 0:
                return None
        
        sol = [Fraction(0)] * n_cols
        for row in range(rank):
            for col in range(n_cols):
                if M[row][col] != 0:
                    sol[col] = M[row][n_cols] / M[row][col]
                    break
        return sol
    
    t_sol = solve_linear(A, b)
    if t_sol is None:
        return None
    
    U = [[Fraction(0)] * r for _ in range(nm)]
    V = [[Fraction(0)] * r for _ in range(mp)]
    W = [[Fraction(0)] * r for _ in range(pn)]
    
    for ci, c_idx in enumerate(best_cover):
        su, sv, sw, covered = candidates_sorted[c_idx]
        tc = t_sol[ci]
        for u_idx in su:
            U[u_idx][ci] = tc
        for v_idx in sv:
            V[v_idx][ci] = Fraction(1)
        for w_idx in sw:
            W[w_idx][ci] = Fraction(1)
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    if _verify_strict(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return U_list, V_list, W_list, r
    
    return None


def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    # Start with the 2+1 split (rank 44)
    U, V, W = _build_2plus1_split()
    best_rank = 44
    best_result = (U, V, W)

    if not _verify_einsum(U, V, W, n, m, p):
        # Fallback to schoolbook
        Us, Vs, Ws = schoolbook(n, m, p)
        return {"U": Us, "V": Vs, "W": Ws}

    # Try to reduce rank via column merging
    if time.time() < deadline - 20:
        merged = _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=10)
        if merged is not None:
            U_m, V_m, W_m = merged
            if _verify_strict(U_m, V_m, W_m, n, m, p, tol=1e-10):
                best_rank = len(U_m[0])
                best_result = (U_m, V_m, W_m)

    # Try the 1+2 split as well
    if time.time() < deadline - 20:
        U12, V12, W12 = _build_1plus2_split()
        if _verify_einsum(U12, V12, W12, n, m, p):
            merged12 = _try_rank_reduction_via_merging(U12, V12, W12, n, m, p, max_time=5)
            if merged12 is not None:
                U12_m, V12_m, W12_m = merged12
                if _verify_strict(U12_m, V12_m, W12_m, n, m, p, tol=1e-10):
                    rank12 = len(U12_m[0])
                    if rank12 < best_rank:
                        best_rank = rank12
                        best_result = (U12_m, V12_m, W12_m)

    # Try optimization for lower ranks
    if time.time() < deadline - 15:
        for r in range(24, best_rank):
            if time.time() > deadline - 10:
                break
            
            result = _try_optimization_for_rank(r, n, m, p, deadline, n_restarts=8)
            if result is not None:
                U_opt, V_opt, W_opt, rank_opt = result
                if _verify_strict(U_opt, V_opt, W_opt, n, m, p, tol=1e-9):
                    best_rank = rank_opt
                    best_result = (U_opt, V_opt, W_opt)
                    # Try to polish
                    if time.time() < deadline - 5:
                        polished = _polish_solution(U_opt, V_opt, W_opt, n, m, p, deadline)
                        if polished is not None:
                            best_result = polished

    # Try support-based search
    if time.time() < deadline - 10:
        support_result = _try_support_based_search(n, m, p, deadline, max_rank=best_rank - 1)
        if support_result is not None:
            U_s, V_s, W_s, rank_s = support_result
            if rank_s < best_rank:
                best_rank = rank_s
                best_result = (U_s, V_s, W_s)

    # Final polish on the best result
    if time.time() < deadline - 3:
        polished = _polish_solution(best_result[0], best_result[1], best_result[2], n, m, p, deadline)
        if polished is not None:
            best_result = polished

    # Final verification
    if _verify_strict(best_result[0], best_result[1], best_result[2], n, m, p, tol=1e-10):
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

    # Fallback: return the best we have
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

# EVOLVE-BLOCK-END
