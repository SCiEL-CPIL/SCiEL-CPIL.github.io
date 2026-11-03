import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _strassen_2x2x2_terms():
    """Return 7 rank-1 terms for M_{2,2,2}. Each term: (u_dict, v_dict, w_dict)
    with keys in local 2x2 coords: U over (i,j), V over (j,k), W over (k,i)."""
    terms = []
    # P1 = (A11+A22)(B11+B22)
    terms.append((
        {(0,0):1,(1,1):1},
        {(0,0):1,(1,1):1},
        {(0,0):1,(1,1):1}
    ))
    # P2 = (A21+A22)B11
    terms.append((
        {(1,0):1,(1,1):1},
        {(0,0):1},
        {(1,0):1,(1,1):-1}
    ))
    # P3 = A11(B12-B22)
    terms.append((
        {(0,0):1},
        {(0,1):1,(1,1):-1},
        {(0,1):1,(1,1):1}
    ))
    # P4 = A22(B21-B11)
    terms.append((
        {(1,1):1},
        {(1,0):1,(0,0):-1},
        {(0,0):1,(1,0):1}
    ))
    # P5 = (A11+A12)B22
    terms.append((
        {(0,0):1,(0,1):1},
        {(1,1):1},
        {(0,0):-1,(0,1):1}
    ))
    # P6 = (A21-A11)(B11+B12)
    terms.append((
        {(1,0):1,(0,0):-1},
        {(0,0):1,(0,1):1},
        {(1,1):1}
    ))
    # P7 = (A12-A22)(B21+B22)
    terms.append((
        {(0,1):1,(1,1):-1},
        {(1,0):1,(1,1):1},
        {(0,0):1}
    ))
    return terms

def _build_structured(n, m, p):
    """Build a structured decomposition for n x m * m x p using a 2x2 Strassen block
    plus schoolbook for the remaining parts. Returns (U, V, W) as lists of lists."""
    # n=2 rows of A, m=3 cols of A / rows of B, p=9 cols of B
    # Split A cols into 2 (Strassen) + 1 (schoolbook)
    # Split B rows into 2 (Strassen) + 1 (schoolbook)
    # C = A[:,0:2] B[0:2,:] + A[:,2] B[2,:]
    # Part 1: 2x2 * 2xp  -> use Strassen over 2x2, each product gives 1xp row
    # Part 2: 2x1 * 1xp  -> schoolbook rank 2*p

    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18

    U = [0.0] * U_rows
    V = [0.0] * V_rows
    W = [0.0] * W_rows
    # We'll accumulate into dict-of-terms then flatten
    terms = []  # list of (u_dict, v_dict, w_dict) over global indices

    # Part 1: 2x2 * 2xp using Strassen.
    # Local Strassen indices: A rows i in {0,1}, A cols j in {0,1} (global j = j)
    # B rows j in {0,1} (global j = j), B cols k in {0..p-1} (global k = k)
    # C rows i in {0,1}, C cols k in {0..p-1}
    strassen = _strassen_2x2x2_terms()
    for (u_dict, v_dict, w_dict) in strassen:
        u_g = {}
        v_g = {}
        w_g = {}
        for (i, j), val in u_dict.items():
            u_g[i * m + j] = val
        for (j, k), val in v_dict.items():
            # B row j (global j), B col k (global k)
            v_g[j * p + k] = val
        for (k, i), val in w_dict.items():
            # C col k, C row i -> index k*n + i
            w_g[k * n + i] = val
        terms.append((u_g, v_g, w_g))

    # Part 2: A[:,2] (2x1) * B[2,:] (1xp) schoolbook, rank 2*p
    for i in range(n):
        for k in range(p):
            u_g = {i * m + 2: 1.0}
            v_g = {2 * p + k: 1.0}
            w_g = {k * n + i: 1.0}
            terms.append((u_g, v_g, w_g))

    # Flatten terms into U, V, W matrices
    R = len(terms)
    U = [[0.0] * R for _ in range(U_rows)]
    V = [[0.0] * R for _ in range(V_rows)]
    W = [[0.0] * R for _ in range(W_rows)]
    for c, (u_g, v_g, w_g) in enumerate(terms):
        for idx, val in u_g.items():
            U[idx][c] = val
        for idx, val in v_g.items():
            V[idx][c] = val
        for idx, val in w_g.items():
            W[idx][c] = val
    return U, V, W

def _verify(U, V, W, n, m, p, atol=1e-8):
    """Check that the decomposition exactly realizes matrix multiplication."""
    R = len(U[0])
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                # C[i,k] should get contribution sum_j A[i,j]B[j,k]
                # In decomposition: sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c]
                # We verify the tensor equals the matmul tensor.
                pass
    # Build the implied tensor T[i,j,k] = sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c]
    # and compare to matmul tensor M[i,j,k] = delta structure:
    # The matmul tensor has T[i,j,k] = 1 if (i,j,k) corresponds to A[i,j]*B[j,k] -> C[i,k]
    # i.e., the standard tensor: T[i,j,k] = 1 for all i,j,k (the elementary product).
    # Actually the matmul tensor T has T[i,j,k] = 1 for every (i,j,k) because
    # C[i,k] = sum_j A[i,j] B[j,k] means tensor entry (i,j,k) = 1.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                implied = 0.0
                for c in range(R):
                    implied += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(implied - 1.0) > atol:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=200):
    """Try to eliminate redundant rank-1 terms. Returns (U,V,W) possibly smaller."""
    R = len(U[0])
    # Represent each term's support
    def term_support(c):
        u = {i: U[i][c] for i in range(len(U)) if U[i][c] != 0}
        v = {j: V[j][c] for j in range(len(V)) if V[j][c] != 0}
        w = {k: W[k][c] for k in range(len(W)) if W[k][c] != 0}
        return u, v, w

    active = list(range(R))
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        found = False
        for idx_pos, c in enumerate(active):
            if time.time() > deadline:
                break
            u, v, w = term_support(c)
            if not (u and v and w):
                active.remove(c)
                found = True
                break
            # Build linear system: can term c be expressed as combo of others?
            others = [o for o in active if o != c]
            # positions where term c is nonzero
            positions = []
            for i, ui in u.items():
                for j, vj in v.items():
                    for k, wk in w.items():
                        positions.append((i, j, k, ui * vj * wk))
            if not positions:
                active.remove(c)
                found = True
                break
            A_rows = []
            b_vec = []
            for (i, j, k, target) in positions:
                row = []
                for o in others:
                    uo, vo, wo = term_support(o)
                    if i in uo and j in vo and k in wo:
                        row.append(uo[i] * vo[j] * wo[k])
                    else:
                        row.append(0.0)
                A_rows.append(row)
                b_vec.append(target)
            A = np.array(A_rows, dtype=float)
            b = np.array(b_vec, dtype=float)
            try:
                if len(others) == 0:
                    continue
                coeffs, res, rankA, sv = np.linalg.lstsq(A, b, rcond=1e-10)
                if len(res) > 0:
                    residual = res[0]
                else:
                    residual = np.linalg.norm(A @ coeffs - b)
                bnorm = np.linalg.norm(b)
                if bnorm == 0:
                    active.remove(c)
                    found = True
                    break
                if residual < 1e-8 * bnorm:
                    active.remove(c)
                    found = True
                    break
            except Exception:
                pass
        if not found:
            break

    if len(active) >= R:
        return U, V, W
    new_R = len(active)
    nU, nV, nW = len(U), len(V), len(W)
    newU = [[0.0]*new_R for _ in range(nU)]
    newV = [[0.0]*new_R for _ in range(nV)]
    newW = [[0.0]*new_R for _ in range(nW)]
    for nc, old in enumerate(active):
        for i in range(nU):
            newU[i][nc] = U[i][old]
        for j in range(nV):
            newV[j][nc] = V[j][old]
        for k in range(nW):
            newW[k][nc] = W[k][old]
    return newU, newV, newW

def _random_search(n, m, p, R, deadline, max_attempts):
    """Random search with {-1,0,1} entries, solve for W with half-integer rounding."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        rng = np.random.RandomState(attempt * 1000 + R)
        U = rng.randint(-1, 2, (n * m, R))
        V = rng.randint(-1, 2, (m * p, R))
        W = np.zeros((p * n, R))
        valid = True
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, R))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w_ls = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    w_half = np.round(w_ls * 2) / 2
                    if not np.allclose(M @ w_half, np.ones(m), atol=1e-9):
                        valid = False
                        break
                    W[k * n + i, :] = w_half
                except Exception:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 115

    # Schoolbook fallback
    rank_fb = n * m * p  # 54
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1.0
                V_fb[j * p + k, col] = 1.0
                W_fb[k * n + i, col] = 1.0
                col += 1

    best = (rank_fb, U_fb.tolist(), V_fb.tolist(), W_fb.tolist())

    # 1. Structured Strassen-based decomposition, then reduce rank
    try:
        U_s, V_s, W_s = _build_structured(n, m, p)
        if _verify(U_s, V_s, W_s, n, m, p):
            R_s = len(U_s[0])
            if R_s < best[0]:
                best = (R_s, U_s, V_s, W_s)
            # Try rank reduction
            U_r, V_r, W_r = _try_reduce_rank(U_s, V_s, W_s, n, m, p, deadline)
            if _verify(U_r, V_r, W_r, n, m, p):
                R_r = len(U_r[0])
                if R_r < best[0]:
                    best = (R_r, U_r, V_r, W_r)
    except Exception:
        pass

    # 2. Random search for lower rank
    for R in [best[0] - 1, best[0] - 2, best[0] - 3, 50, 49]:
        if R < n * m:  # need at least n*m terms to cover U rows
            continue
        if time.time() > deadline - 10:
            break
        res = _random_search(n, m, p, R, deadline, max_attempts=150)
        if res is not None:
            U_t, V_t, W_t = res
            if _verify(U_t, V_t, W_t, n, m, p):
                if R < best[0]:
                    best = (R, U_t, V_t, W_t)
                break

    return {"U": best[1], "V": best[2], "W": best[3]}

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
