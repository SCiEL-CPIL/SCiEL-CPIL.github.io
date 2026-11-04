"""Matrix multiplication 4x4 by 4x8 rank-minimal search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 8]
n, m, p = DIMENSIONS

def _build_tensor():
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_float(U, V, W, T, tol=1e-9):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    T_rec = np.einsum('ic,jc,kc->ijk', Ua, Va, Wa)
    return np.allclose(T_rec, T, atol=tol)

def _verify_exact(U, V, W, T):
    rank = len(U[0]) if U else 0
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i][c]) * Fraction(V[j][c]) * Fraction(W[k][c])
                if s != Fraction(T[i, j, k]):
                    return False
    return True

def _strassen_2x2x2():
    U = [[1, 0, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [0, 1, 0, 0, 0, -1, 0],
         [1, 1, 0, 1, 0, 0, -1]]
    V = [[1, 1, 0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, -1, 0, 0, 1],
         [1, 0, -1, 0, 1, 0, 1]]
    W = [[1, 0, 0, 1, -1, 0, 1],
         [0, 0, 1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0, 0, 0],
         [1, -1, 1, 0, 0, 1, 0]]
    return U, V, W

def _kron_decomp(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    r = r1 * r2
    n1 = len(U1)
    n2 = len(U2)
    m1 = len(V1)
    m2 = len(V2)
    p1 = len(W1)
    p2 = len(W2)
    U = [[0] * r for _ in range(n1 * n2)]
    V = [[0] * r for _ in range(m1 * m2)]
    W = [[0] * r for _ in range(p1 * p2)]
    for i1 in range(n1):
        for i2 in range(n2):
            row = i1 * n2 + i2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    U[row][col] = U1[i1][c1] * U2[i2][c2]
    for j1 in range(m1):
        for j2 in range(m2):
            row = j1 * m2 + j2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    V[row][col] = V1[j1][c1] * V2[j2][c2]
    for k1 in range(p1):
        for k2 in range(p2):
            row = k1 * p2 + k2
            for c1 in range(r1):
                for c2 in range(r2):
                    col = c1 * r2 + c2
                    W[row][col] = W1[k1][c1] * W2[k2][c2]
    return U, V, W

def _strassen_4x4x4():
    U2, V2, W2 = _strassen_2x2x2()
    return _kron_decomp(U2, V2, W2, U2, V2, W2)

def _split_4x4x8(U44, V44, W44):
    """Split 4x4x4 into two 4x4x4 blocks for 4x4x8."""
    r = len(U44[0])
    U = [row + row for row in U44]
    V = []
    for row in V44:
        V.append(row + [0] * r)
    for row in V44:
        V.append([0] * r + row)
    W = []
    for row in W44:
        W.append(row + [0] * r)
    for row in W44:
        W.append([0] * r + row)
    return U, V, W

def _try_reduce_rank(U, V, W, T, max_iters=20, tol=1e-8):
    """Try to reduce rank by finding and eliminating redundant columns."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    for _iter in range(max_iters):
        reduced = False
        for c1 in range(rank):
            for c2 in range(c1 + 1, rank):
                u1 = Ua[:, c1]
                u2 = Ua[:, c2]
                v1 = Va[:, c1]
                v2 = Va[:, c2]
                w1 = Wa[:, c1]
                w2 = Wa[:, c2]
                
                nonzero = np.abs(u1) > tol
                if np.any(nonzero):
                    alphas_u = u2[nonzero] / u1[nonzero]
                    if np.allclose(alphas_u, alphas_u[0], atol=tol):
                        alpha = alphas_u[0]
                        if np.allclose(v2, alpha * v1, atol=tol) and np.allclose(w2, alpha * w1, atol=tol):
                            Ua = np.delete(Ua, c2, axis=1)
                            Va = np.delete(Va, c2, axis=1)
                            Wa = np.delete(Wa, c2, axis=1)
                            rank -= 1
                            reduced = True
                            break
            if reduced:
                break
        if not reduced:
            break
    
    if rank == len(U[0]):
        return None
    
    U_new = Ua.tolist()
    V_new = Va.tolist()
    W_new = Wa.tolist()
    return U_new, V_new, W_new

def _alternating_minimization(U, V, W, T, n_terms, max_iters=30, tol=1e-8):
    """Try to find a rank-n_terms decomposition via alternating minimization."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    np.random.seed(42)
    U_rand = np.random.randn(len(U), n_terms)
    V_rand = np.random.randn(len(V), n_terms)
    W_rand = np.random.randn(len(W), n_terms)
    
    for it in range(max_iters):
        for i in range(len(U)):
            M = np.zeros((len(V) * len(W), n_terms))
            for j in range(len(V)):
                for k in range(len(W)):
                    M[j * len(W) + k, :] = Va[j, :] * Wa[k, :]
            b = T[i, :, :].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=tol)
                U_rand[i, :] = sol
            except:
                pass
        
        for j in range(len(V)):
            M = np.zeros((len(U) * len(W), n_terms))
            for i in range(len(U)):
                for k in range(len(W)):
                    M[i * len(W) + k, :] = U_rand[i, :] * Wa[k, :]
            b = T[:, j, :].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=tol)
                V_rand[j, :] = sol
            except:
                pass
        
        for k in range(len(W)):
            M = np.zeros((len(U) * len(V), n_terms))
            for i in range(len(U)):
                for j in range(len(V)):
                    M[i * len(V) + j, :] = U_rand[i, :] * V_rand[j, :]
            b = T[:, :, k].flatten()
            try:
                sol, _, _, _ = np.linalg.lstsq(M, b, rcond=tol)
                W_rand[k, :] = sol
            except:
                pass
        
        T_rec = np.einsum('ic,jc,kc->ijk', U_rand, V_rand, W_rand)
        if np.allclose(T_rec, T, atol=1e-6):
            break
    
    T_rec = np.einsum('ic,jc,kc->ijk', U_rand, V_rand, W_rand)
    if not np.allclose(T_rec, T, atol=1e-5):
        return None
    
    U_rounded = np.round(U_rand * 2) / 2
    V_rounded = np.round(V_rand * 2) / 2
    W_rounded = np.round(W_rand * 2) / 2
    
    T_rec2 = np.einsum('ic,jc,kc->ijk', U_rounded, V_rounded, W_rounded)
    if not np.allclose(T_rec2, T, atol=1e-9):
        return None
    
    U_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U_rounded.tolist()]
    V_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V_rounded.tolist()]
    W_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W_rounded.tolist()]
    
    if _verify_exact(U_frac, V_frac, W_frac, T):
        return U_frac, V_frac, W_frac
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    T = _build_tensor()
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-based construction
    try:
        U44, V44, W44 = _strassen_4x4x4()
        U, V, W = _split_4x4x8(U44, V44, W44)
        if _verify_float(U, V, W, T):
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            if _verify_exact(U_frac, V_frac, W_frac, T):
                best_rank = len(U_frac[0])
                best = {"U": U_frac, "V": V_frac, "W": W_frac}
                print(f"Strassen 4x4x8: rank {best_rank}")
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Try rank reduction on the Strassen decomposition
    if best_rank < n * m * p:
        try:
            U, V, W = best["U"], best["V"], best["W"]
            reduced = _try_reduce_rank(U, V, W, T)
            if reduced:
                U_r, V_r, W_r = reduced
                U_frac = [[Fraction(x) for x in row] for row in U_r]
                V_frac = [[Fraction(x) for x in row] for row in V_r]
                W_frac = [[Fraction(x) for x in row] for row in W_r]
                if _verify_exact(U_frac, V_frac, W_frac, T):
                    best_rank = len(U_frac[0])
                    best = {"U": U_frac, "V": V_frac, "W": W_frac}
                    print(f"After reduction: rank {best_rank}")
        except Exception as e:
            print(f"Reduction failed: {e}")
    
    # Try alternating minimization for lower rank
    if time.time() < deadline - 40:
        for target_rank in [best_rank - 1, best_rank - 2, best_rank - 4, 64, 56, 49, 42, 40]:
            if time.time() > deadline - 15:
                break
            if target_rank < 1:
                continue
            try:
                result = _alternating_minimization(
                    best["U"], best["V"], best["W"], T, target_rank, max_iters=30
                )
                if result:
                    U_a, V_a, W_a = result
                    if len(U_a[0]) < best_rank:
                        best_rank = len(U_a[0])
                        best = {"U": U_a, "V": V_a, "W": W_a}
                        print(f"AM found rank {best_rank}")
                        break
            except Exception as e:
                pass
    
    return best
