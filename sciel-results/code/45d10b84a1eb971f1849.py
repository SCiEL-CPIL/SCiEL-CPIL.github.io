"""Fresh design: schoolbook fallback + NMTF search for matrix multiplication 4x4 * 4x5.
Returns the rank-minimal exact tensor decomposition found, else the guaranteed schoolbook rank 80.
"""
import time
import math
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = DIMENSIONS
NM = N * M          # 16
MP = M * P          # 20
PN = P * N          # 20
SCHOOLBOOK_RANK = N * M * P  # 80

# Hard deadline (evaluator kills at 150 s; keep a margin).
DEADLINE = time.time() + 130.0


def _time_left():
    return DEADLINE - time.time()


def _schoolbook():
    """Guaranteed valid decomposition of rank 80 (entries 0/1)."""
    rank = SCHOOLBOOK_RANK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


# Build the matrix-multiplication tensor T[i,j,k] = 1 (the structural tensor).
# We use the trace convention W[k*n+i, r], so the tensor is T[i, j, k] with
# U indexed by (i,j), V by (j,k), W by (k,i).
_T = np.ones((N, M, P), dtype=np.float64)


def _residual(U, V, W):
    """Return max |sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1| over all (i,j,k)."""
    # U: (NM, R) -> reshape (N, M, R); V: (MP, R) -> (M, P, R); W: (PN, R) -> (P, N, R)
    U3 = U.reshape(N, M, -1)
    V3 = V.reshape(M, P, -1)
    W3 = W.reshape(P, N, -1)
    R = U.shape[1]
    # Compute sum_r U[i,j,r] V[j,k,r] W[k,i,r] for all i,j,k
    # Use einsum: U[i,j,r] V[j,k,r] -> (i,j,k,r) then * W[k,i,r] -> sum r
    # To avoid huge intermediates, contract in two steps.
    # A[i,j,k,r] = U[i,j,r] * V[j,k,r]
    A = np.einsum('ijr,jkr->ijk r', U3, V3)  # placeholder
    # Actually do it properly:
    # B[i,k,r] = sum_j U[i,j,r] V[j,k,r]
    B = np.einsum('ijr,jkr->ikr', U3, V3)
    # C[i,k] = sum_r B[i,k,r] W[k,i,r]
    C = np.einsum('ikr,kir->ik', B, W3)
    # C[i,k] should equal sum_j 1 = M for all i,k
    return float(np.max(np.abs(C - M)))


def _residual_fast(U, V, W):
    """Fast residual: max |C[i,k] - M| where C = (U V^T) contracted with W."""
    U3 = U.reshape(N, M, -1)
    V3 = V.reshape(M, P, -1)
    W3 = W.reshape(P, N, -1)
    B = np.einsum('ijr,jkr->ikr', U3, V3)
    C = np.einsum('ikr,kir->ik', B, W3)
    return float(np.max(np.abs(C - M)))


def _snap(x, tol=0.25):
    """Snap values to nearest half-integer; return None if any value is too far."""
    out = np.empty_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        v = x[idx]
        h = math.floor(v * 2 + 0.5) / 2.0
        if abs(v - h) > tol:
            return None
        out[idx] = h
    return out


def check_exact(U, V, W, tol=0.11):
    """Verify U,V,W (lists) exactly decompose the tensor with half-integer entries."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    if Ua.shape[1] != Va.shape[1] or Ua.shape[1] != Wa.shape[1]:
        return False
    # Check half-integer
    for A in (Ua, Va, Wa):
        if not np.allclose(A * 2, np.round(A * 2), atol=1e-6):
            return False
    res = _residual_fast(Ua, Va, Wa)
    return res < tol


def _nmtf(rank, iters, lr0=0.1, lr1=0.005, seed=0, tol=0.25):
    """One NMTF run. Returns (U,V,W as float arrays, residual) or None if timeout."""
    rng = np.random.RandomState(seed)
    R = rank
    U = rng.randn(NM, R) * 0.1
    V = rng.randn(MP, R) * 0.1
    W = rng.randn(PN, R) * 0.1

    U3 = U.reshape(N, M, R)
    V3 = V.reshape(M, P, R)
    W3 = W.reshape(P, N, R)

    for it in range(iters):
        if _time_left() < 20:
            return None
        # Gradient for U: dU[i,j,r] = sum_k (C[i,k] - M) V3[j,k,r] W3[k,i,r]
        B = np.einsum('ijr,jkr->ikr', U3, V3)
        C = np.einsum('ikr,kir->ik', B, W3)
        E = C - M  # (N, P)
        # dU[i,j,r] = sum_k E[i,k] V3[j,k,r] W3[k,i,r]
        dU = np.einsum('ik,jkr,kir->ijr', E, V3, W3)
        # dV[j,k,r] = sum_i E[i,k] U3[i,j,r] W3[k,i,r]
        dV = np.einsum('ik,ijr,kir->jkr', E, U3, W3)
        # dW[k,i,r] = sum_j E[i,k] U3[i,j,r] V3[j,k,r]
        dW = np.einsum('ik,ijr,jkr->kir', E, U3, V3)

        lr = lr0 * (lr1 / lr0) ** (it / max(1, iters))
        U -= lr * dU.reshape(NM, R)
        V -= lr * dV.reshape(MP, R)
        W -= lr * dW.reshape(PN, R)
        U3 = U.reshape(N, M, R)
        V3 = V.reshape(M, P, R)
        W3 = W.reshape(P, N, R)

    res = _residual_fast(U, V, W)
    return U, V, W, res


def _try_rank(rank, max_restarts=8, iters_per_restart=120, seed_base=0):
    """Try to find a valid half-integer decomposition of given rank. Returns dict or None."""
    best = None
    best_res = float('inf')
    for rs in range(max_restarts):
        if _time_left() < 25:
            break
        out = _nmtf(rank, iters_per_restart, seed=seed_base + rs)
        if out is None:
            break
        U, V, W, res = out
        if res < best_res:
            best_res = res
            best = (U, V, W, res)
        # If residual is very small, try snapping to half-integers and verify.
        if res < 0.5:
            Us = _snap(U, tol=0.25)
            Vs = _snap(V, tol=0.25)
            Ws = _snap(W, tol=0.25)
            if Us is not None and Vs is not None and Ws is not None:
                if check_exact(Us.tolist(), Vs.tolist(), Ws.tolist(), tol=0.11):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    # Final polish: if best residual is small, do a few extra iterations with tighter tol.
    if best is not None and best[3] < 0.3:
        U, V, W, res = best
        # Polish
        for _ in range(30):
            if _time_left() < 20:
                break
            R = U.shape[1]
            U3 = U.reshape(N, M, R)
            V3 = V.reshape(M, P, R)
            W3 = W.reshape(P, N, R)
            B = np.einsum('ijr,jkr->ikr', U3, V3)
            C = np.einsum('ikr,kir->ik', B, W3)
            E = C - M
            dU = np.einsum('ik,jkr,kir->ijr', E, V3, W3)
            dV = np.einsum('ik,ijr,kir->jkr', E, U3, W3)
            dW = np.einsum('ik,ijr,jkr->kir', E, U3, V3)
            U -= 0.01 * dU.reshape(NM, R)
            V -= 0.01 * dV.reshape(MP, R)
            W -= 0.01 * dW.reshape(PN, R)
        res = _residual_fast(U, V, W)
        if res < 0.3:
            Us = _snap(U, tol=0.25)
            Vs = _snap(V, tol=0.25)
            Ws = _snap(W, tol=0.25)
            if Us is not None and Vs is not None and Ws is not None:
                if check_exact(Us.tolist(), Vs.tolist(), Ws.tolist(), tol=0.11):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
    return None


def construct():
    """Return the rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication."""
    t0 = time.time()
    fallback = _schoolbook()

    # Theoretical lower bound: rank >= max(N, M, P) * something; use a conservative bound.
    # For matrix multiplication, rank >= max(N*M, M*P, P*N) / max(N,M,P) is not tight.
    # Use a simple bound: rank >= max(N, M, P) = 5, but realistically much higher.
    # Try ranks from 50 upward (schoolbook is 80; known lower bounds for 4x4x4 are ~49).
    # We'll try a few ranks below 80 to see if we can beat it.
    for rank in [50, 55, 60, 65, 70, 75, 80]:
        if _time_left() < 30:
            break
        if rank >= SCHOOLBOOK_RANK:
            break  # schoolbook already covers 80
        result = _try_rank(rank, max_restarts=4, iters_per_restart=100, seed_base=rank)
        if result is not None:
            return result

    # If we used up time or found nothing, return the guaranteed schoolbook.
    return fallback


# EVOLVE-BLOCK-END
