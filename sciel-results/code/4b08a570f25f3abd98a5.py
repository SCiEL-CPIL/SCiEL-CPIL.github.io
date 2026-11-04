"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 120  # safety margin under the 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    # Build T via einsum
    # U: (n*m, rank) -> (n, m, rank); V: (m*p, rank) -> (m, p, rank); W: (p*n, rank) -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Target: T[i,j,k] = 1 for all i,j,k (matrix mult tensor is all-ones)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _to_halfint(Mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * Mat) / 2.0

def _try_rank(rank, seed, time_budget):
    """Try to find an exact half-integer decomposition of given rank."""
    np.random.seed(seed)
    n, m, p = N, M, P
    best_err = np.inf
    best = None
    # Random init
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    # Normalize to reduce condition issues
    for r in range(rank):
        nU = np.linalg.norm(U[:, r]); nV = np.linalg.norm(V[:, r]); nW = np.linalg.norm(W[:, r])
        if nU < 1e-12 or nV < 1e-12 or nW < 1e-12:
            continue
        # Move scale from U to W
        U[:, r] /= nU
        V[:, r] /= nV
        W[:, r] /= nW
        # Rebalance: put equal weight
        scale = (nU * nV * nW) ** (1.0 / 3.0)
        U[:, r] *= scale
        V[:, r] *= scale
        W[:, r] *= scale

    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)

    # Simple gradient-free local search: coordinate descent on the "product" objective
    # Objective: sum_{i,j,k} (sum_r U3[i,j,r]*V3[j,k,r]*W3[k,i,r] - 1)^2
    steps = 400
    for it in range(steps):
        if time.time() > time_budget:
            break
        # Round to half-integers and check
        Ur = _to_halfint(U)
        Vr = _to_halfint(V)
        Wr = _to_halfint(W)
        U3r = Ur.reshape(n, m, rank)
        V3r = Vr.reshape(m, p, rank)
        W3r = Wr.reshape(p, n, rank)
        Tr = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r)
        err = np.max(np.abs(Tr - 1.0))
        if err < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
        if err < best_err:
            best_err = err
            best = (Ur.copy(), Vr.copy(), Wr.copy())

        # Perturbation: pick a random entry in one matrix and nudge
        # Use a simple random walk with occasional larger jumps
        choice = np.random.randint(3)
        if choice == 0:
            idx = (np.random.randint(n * m), np.random.randint(rank))
            U[idx] += np.random.randn() * 0.1
        elif choice == 1:
            idx = (np.random.randint(m * p), np.random.randint(rank))
            V[idx] += np.random.randn() * 0.1
        else:
            idx = (np.random.randint(p * n), np.random.randint(rank))
            W[idx] += np.random.randn() * 0.1

    # Final check on best
    if best is not None:
        Ur, Vr, Wr = best
        U3r = Ur.reshape(n, m, rank)
        V3r = Vr.reshape(m, p, rank)
        W3r = Wr.reshape(p, n, rank)
        Tr = np.einsum('ijr,jkr,kir->ijk', U3r, V3r, W3r)
        if np.max(np.abs(Tr - 1.0)) < 1e-9:
            return Ur.tolist(), Vr.tolist(), Wr.tolist()
    return None

def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    # Schoolbook fallback (always valid)
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Known lower bound: rank >= max(n*m, m*p, p*n) = 20
    # Try ranks from 20 upward
    for rank in range(20, best_rank + 1):
        if time.time() > DEADLINE - 5:
            break
        # Multiple seeds per rank
        for seed in range(50):
            if time.time() > DEADLINE - 5:
                break
            result = _try_rank(rank, seed, DEADLINE - 3)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify(U, V, W):
                    return {"U": U, "V": V, "W": W}
    return best_result

# EVOLVE-BLOCK-END
