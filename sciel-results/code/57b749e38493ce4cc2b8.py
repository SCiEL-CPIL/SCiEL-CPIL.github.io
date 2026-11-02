import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _kronecker_decomposition(n, m, p):
    """
    Decompose T[n,m,p] = T[n1,m1,p1] ⊗ T[n2,m2,p2]
    where n=n1*n2, m=m1*m2, p=p1*p2.
    The resulting rank is rank1 * rank2.
    Index mapping:
      T[i,j,k] with i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
      T[i,j,k] = T1[i1,j1,k1] * T2[i2,j2,k2]
    Factor matrices:
      U[i*m+j, r] = U1[i1*m1+j1, r1] * U2[i2*m2+j2, r2], r = r1*rank2 + r2
      V[j*p+k, r] = V1[j1*p1+k1, r1] * V2[j2*p2+k2, r2]
      W[k*n+i, r] = W1[k1*n1+i1, r1] * W2[k2*n2+i2, r2]
    """
    # Try to factor dimensions
    def factors(x):
        out = []
        for a in range(2, x + 1):
            if x % a == 0:
                out.append((a, x // a))
        return out

    best = None
    for (n1, n2) in factors(n):
        for (m1, m2) in factors(m):
            for (p1, p2) in factors(p):
                # Build sub-decompositions (schoolbook for now, could be Strassen)
                U1, V1, W1, r1 = _schoolbook(n1, m1, p1)
                U2, V2, W2, r2 = _schoolbook(n2, m2, p2)
                rank = r1 * r2
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                for i1 in range(n1):
                    for i2 in range(n2):
                        for j1 in range(m1):
                            for j2 in range(m2):
                                for k1 in range(p1):
                                    for k2 in range(p2):
                                        i = i1 * n2 + i2
                                        j = j1 * m2 + j2
                                        k = k1 * p2 + k2
                                        for r1 in range(r1):
                                            for r2 in range(r2):
                                                r = r1 * r2 + r2
                                                U[i * m + j, r] = U1[i1 * m1 + j1, r1] * U2[i2 * m2 + j2, r2]
                                                V[j * p + k, r] = V1[j1 * p1 + k1, r1] * V2[j2 * p2 + k2, r2]
                                                W[k * n + i, r] = W1[k1 * n1 + i1, r1] * W2[k2 * n2 + i2, r2]
                if best is None or rank < best[3]:
                    best = (U, V, W, rank)
    return best

def _random_refined_search(n, m, p, deadline, seed=0):
    """
    Structured random search with numerical refinement.
    Objective: minimize sum over (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    Use a smooth least-squares surrogate, then polish to exact.
    """
    rng = np.random.default_rng(seed)
    best_rank = n * m * p
    best = None

    # Try a few target ranks below schoolbook
    for target_rank in range(best_rank - 1, max(1, best_rank - 12), -1):
        if time.time() > deadline - 5:
            break
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            # Initialize with schoolbook-like structure plus noise
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            # Seed with schoolbook first target_rank terms
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            # Add small random perturbation to encourage exploration
            U += rng.normal(0, 0.1, U.shape)
            V += rng.normal(0, 0.1, V.shape)
            W += rng.normal(0, 0.1, W.shape)

            # Gradient-free refinement: coordinate descent on rank terms
            # Smooth surrogate: least squares
            def loss():
                total = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(target_rank):
                                s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            total += (s - 1.0) ** 2
                return total

            # Simple local search: try small perturbations on each term
            improved = True
            step = 0.05
            while improved and time.time() < deadline - 3:
                improved = False
                for r in range(target_rank):
                    for mat, idx in [(U, (slice(None), r)), (V, (slice(None), r)), (W, (slice(None), r))]:
                        for _ in range(2):
                            base = loss()
                            idx_flat = idx[0]
                            mat[idx_flat] += step
                            new = loss()
                            if new < base - 1e-12:
                                improved = True
                            else:
                                mat[idx_flat] -= 2 * step
                                new = loss()
                                if new < base - 1e-12:
                                    improved = True
                                else:
                                    mat[idx_flat] += step
                step *= 0.8

            # Check if we reached near-exact
            if loss() < 1e-6:
                if _verify_decomposition(U, V, W, n, m, p):
                    if best is None or target_rank < best[3]:
                        best = (U.copy(), V.copy(), W.copy(), target_rank)
    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try Kronecker product decomposition (structured, guaranteed valid)
    if time.time() < deadline - 30:
        kr = _kronecker_decomposition(n, m, p)
        if kr is not None:
            Uk, Vk, Wk, rankk = kr
            if rankk < best_rank:
                if _verify_decomposition(Uk, Vk, Wk, n, m, p):
                    best_U, best_V, best_W = Uk, Vk, Wk
                    best_rank = rankk

    # Try randomized numerical refinement to push below
    if time.time() < deadline - 10:
        rr = _random_refined_search(n, m, p, deadline, seed=42)
        if rr is not None:
            Ur, Vr, Wr, rankr = rr
            if rankr < best_rank:
                if _verify_decomposition(Ur, Vr, Wr, n, m, p):
                    best_U, best_V, best_W = Ur, Vr, Wr
                    best_rank = rankr

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
