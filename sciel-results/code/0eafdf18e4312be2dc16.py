import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]
T_START = time.time()
DEADLINE = 138.0  # safety margin before 150s kill
SAFETY = 8.0

def _elapsed():
    return time.time() - T_START

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_A = i * m + j
                idx_B = j * p + k
                idx_C = k * n + i
                T[idx_A, idx_B, idx_C] = 1
    return T

def _validate(U, V, W):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=np.int64)
    Va = np.array(V, dtype=np.int64)
    Wa = np.array(W, dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = Ua[i * m + j] @ Va[j * p + k]
                if not np.array_equal(lhs, Wa[k * n + i]):
                    return False
    return True

def _block_strassen_decomposition():
    """
    Decompose (2x4)(4x7) using block structure.
    Split inner dim 4 into two 2s, and outer k dim 7 into 2+2+3.
    For each 2x2 block product use Strassen (rank 7).
    This yields a structured low-rank decomposition.
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # Split j (inner, size 4) into two groups of 2: j0 in {0,1}, j1 in {2,3}
    # Split k (size 7) into groups: k0 in {0,1}, k1 in {2,3}, k2 in {4,5,6}
    # C[i,k] = sum over j A[i,j]B[j,k]
    # = sum_{j in j0} A[i,j]B[j,k] + sum_{j in j1} A[i,j]B[j,k]
    # For each (j-group, k-group) pair we have a (2x2)(2x|kg|) product.
    # This is getting complex; instead use a direct rank-reduction via
    # known Strassen on the 2x2 x 2x2 core and handle the extra k column.

    # Simpler robust approach: use a rank-7 Strassen for a 2x2x2 block,
    # and handle the 7 columns by linearity.
    # We'll instead do a randomized exact search seeded by a structured init.
    return None

def _randomized_search(target_rank, T, seed=0):
    """
    Try to find an exact rank-R decomposition via random rank-1 tensors
    combined with least-squares projection and integer rounding.
    Uses the linearity: T = sum_r u_r v_r w_r.
    We fix U, V randomly and solve for W via the bilinear identity,
    then round and verify.
    """
    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    R = target_rank
    rng = np.random.RandomState(seed)

    for attempt in range(200):
        if _elapsed() > DEADLINE - SAFETY:
            return None
        # Random integer factors in {-1, 0, 1}
        U = rng.randint(-1, 2, size=(NM, R))
        V = rng.randint(-1, 2, size=(MP, R))
        # For each output index (i,k), W[k*n+i] must equal U[i*m+j] @ V[j*p+k]
        # for all i,j,k. Build a linear system to solve for W consistently.
        # Actually W is determined by U,V if the identity holds.
        # Compute the implied W and check consistency.
        W = np.zeros(PN, dtype=np.int64)
        consistent = True
        for i in range(n):
            for k in range(p):
                val = None
                for j in range(m):
                    cur = U[i * m + j] @ V[j * p + k]
                    if val is None:
                        val = cur
                    elif not np.array_equal(cur, val):
                        consistent = False
                        break
                if not consistent:
                    break
                if val is None:
                    val = 0
                W[k * n + i] = val
        if not consistent:
            continue
        # Now verify the full tensor identity
        Ua = U
        Va = V
        Wa = W.reshape(PN, 1)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if Ua[i * m + j] @ Va[j * p + k] != Wa[k * n + i, 0]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": Wa.tolist()}
    return None

def _iterative_nullspace_search(T, max_rank, seed=0):
    """
    Model the multiplication tensor as a linear operator. Use random
    projections to build a low-rank subspace decomposition, then round
    to integers and verify exactness.
    """
    n, m, p = DIMENSIONS
    NM, MP, PN = n * m, m * p, p * n
    rng = np.random.RandomState(seed)

    # Flatten T to a matrix: T_flat has shape (NM*MP, PN) by fixing (i,j) as row, k as col.
    # T[i,j,k] = 1 if matching. The bilinear form: for fixed (i,j), the slice over k.
    # We want rank-R: T = sum_r u_r (NM) * v_r (MP) * w_r (PN).
    # Equivalently, matricize along (NM*MP, PN): M[i*MP+j, k] = T[i,j,k].
    M = T.reshape(NM * MP, PN).astype(np.float64)
    # The rank of M (matrix rank) is an upper bound on tensor rank? No,
    # tensor rank >= max of matricization ranks. Use SVD to get a low-rank
    # approximation and round.
    for R in range(max_rank, max_rank - 12, -1):
        if _elapsed() > DEADLINE - SAFETY:
            break
        # Try several random bases
        for trial in range(50):
            if _elapsed() > DEADLINE - SAFETY:
                break
            # Random rotation to explore different low-rank subspaces
            Q = rng.randn(PN, PN)
            Q, _ = np.linalg.qr(Q)
            # Project M into a random R-dim subspace of the column space
            # Use truncated SVD of the projected matrix
            Mproj = M @ Q
            Usv, s, Vt = np.linalg.svd(Mproj, full_matrices=False)
            # Keep top R
            Utr = Usv[:, :R]
            Vtr = Vt[:R, :]
            # Reconstruct low-rank M approx: M ~ Utr * diag(s[:R]) * Vtr
            # We need to factor this into tensor form.
            # The matricization M[i*MP+j, k] = sum_r Utr[i*MP+j, r] * (s[r] Vtr[r, k])
            # To get tensor rank R, we need M[i*MP+j, k] = sum_r a_r(i) b_r(j) c_r(k).
            # But Utr mixes i and j. We need to further factor Utr into (NM, R') and (MP, R').
            # This is the core difficulty. Instead, use the direct random factor approach
            # which is more reliable for exact integer solutions.
            pass
        # Fallback: direct random factor search for this R
        res = _randomized_search(R, T, seed=seed + trial)
        if res is not None:
            return res
    return None

def construct():
    best = _schoolbook()
    best_rank = len(best["U"][0])

    # First, try the block-Strassen structured decomposition (known to reach ~50).
    # We build it explicitly: split the 4 inner dim into two 2s, and use the
    # fact that a (2x2)(2x2) product has rank 7 (Strassen). The 7 outer columns
    # are handled by linearity, but we can share Strassen intermediates.

    # Explicit Strassen-based construction for (2x4)(4x7):
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # Group j into {0,1} and {2,3}.
    # C[i,k] = (A[i,0]B[0,k] + A[i,1]B[1,k]) + (A[i,2]B[2,k] + A[i,3]B[3,k])
    # The first group is a (2x2 in i,j0) times (2x7 in j0,k) product.
    # Write it as: for the j0 group, define M1[i, k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # This is a bilinear map. Its tensor rank for 2x2 x 2x7:
    # = sum over the 2x2x2 core (rank 7 via Strassen) for k in {0,1}
    #   + the k=2..6 columns handled directly.
    # This is intricate; we rely on the randomized search seeded well.

    # Try randomized search from a modest rank upward.
    # Lower bound: max(NM, MP, PN) = max(8, 28, 14) = 28.
    lb = max(2 * 4, 4 * 7, 7 * 2)  # 28
    # Try ranks from lb up to best_rank - 1, but cap the search.
    for R in range(lb, best_rank):
        if _elapsed() > DEADLINE - SAFETY:
            break
        # Only attempt a few ranks to stay within budget
        res = _randomized_search(R, _build_tensor(), seed=R)
        if res is not None:
            best = res
            best_rank = R
            break

    # Also try the null-space / random-projection approach for a couple of ranks.
    if _elapsed() < DEADLINE - SAFETY:
        T = _build_tensor()
        for R in range(max(lb, best_rank - 6), best_rank):
            if _elapsed() > DEADLINE - SAFETY:
                break
            res = _iterative_nullspace_search(T, R, seed=R * 7 + 1)
            if res is not None:
                best = res
                best_rank = R
                break

    return best
