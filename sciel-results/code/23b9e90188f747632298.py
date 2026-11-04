"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DL = t0 + 110  # safety margin before 120s hard kill

    n, m, p = DIMENSIONS
    Nr = n * m          # 16 rows for U
    Mr = m * p          # 36 rows for V
    Pr = p * n          # 36 rows for W

    # Build the set of (a,b,c) triples where T[a,b,c] = 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    Tset = set(triples)
    ntrip = len(triples)  # 144

    # Precompute index maps for fast target-vector construction
    # For each a, list of (b,c) where T[a,b,c]=1
    a_to_bc = {a: [] for a in range(Nr)}
    b_to_ac = {b: [] for b in range(Mr)}
    c_to_ab = {c: [] for c in range(Pr)}
    for (a, b, c) in triples:
        a_to_bc[a].append((b, c))
        b_to_ac[b].append((a, c))
        c_to_ab[c].append((a, b))

    # ---- schoolbook rank-144 baseline ----
    R0 = ntrip
    U0 = np.zeros((Nr, R0))
    V0 = np.zeros((Mr, R0))
    W0 = np.zeros((Pr, R0))
    for idx, (a, b, c) in enumerate(triples):
        U0[a, idx] = 1.0
        V0[b, idx] = 1.0
        W0[c, idx] = 1.0

    def _check_full(U, V, W, tol=1e-9):
        """Vectorized exact check."""
        # Compute full tensor T_hat[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can do this by iterating r, but vectorized over a,b,c is faster if we use broadcasting?
        # T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 16x36x36 array.
        T_hat = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check against target
        # Target is 1 at triples, 0 elsewhere.
        # Create target tensor
        T_target = np.zeros((Nr, Mr, Pr))
        for (a, b, c) in triples:
            T_target[a, b, c] = 1.0
            
        # Check max diff
        if np.max(np.abs(T_hat - T_target)) > tol:
            return False
        return True

    def _check_triples(U, V, W, tol=1e-9):
        """Check only the 144 triple entries (faster, for iteration control)."""
        for (a, b, c) in triples:
            if abs(np.dot(U[a] * V[b], W[c]) - 1.0) > tol:
                return False
        return True

    def _err(U, V, W):
        e = 0.0
        for (a, b, c) in triples:
            d = np.dot(U[a] * V[b], W[c]) - 1.0
            e += d * d
        return e

    # ---- ALS with structured initialization from rank reduction ----
    def _als(U_init, V_init, W_init, rank, max_it=200, seed=0):
        rng = np.random.RandomState(seed)
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        if U.shape[1] != rank:
            U = rng.randn(Nr, rank) * 0.1
            V = rng.randn(Mr, rank) * 0.1
            W = rng.randn(Pr, rank) * 0.1

        prev_e = float('inf')
        for it in range(max_it):
            if time.time() > DL - 8:
                break

            # --- update U ---
            B = np.stack([np.outer(V[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for a in range(Nr):
                tv = np.zeros(Mr * Pr)
                for (b, c) in a_to_bc[a]:
                    tv[b * Pr + c] = 1.0
                U[a] = np.linalg.lstsq(B.T, tv, rcond=None)[0]

            # --- update V ---
            C = np.stack([np.outer(U[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for b in range(Mr):
                tv = np.zeros(Nr * Pr)
                for (a, c) in b_to_ac[b]:
                    tv[a * Pr + c] = 1.0
                V[b] = np.linalg.lstsq(C.T, tv, rcond=None)[0]

            # --- update W ---
            D = np.stack([np.outer(U[:, r], V[:, r]).ravel() for r in range(rank)], axis=0)
            for c in range(Pr):
                tv = np.zeros(Nr * Mr)
                for (a, b) in c_to_ab[c]:
                    tv[a * Mr + b] = 1.0
                W[c] = np.linalg.lstsq(D.T, tv, rcond=None)[0]

            e = _err(U, V, W)
            if abs(prev_e - e) < 1e-12 * max(1.0, prev_e):
                break
            prev_e = e

        return U, V, W

    def _polish(U, V, W, rank, max_it=100, seed=0):
        """High-precision polish: very tight ALS + final rounding attempt."""
        U, V, W = _als(U, V, W, rank, max_it=max_it, seed=seed)
        return U, V, W

    def _try_rank(rank, n_seeds=4):
        """Attempt to find a valid decomposition at a given rank."""
        if time.time() > DL - 12:
            return None

        for s in range(n_seeds):
            if time.time() > DL - 12:
                break

            # Initialization: subsample the schoolbook factors
            # Pick 'rank' columns out of 144, or random init
            if s == 0:
                # Structured init: take first 'rank' schoolbook columns
                idx = list(range(rank)) if rank <= R0 else list(range(R0))
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
            elif s == 1:
                # Random subset
                rng0 = np.random.RandomState(1000 + s)
                idx = rng0.choice(R0, size=min(rank, R0), replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
            else:
                # Pure random init
                rng0 = np.random.RandomState(2000 + s)
                U = rng0.randn(Nr, rank) * 0.05
                V = rng0.randn(Mr, rank) * 0.05
                W = rng0.randn(Pr, rank) * 0.05

            U, V, W = _als(U, V, W, rank, max_it=150, seed=s)
            if not _check_triples(U, V, W, tol=1e-8):
                continue

            # Polish
            U, V, W = _polish(U, V, W, rank, max_it=80, seed=s)
            if not _check_triples(U, V, W, tol=1e-9):
                continue

            # Full exact check
            if _check_full(U, V, W, tol=1e-9):
                # Try half-integer rounding
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if _check_full(Ur, Vr, Wr, tol=1e-9):
                    return Ur, Vr, Wr
                # Try integer rounding
                Ui = np.round(U)
                Vi = np.round(V)
                Wi = np.round(W)
                if _check_full(Ui, Vi, Wi, tol=1e-9):
                    return Ui, Vi, Wi
                return U, V, W

        return None

    # ---- search ranks from lower bound upward ----
    # Lower bound: max(Nr, Mr, Pr) = 36
    # Known: schoolbook = 144. Try to beat that.
    # For 4x4x9, the matrix multiplication rank is at most 4*4*9=144.
    # There are known results: rank(4x4x4) <= 49 (Strassen improvements)
    # For 4x4x9 we can try structured block decompositions.

    # Strategy A: structured "block" decomposition
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Group by j (the shared middle index). For each j, we have n*p terms.
    # If we use n*p = 36 terms per j-block, total = m * n * p = 144 (schoolbook).
    # But we can potentially merge across j if we use Strassen-like tricks on the i-k part.
    # For each j, the sub-tensor is T[i, k] with a=i, b=k, c=k*n+i → this is just
    # the identity-like structure on (i,k) with a fixed j.
    # Actually for fixed j: T[i*m+j, j*p+k, k*n+i] = 1 for all i,k.
    # This is a rank-1 tensor in (i,k) space for each j, so total rank = m = 4 per j?
    # No: for fixed j, T is 1 for ALL (i,k), so it's the all-ones matrix of size n x p = 4x9.
    # The all-ones matrix has rank 1! So for each j, the sub-tensor has rank 1.
    # Total rank = m * 1 = 4? That can't be right because the (a,b,c) structure couples them.
    #
    # Let me reconsider. For fixed j:
    #   a = i*m + j  → varies with i
    #   b = j*p + k  → varies with k
    #   c = k*n + i  → varies with both i and k
    # So T[a,b,c] for fixed j is 1 iff a-i is consistent, b-k is consistent, c = k*n+i.
    # The "c" index couples i and k, so it's not simply separable.
    #
    # Actually, let me think about it differently.
    # T[a,b,c] = 1 iff ∃ i,j,k: a=im+j, b=jp+k, c=kn+i
    # Given (a,b,c), i = c % n, j = a % m, k = (c - i) / n (must equal b % p and (b - k)/p = j)
    # So T is fully determined.
    #
    # Key insight: c = k*n + i determines i = c mod n and k = (c - i)/n.
    # Then a = i*m + j requires j = (a - i*m)/m = a - i*m (since i < n ≤ m? No, n=m=4)
    # j = a - i*m must be in [0, m)
    # b = j*p + k requires k = (b - j*p)/p = b - j*p, must be in [0, p)
    # So the tensor is a permutation-like structure.

    # Let's try the straightforward search
    best = None
    best_rank = R0  # 144

    # Try ranks: 36, 40, 44, 48, 52, 56, 60, ..., up to 144
    # But ALS at low rank may not converge to exact solutions.
    # Let's be strategic: try a few key ranks.

    # Extended candidate ranks, including higher ones where convergence is more likely
    # but still potentially better than 144.
    # Step 2 for finer granularity in lower ranges, larger steps in higher.
    candidate_ranks = []
    for r in range(36, 141, 2):
        candidate_ranks.append(r)
    # Ensure we don't waste time on too many ranks if we are slow
    # Prioritize ranges where known algorithms might land (e.g. around 100-120)
    # But we must find the MINIMAL, so we must check lower ones first.
    
    # Filter valid ranks
    candidate_ranks = [r for r in candidate_ranks if r < best_rank and r >= 36]

    for rk in candidate_ranks:
        if time.time() > DL - 15:
            break
        
        # Use more seeds for ranks that are "harder" (lower) or "easier" (higher)?
        # Lower ranks are harder to converge to. Higher ranks (closer to 144) are easier.
        # But we want minimal.
        # Let's use 5 seeds for all, as the check is now faster.
        res = _try_rank(rk, n_seeds=5)
        if res is not None:
            U, V, W = res
            best_rank = rk
            best = (U, V, W)
            break  # Found minimal rank in the sorted list

    # If nothing found below 144, use schoolbook
    if best is not None:
        Uo, Vo, Wo = best
        return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}

    # Fallback: schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
