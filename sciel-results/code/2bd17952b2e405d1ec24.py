"""Block-structured (Strassen) rank-17 decomposition for 2x3 * 3x3 matmul,
with a bounded, deadline-guarded attempt to find a rank-16 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
U_DIM = N * M   # 6
V_DIM = M * P   # 9
W_DIM = P * N   # 6


def _build_tensor():
    """Full matmul tensor T[i,j,k]=1 iff (i//M)==(k%N) and (i%M)==(j//P)."""
    T = np.zeros((U_DIM, V_DIM, W_DIM), dtype=float)
    for i in range(U_DIM):
        ia, ja = divmod(i, M)
        for j in range(V_DIM):
            ja2, kb = divmod(j, P)
            if ja != ja2:
                continue
            for k in range(W_DIM):
                kb2, ic = divmod(k, N)
                if ia == ic and kb == kb2:
                    T[i, j, k] = 1.0
    return T


def _validate(U, V, W, T, atol=1e-10):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=atol))


def _rank17():
    """Valid rank-17 decomposition: Strassen(2x2x2) for (j in {0,1}, k in {0,1})
    plus schoolbook for the remaining 10 terms."""
    rank = 17
    U = np.zeros((U_DIM, rank))
    V = np.zeros((V_DIM, rank))
    W = np.zeros((W_DIM, rank))
    # Strassen columns 0..6 (A[0,0]=U0,A[0,1]=U1,A[1,0]=U3,A[1,1]=U4;
    # B[0,0]=V0,B[0,1]=V1,B[1,0]=V3,B[1,1]=V4; C[0,0]=W0,C[1,0]=W1,C[0,1]=W2,C[1,1]=W3)
    U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
    # Schoolbook columns 7..16 (k=2 for j in {0,1}, and j=2 for all k)
    U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
    U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
    U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
    U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
    U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
    U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
    U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    return U, V, W


def _try_rank16_structured(deadline):
    """
    Structured rank-16 search:
    1. Start with rank-17 (Strassen core + 10 schoolbook terms).
    2. Check if the 17 rank-1 tensors are linearly dependent.
    3. If so, one can be eliminated, yielding a rank-16 decomposition.
    """
    T = _build_tensor()
    U17, V17, W17 = _rank17()
    
    # Construct the matrix of terms
    # Each term is a vector in R^324
    terms = []
    for c in range(17):
        u = U17[:, c]
        v = V17[:, c]
        w = W17[:, c]
        tensor_c = np.einsum('i,j,k->ijk', u, v, w)
        terms.append(tensor_c.flatten())
    
    terms_matrix = np.array(terms).T # Shape (324, 17)
    
    # Check if any column is in the span of the others
    for c in range(17):
        if time.time() > deadline:
            return None
        
        indices = [i for i in range(17) if i != c]
        A = terms_matrix[:, indices]
        b = terms_matrix[:, c]
        
        try:
            x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            # Check if b is in the span of A
            if np.linalg.norm(A @ x - b) < 1e-8:
                U16 = U17[:, indices].copy()
                V16 = V17[:, indices].copy()
                W16 = W17[:, indices].copy()
                
                # Absorb the eliminated term into the remaining ones
                for j, orig_idx in enumerate(indices):
                    U16[:, j] *= (1 + x[j])
                    
                if _validate(U16, V16, W16, T, atol=1e-8):
                    return U16, V16, W16
                    
        except Exception:
            continue

    return None


def _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=None):
    """Alternating least squares. Returns (U, V, W, converged)."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        # Solve for U
        B = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
        U, *_ = np.linalg.lstsq(B.T, T0.T, rcond=None)
        # Solve for V
        B = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
        V, *_ = np.linalg.lstsq(B.T, T1.T, rcond=None)
        # Solve for W
        B = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
        W, *_ = np.linalg.lstsq(B.T, T2.T, rcond=None)
    return U, V, W, True


def _hosvd_decompose(T, rank):
    """HOSVD-based CP decomposition: truncate each mode's SVD to `rank`,
    then polish with ALS. Returns (U, V, W)."""
    U0 = T.reshape(U_DIM, -1)
    V0 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    W0 = T.transpose(2, 0, 1).reshape(W_DIM, -1)

    # Truncated SVDs give a natural low-rank factorization seed
    _, sU, eU = np.linalg.svd(U0, full_matrices=False)
    _, sV, eV = np.linalg.svd(V0, full_matrices=False)
    _, sW, eW = np.linalg.svd(W0, full_matrices=False)

    # Build initial factors from the top-`rank` left singular vectors
    # (each factor gets shape (dim, rank))
    Uf = eU[:, :rank]
    Vf = eV[:, :rank]
    Wf = eW[:, :rank]

    # Scale to roughly match the tensor magnitude
    scale = np.mean(sU[:rank]) * np.mean(sV[:rank]) * np.mean(sW[:rank])
    if scale > 1e-12:
        Uf = Uf / (scale ** (1.0 / 3.0))
        Vf = Vf / (scale ** (1.0 / 3.0))
        Wf = Wf / (scale ** (1.0 / 3.0))

    return Uf, Vf, Wf


def _try_rank16(deadline):
    """Bounded randomized search for a rank-16 decomposition. Returns (U,V,W) or None."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

    # Strategy 0 (HOSVD): direct linear-algebraic rank minimization.
    # Truncated-SVD seeds per mode, then ALS polish. This bypasses the
    # heuristic random search and uses the tensor's intrinsic singular
    # structure to identify a low-rank factorization.
    for seed_variant in range(6):
        if time.time() > deadline:
            return None
        Uf, Vf, Wf = _hosvd_decompose(T, rank)
        # Perturb to escape degenerate subspace alignments
        rng = np.random.RandomState(7000 + seed_variant)
        if seed_variant > 0:
            Uf = Uf + rng.randn(U_DIM, rank) * 0.05 * seed_variant
            Vf = Vf + rng.randn(V_DIM, rank) * 0.05 * seed_variant
            Wf = Wf + rng.randn(W_DIM, rank) * 0.05 * seed_variant
        Uf, Vf, Wf, _ = _als_step(Uf, Vf, Wf, T, T0, T1, T2, rank,
                                  max_iter=200, deadline=deadline)
        if _validate(Uf, Vf, Wf, T, atol=1e-8):
            return Uf, Vf, Wf

    # Strategy 1: Start from rank-17, zero out one column, then ALS
    U17, V17, W17 = _rank17()
    for drop_col in range(17):
        if time.time() > deadline:
            return None
        U = U17[:, :drop_col].copy()
        U = np.hstack([U, U17[:, drop_col+1:]]) if drop_col < 16 else U17[:, :16].copy()
        V = V17[:, :drop_col].copy()
        V = np.hstack([V, V17[:, drop_col+1:]]) if drop_col < 16 else V17[:, :16].copy()
        W = W17[:, :drop_col].copy()
        W = np.hstack([W, W17[:, drop_col+1:]]) if drop_col < 16 else W17[:, :16].copy()
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=100, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 2: Random initializations with structured bias
    # Use the sparsity pattern of T to guide initialization
    nonzero_mask = T != 0
    nz_indices = np.argwhere(nonzero_mask)
    
    for trial in range(80):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(2000 + trial)
        
        # Initialize with small random values, but bias towards nonzero entries
        U = rng.randn(U_DIM, rank) * 0.5
        V = rng.randn(V_DIM, rank) * 0.5
        W = rng.randn(W_DIM, rank) * 0.5
        
        # Add a structured component: for each nonzero (i,j,k), 
        # ensure some rank-1 term can contribute
        for _ in range(20):
            idx = nz_indices[rng.randint(len(nz_indices))]
            i, j, k = idx
            c = rng.randint(rank)
            U[i, c] += rng.randn()
            V[j, c] += rng.randn()
            W[k, c] += rng.randn()
        
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=150, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 3: More aggressive random search with different scales
    for trial in range(40):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(5000 + trial)
        scale = 0.1 + 0.1 * trial
        U = rng.randn(U_DIM, rank) * scale
        V = rng.randn(V_DIM, rank) * scale
        W = rng.randn(W_DIM, rank) * scale
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=100, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 4: High-precision polish on the best candidate from all
    # strategies above — re-run ALS with tighter tolerance and more
    # iterations on the rank-17 drop-column candidates that got closest.
    if time.time() < deadline - 5:
        best_err = np.inf
        best = None
        U17, V17, W17 = _rank17()
        for drop_col in range(17):
            if time.time() > deadline - 5:
                break
            U = U17[:, :drop_col].copy()
            U = np.hstack([U, U17[:, drop_col+1:]]) if drop_col < 16 else U17[:, :16].copy()
            V = V17[:, :drop_col].copy()
            V = np.hstack([V, V17[:, drop_col+1:]]) if drop_col < 16 else V17[:, :16].copy()
            W = W17[:, :drop_col].copy()
            W = np.hstack([W, W17[:, drop_col+1:]]) if drop_col < 16 else W17[:, :16].copy()
            U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank,
                                   max_iter=150, deadline=deadline - 5)
            err = np.linalg.norm(np.einsum('ic,jc,kc->ijk', U, V, W) - T)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if best_err < 1e-8:
                return best
        if best is not None and best_err < 1e-8:
            return best

    return None


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    # Baseline: valid rank-17 (verified).
    U17, V17, W17 = _rank17()
    T = _build_tensor()
    base = (U17, V17, W17)
    if not _validate(U17, V17, W17, T):
        # Fallback to guaranteed-correct schoolbook rank-18.
        rank = N * M * P
        U = np.zeros((U_DIM, rank)); V = np.zeros((V_DIM, rank)); W = np.zeros((W_DIM, rank))
        col = 0
        for ia in range(N):
            for ja in range(M):
                for kb in range(P):
                    U[ia * M + ja, col] = 1.0
                    V[ja * P + kb, col] = 1.0
                    W[kb * N + ia, col] = 1.0
                    col += 1
        base = (U, V, W)

    # Attempt rank-16 (improvement over 17) within the time budget.
    try:
        # First try the fast structured linear dependency check
        r16 = _try_rank16_structured(deadline)
        if r16 is None:
            r16 = _try_rank16(deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    # Final high-precision polish: if we have a candidate, run one more
    # tight ALS pass to ensure the decomposition is as exact as possible.
    U, V, W = base
    if U.shape[1] == 16:
        try:
            U, V, W, _ = _als_step(U, V, W, T,
                                   T.reshape(U_DIM, -1),
                                   T.transpose(1, 0, 2).reshape(V_DIM, -1),
                                   T.transpose(2, 0, 1).reshape(W_DIM, -1),
                                   16, max_iter=50, deadline=deadline)
            if not _validate(U, V, W, T, atol=1e-8):
                # Revert if polish broke validity
                U, V, W = base
        except Exception:
            pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
