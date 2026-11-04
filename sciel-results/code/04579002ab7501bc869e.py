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


def _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=None):
    """Alternating least squares with convergence check. Returns (U, V, W, converged)."""
    prev_err = np.inf
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
        
        # Check convergence
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        if err < 1e-10:
            return U, V, W, True
        if abs(prev_err - err) < 1e-12 * max(1.0, prev_err):
            return U, V, W, True
        prev_err = err
    return U, V, W, False


def _try_rank16(deadline):
    """Bounded randomized search for a rank-16 decomposition. Returns (U,V,W) or None."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

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
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=150, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 2: Random initializations with structured bias
    # Use the sparsity pattern of T to guide initialization
    nonzero_mask = T != 0
    nz_indices = np.argwhere(nonzero_mask)
    
    for trial in range(100):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(2000 + trial)
        
        # Initialize with small random values, but bias towards nonzero entries
        U = rng.randn(U_DIM, rank) * 0.5
        V = rng.randn(V_DIM, rank) * 0.5
        W = rng.randn(W_DIM, rank) * 0.5
        
        # Add a structured component: for each nonzero (i,j,k), 
        # ensure some rank-1 term can contribute
        for _ in range(30):
            idx = nz_indices[rng.randint(len(nz_indices))]
            i, j, k = idx
            c = rng.randint(rank)
            U[i, c] += rng.randn()
            V[j, c] += rng.randn()
            W[k, c] += rng.randn()
        
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=200, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 3: More aggressive random search with different scales
    for trial in range(50):
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(5000 + trial)
        scale = 0.1 + 0.1 * trial
        U = rng.randn(U_DIM, rank) * scale
        V = rng.randn(V_DIM, rank) * scale
        W = rng.randn(W_DIM, rank) * scale
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=150, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

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
        r16 = _try_rank16(deadline)
        if r16 is not None and _validate(r16[0], r16[1], r16[2], T, atol=1e-8):
            base = r16
    except Exception:
        pass

    U, V, W = base
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
