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


def _greedy_rank1_completion(T, U, V, W, target_rank, deadline=None):
    """Greedily add rank-1 terms to minimize residual norm.
    Starts from current (U,V,W) with fewer columns, adds rank-1 terms
    via truncated SVD on mode-unfoldings of the residual."""
    current_rank = U.shape[1]
    if current_rank >= target_rank:
        return U, V, W
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for new_col in range(current_rank, target_rank):
        if deadline is not None and time.time() > deadline:
            return U, V, W
        
        # Compute residual tensor
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        residual = T - recon
        
        # Find best rank-1 approximation via mode-1 SVD
        # Mode-1 unfolding: residual reshaped to (U_DIM, V_DIM*W_DIM)
        R1 = residual.reshape(U_DIM, V_DIM * W_DIM)
        U_s, S_s, Vt_s = np.linalg.svd(R1, full_matrices=False)
        
        # Best rank-1: first left singular vector and first right singular vector
        u_vec = U_s[:, 0]
        vw_mat = Vt_s[0, :].reshape(V_DIM, W_DIM)
        
        # Now decompose vw_mat into best rank-1: use SVD again
        V_s, S_s2, Wt_s = np.linalg.svd(vw_mat, full_matrices=False)
        v_vec = V_s[:, 0]
        w_vec = Wt_s[0, :]
        
        # Scale: the rank-1 term is u_vec ⊗ v_vec ⊗ w_vec
        # We need to match the singular value
        scale = S_s[0] * S_s2[0] / (np.linalg.norm(u_vec) * np.linalg.norm(v_vec) * np.linalg.norm(w_vec)) if (np.linalg.norm(u_vec) * np.linalg.norm(v_vec) * np.linalg.norm(w_vec)) > 1e-15 else 1.0
        
        # Actually, let's just use the vectors directly and let ALS refine
        U = np.column_stack([U, u_vec])
        V = np.column_stack([V, v_vec])
        W = np.column_stack([W, w_vec])
    
    return U, V, W


def _greedy_search(T, target_rank, deadline, n_restarts=30):
    """Random restarts: start with random rank-1 terms, greedily complete."""
    best_result = None
    best_err = float('inf')
    
    nz_indices = np.argwhere(T != 0)
    
    for trial in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
        rng = np.random.RandomState(42 + trial * 7919)
        
        # Initialize with a few random rank-1 terms
        init_rank = min(3, target_rank)
        U = np.zeros((U_DIM, target_rank))
        V = np.zeros((V_DIM, target_rank))
        W = np.zeros((W_DIM, target_rank))
        
        for c in range(init_rank):
            # Pick a random nonzero entry to seed
            i, j, k = nz_indices[rng.randint(len(nz_indices))]
            U[i, c] = rng.randn()
            V[j, c] = rng.randn()
            W[k, c] = rng.randn()
            # Also add some spread
            for _ in range(5):
                ii = rng.randint(U_DIM)
                jj = rng.randint(V_DIM)
                kk = rng.randint(W_DIM)
                U[ii, c] += rng.randn() * 0.3
                V[jj, c] += rng.randn() * 0.3
                W[kk, c] += rng.randn() * 0.3
        
        # Greedily complete to target_rank
        U, V, W = _greedy_rank1_completion(T, U, V, W, target_rank, deadline)
        
        # Refine with a few ALS iterations
        T0 = T.reshape(U_DIM, -1)
        T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
        T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, target_rank, max_iter=50, deadline=deadline)
        
        # Check error
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(recon - T)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-8:
            break
    
    return best_result


def _try_rank16(deadline):
    """Bounded search for a rank-16 decomposition using greedy rank-1 completion.
    Returns (U,V,W) or None."""
    T = _build_tensor()
    T0 = T.reshape(U_DIM, -1)
    T1 = T.transpose(1, 0, 2).reshape(V_DIM, -1)
    T2 = T.transpose(2, 0, 1).reshape(W_DIM, -1)
    rank = 16

    # Strategy 1: Start from rank-17, drop one column, refine with ALS
    U17, V17, W17 = _rank17()
    for drop_col in range(17):
        if time.time() > deadline:
            break
        U = U17[:, :drop_col].copy()
        U = np.hstack([U, U17[:, drop_col+1:]]) if drop_col < 16 else U17[:, :16].copy()
        V = V17[:, :drop_col].copy()
        V = np.hstack([V, V17[:, drop_col+1:]]) if drop_col < 16 else V17[:, :16].copy()
        W = W17[:, :drop_col].copy()
        W = np.hstack([W, W17[:, drop_col+1:]]) if drop_col < 16 else W17[:, :16].copy()
        U, V, W, _ = _als_step(U, V, W, T, T0, T1, T2, rank, max_iter=80, deadline=deadline)
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 2: Greedy rank-1 completion from random seeds
    result = _greedy_search(T, rank, deadline, n_restarts=25)
    if result is not None:
        U, V, W = result
        if _validate(U, V, W, T, atol=1e-8):
            return U, V, W

    # Strategy 3: Greedy from rank-17 subsets (drop 2 columns, greedily fill)
    for drop1 in range(17):
        if time.time() > deadline:
            break
        for drop2 in range(drop1 + 1, 17):
            if time.time() > deadline:
                break
            # Keep all columns except drop1 and drop2
            keep = [c for c in range(17) if c not in (drop1, drop2)]
            U_sub = U17[:, keep].copy()
            V_sub = V17[:, keep].copy()
            W_sub = W17[:, keep].copy()
            # Greedily add 1 rank-1 term
            U_g, V_g, W_g = _greedy_rank1_completion(T, U_sub, V_sub, W_sub, rank, deadline)
            # Refine
            U_g, V_g, W_g, _ = _als_step(U_g, V_g, W_g, T, T0, T1, T2, rank, max_iter=40, deadline=deadline)
            if _validate(U_g, V_g, W_g, T, atol=1e-8):
                return U_g, V_g, W_g

    return None


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill

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
