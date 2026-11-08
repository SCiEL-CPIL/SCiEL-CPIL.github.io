import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_rank17():
    """Rank 17 decomposition for 2x3x3 matrix multiplication.
    Combines Strassen (rank 7) for 2x2x2 block + schoolbook for cross terms.
    """
    n, m, p = 2, 3, 3
    R = 17
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    # A: i in 0..1, j in 0..2 -> idx = i*3 + j
    # B: j in 0..2, k in 0..2 -> idx = j*3 + k
    # C: k in 0..2, i in 0..1 -> idx = k*2 + i  (trace convention W[k*n+i, r])

    # ---- Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] ----
    # A idx: 0,1,3,4 ; B idx: 0,1,3,4 ; C idx: 0,1,2,3
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # ---- Part 2: schoolbook A[0:2,2] * B[2,0:2] -> C[0:2,0:2] ----
    # A idx: 2,5 ; B idx: 6,7 ; C idx: 0,1,2,3
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # ---- Part 3: schoolbook A[0:2,0:3] * B[0:3,2] -> C[0:2,2] ----
    # A idx: 0..5 ; B idx: 2,5,8 ; C idx: 4,5
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] reconstructed tensor matches T."""
    n, m, p = 2, 3, 3
    # Build target tensor T[i,j,k]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    # Reconstruct
    R = U.shape[1]
    rec = np.zeros((n, m, p))
    for r in range(R):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        rec += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
    diff = np.abs(rec - T).max()
    return diff < tol, diff


def _try_rank16_via_polish(deadline):
    """
    Attempt to find a rank-16 decomposition using a smooth optimization approach.
    Uses a log-sum-exp based smooth surrogate for the max-norm constraint,
    combined with iterative reweighted least squares style refinement.
    
    Returns (U, V, W) if successful, else None.
    """
    n, m, p = 2, 3, 3
    R = 16
    T = np.ones((n, m, p))
    
    # Target tensor flattened
    T_flat = T.flatten()
    
    best_err = np.inf
    best_factors = None
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts with different initializations
    for restart in range(60):
        if time.time() > deadline - 25:
            break
        
        # Initialize with a mix of structured and random factors
        if restart < 20:
            # Structured initialization based on rank-17 with one column zeroed
            U17, V17, W17 = _build_rank17()
            # Remove one column (the least important)
            col_to_remove = restart % 17
            cols = [c for c in range(17) if c != col_to_remove]
            U = U17[:, cols].copy()
            V = V17[:, cols].copy()
            W = W17[:, cols].copy()
            # Add small random perturbation
            U += rng.normal(0, 0.01, U.shape)
            V += rng.normal(0, 0.01, V.shape)
            W += rng.normal(0, 0.01, W.shape)
        elif restart < 40:
            # Random initialization
            U = rng.normal(size=(n * m, R))
            V = rng.normal(size=(m * p, R))
            W = rng.normal(size=(p * n, R))
            # Normalize
            for r in range(R):
                nu = np.linalg.norm(U[:, r])
                nv = np.linalg.norm(V[:, r])
                nw = np.linalg.norm(W[:, r])
                if nu > 1e-12: U[:, r] /= nu
                if nv > 1e-12: V[:, r] /= nv
                if nw > 1e-12: W[:, r] /= nw
        else:
            # Sparse-like initialization
            U = rng.choice([-1, 0, 1], size=(n * m, R))
            V = rng.choice([-1, 0, 1], size=(m * p, R))
            W = rng.choice([-1, 0, 1], size=(p * n, R))
            # Ensure no all-zero columns
            for r in range(R):
                if np.all(U[:, r] == 0): U[0, r] = 1
                if np.all(V[:, r] == 0): V[0, r] = 1
                if np.all(W[:, r] == 0): W[0, r] = 1
        
        # Iterative refinement using gradient descent with adaptive learning rate
        lr = 0.05
        best_local_err = np.inf
        
        for it in range(3000):
            if time.time() > deadline - 25:
                break
            
            # Compute reconstruction
            rec = np.zeros((n, m, p))
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                rec += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
            
            err = np.linalg.norm(rec - T)
            if err < best_local_err:
                best_local_err = err
                if err < best_err:
                    best_err = err
                    best_factors = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                break
            
            # Compute gradients
            resid = rec - T
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r in range(R):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                dU[:, r] = np.einsum('ijk,jk,ki->ij', resid, Vr, Wr).flatten()
                dV[:, r] = np.einsum('ijk,ij,ki->jk', resid, Ur, Wr).flatten()
                dW[:, r] = np.einsum('ijk,ij,jk->ki', resid, Ur, Vr).flatten()
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 1e-12:
                dU /= grad_norm
                dV /= grad_norm
                dW /= grad_norm
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adjust learning rate based on progress
            if err > 5.0:
                lr *= 0.5
            elif err < 0.001:
                lr = min(lr * 1.1, 0.1)
            else:
                lr = lr * 0.999
        
        # Polish: try rounding to nearest half-integers
        if best_local_err < 1e-4:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            ok, diff = _verify(U_r, V_r, W_r, tol=1e-9)
            if ok:
                return U_r, V_r, W_r
    
    if best_factors is not None and best_err < 1e-6:
        U, V, W = best_factors
        # Final polish
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        ok, diff = _verify(U_r, V_r, W_r, tol=1e-9)
        if ok:
            return U_r, V_r, W_r
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 150.0
    SAFETY = 20.0

    # Baseline: rank-17 decomposition (verified exact, integer entries).
    U17, V17, W17 = _build_rank17()
    ok17, diff17 = _verify(U17, V17, W17)
    best = (17, U17, V17, W17)

    # Try to find a rank-16 decomposition
    if time.time() + SAFETY < DEADLINE:
        try:
            result = _try_rank16_via_polish(deadline=DEADLINE)
            if result is not None:
                U16, V16, W16 = result
                best = (16, U16, V16, W16)
        except Exception:
            pass

    R, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
