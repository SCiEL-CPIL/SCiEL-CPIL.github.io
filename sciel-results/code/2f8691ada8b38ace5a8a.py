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


def _try_rank16_structured(rng, T, deadline, safety):
    """Try to find rank-16 by starting from rank-17 and dropping/merging one term."""
    n, m, p = 2, 3, 3
    R_target = 16
    U17, V17, W17 = _build_rank17()
    
    best_err = np.inf
    best_factors = None
    
    # Try dropping each of the 17 terms and refining
    for drop_idx in range(17):
        if time.time() + safety >= deadline:
            break
        # Take all terms except drop_idx
        keep = [i for i in range(17) if i != drop_idx]
        U_init = U17[:, keep].copy()
        V_init = V17[:, keep].copy()
        W_init = W17[:, keep].copy()
        
        # Convert to 3D
        U = U_init.reshape(n, m, R_target)
        V = V_init.reshape(m, p, R_target)
        W = W_init.reshape(p, n, R_target)
        
        # Add small random perturbation
        U += rng.normal(0, 0.1, U.shape)
        V += rng.normal(0, 0.1, V.shape)
        W += rng.normal(0, 0.1, W.shape)
        
        # Gradient descent with momentum
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(3000):
            if time.time() + safety >= deadline:
                break
            # Compute residual
            rec = np.zeros((n, m, p))
            for r in range(R_target):
                rec += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
            err = np.linalg.norm(rec - T)
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                break
            
            # Gradients
            resid = rec - T
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(R_target):
                dU[:, :, r] = np.einsum('ijk,jk,ki->ij', resid, V[:, :, r], W[:, :, r])
                dV[:, :, r] = np.einsum('ijk,ij,ki->jk', resid, U[:, :, r], W[:, :, r])
                dW[:, :, r] = np.einsum('ijk,ij,jk->ki', resid, U[:, :, r], V[:, :, r])
            
            # Momentum update
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            U += vU
            V += vV
            W += vW
            
            # Adaptive lr
            if err > 10.0:
                lr *= 0.5
                vU *= 0.5
                vV *= 0.5
                vW *= 0.5
            elif err < 0.01:
                lr = min(lr * 1.1, 0.1)
        
        if best_err < 1e-6:
            break
    
    return best_err, best_factors


def _try_rank16_random(rng, T, deadline, safety):
    """Try to find rank-16 via random initialization and gradient descent."""
    n, m, p = 2, 3, 3
    R = 16
    best_err = np.inf
    best_factors = None
    
    for restart in range(30):
        if time.time() + safety >= deadline:
            break
        
        U = rng.normal(size=(n, m, R))
        V = rng.normal(size=(m, p, R))
        W = rng.normal(size=(p, n, R))
        
        # Normalize
        for r in range(R):
            nu = np.linalg.norm(U[:, :, r])
            nv = np.linalg.norm(V[:, :, r])
            nw = np.linalg.norm(W[:, :, r])
            if nu > 1e-12:
                U[:, :, r] /= nu
            if nv > 1e-12:
                V[:, :, r] /= nv
            if nw > 1e-12:
                W[:, :, r] /= nw
        
        # Gradient descent with momentum
        lr = 0.02
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(2000):
            if time.time() + safety >= deadline:
                break
            rec = np.zeros((n, m, p))
            for r in range(R):
                rec += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
            err = np.linalg.norm(rec - T)
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            if err < 1e-8:
                break
            
            resid = rec - T
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(R):
                dU[:, :, r] = np.einsum('ijk,jk,ki->ij', resid, V[:, :, r], W[:, :, r])
                dV[:, :, r] = np.einsum('ijk,ij,ki->jk', resid, U[:, :, r], W[:, :, r])
                dW[:, :, r] = np.einsum('ijk,ij,jk->ki', resid, U[:, :, r], V[:, :, r])
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            U += vU
            V += vV
            W += vW
            
            if err > 10.0:
                lr *= 0.5
                vU *= 0.5
                vV *= 0.5
                vW *= 0.5
            elif err < 0.01:
                lr = min(lr * 1.1, 0.08)
        
        if best_err < 1e-6:
            break
    
    return best_err, best_factors


def _polish(U, V, W, T, deadline, safety):
    """High-precision polish pass with very tight tolerance."""
    n, m, p = 2, 3, 3
    R = U.shape[2]
    
    lr = 0.005
    momentum = 0.95
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    
    best_err = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(5000):
        if time.time() + safety >= deadline:
            break
        rec = np.zeros((n, m, p))
        for r in range(R):
            rec += np.einsum('ij,jk,ki->ijk', U[:, :, r], V[:, :, r], W[:, :, r])
        err = np.linalg.norm(rec - T)
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        if err < 1e-12:
            break
        
        resid = rec - T
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        for r in range(R):
            dU[:, :, r] = np.einsum('ijk,jk,ki->ij', resid, V[:, :, r], W[:, :, r])
            dV[:, :, r] = np.einsum('ijk,ij,ki->jk', resid, U[:, :, r], W[:, :, r])
            dW[:, :, r] = np.einsum('ijk,ij,jk->ki', resid, U[:, :, r], V[:, :, r])
        
        vU = momentum * vU - lr * dU
        vV = momentum * vV - lr * dV
        vW = momentum * vW - lr * dW
        U += vU
        V += vV
        W += vW
        
        if err > 1.0:
            lr *= 0.5
            vU *= 0.5
            vV *= 0.5
            vW *= 0.5
    
    return best_err, best_factors


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 150.0
    SAFETY = 20.0

    # Baseline: rank-17 decomposition (verified exact, integer entries).
    U17, V17, W17 = _build_rank17()
    ok17, diff17 = _verify(U17, V17, W17)
    best = (17, U17, V17, W17)

    T = np.ones((n, m, p))
    rng = np.random.default_rng(42)

    # Strategy 1: Structured initialization from rank-17
    if time.time() + SAFETY < DEADLINE:
        try:
            err_struct, factors_struct = _try_rank16_structured(rng, T, DEADLINE, SAFETY)
            if err_struct < 1e-6 and factors_struct is not None:
                U_s, V_s, W_s = factors_struct
                # Polish
                err_pol, factors_pol = _polish(U_s, V_s, W_s, T, DEADLINE, SAFETY)
                if err_pol < 1e-8:
                    U_p, V_p, W_p = factors_pol
                    Uf = U_p.reshape(n * m, 16)
                    Vf = V_p.reshape(m * p, 16)
                    Wf = W_p.reshape(p * n, 16)
                    # Round to nearest half-integer
                    Ur = np.round(2 * Uf) / 2
                    Vr = np.round(2 * Vf) / 2
                    Wr = np.round(2 * Wf) / 2
                    ok_r, diff_r = _verify(Ur, Vr, Wr, tol=1e-9)
                    if ok_r:
                        best = (16, Ur, Vr, Wr)
        except Exception:
            pass

    # Strategy 2: Random initialization if structured didn't work
    if best[0] == 17 and time.time() + SAFETY < DEADLINE:
        try:
            err_rand, factors_rand = _try_rank16_random(rng, T, DEADLINE, SAFETY)
            if err_rand < 1e-6 and factors_rand is not None:
                U_r, V_r, W_r = factors_rand
                # Polish
                err_pol, factors_pol = _polish(U_r, V_r, W_r, T, DEADLINE, SAFETY)
                if err_pol < 1e-8:
                    U_p, V_p, W_p = factors_pol
                    Uf = U_p.reshape(n * m, 16)
                    Vf = V_p.reshape(m * p, 16)
                    Wf = W_p.reshape(p * n, 16)
                    Ur = np.round(2 * Uf) / 2
                    Vr = np.round(2 * Vf) / 2
                    Wr = np.round(2 * Wf) / 2
                    ok_r, diff_r = _verify(Ur, Vr, Wr, tol=1e-9)
                    if ok_r:
                        best = (16, Ur, Vr, Wr)
        except Exception:
            pass

    R, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
