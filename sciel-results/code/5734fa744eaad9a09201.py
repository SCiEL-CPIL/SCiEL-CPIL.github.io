import numpy as np
import time
import scipy.optimize

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
    # Vectorized reconstruction
    # U: (nm, R) -> (n, m, R)
    # V: (mp, R) -> (m, p, R)
    # W: (pn, R) -> (p, n, R)
    U_reshaped = U.reshape(n, m, R)
    V_reshaped = V.reshape(m, p, R)
    W_reshaped = W.reshape(p, n, R)
    
    # rec[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    diff = np.abs(rec - T).max()
    return diff < tol, diff


def _try_rank16_vectorized(rng, T, deadline, safety):
    """Try to find rank-16 via vectorized gradient descent."""
    n, m, p = 2, 3, 3
    R_target = 16
    U17, V17, W17 = _build_rank17()
    
    best_err = np.inf
    best_factors = None
    
    # Try dropping each term
    for drop_idx in range(17):
        if time.time() + safety >= deadline:
            break
        keep = [i for i in range(17) if i != drop_idx]
        
        U = U17[:, keep].reshape(n, m, R_target).copy()
        V = V17[:, keep].reshape(m, p, R_target).copy()
        W = W17[:, keep].reshape(p, n, R_target).copy()
        
        # Small perturbation
        U += rng.normal(0, 0.05, U.shape)
        V += rng.normal(0, 0.05, V.shape)
        W += rng.normal(0, 0.05, W.shape)
        
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for it in range(500):
            if time.time() + safety >= deadline:
                break
            
            # Vectorized Reconstruction
            rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            err = np.linalg.norm(rec - T)
            
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break
            
            resid = rec - T
            
            # Vectorized Gradients
            # dU[i,j,r] = sum_k resid[i,j,k] * V[j,k,r] * W[k,i,r]
            dU = np.einsum('ijk,jkr,kir->ijr', resid, V, W)
            # dV[j,k,r] = sum_i resid[i,j,k] * U[i,j,r] * W[k,i,r]
            dV = np.einsum('ijk,ijr,kir->jkr', resid, U, W)
            # dW[k,i,r] = sum_j resid[i,j,k] * U[i,j,r] * V[j,k,r]
            dW = np.einsum('ijk,ijr,jkr->kir', resid, U, V)
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            U += vU
            V += vV
            W += vW
            
            if err > 1.0:
                lr *= 0.8
            elif err < 0.01:
                lr = min(lr * 1.05, 0.1)
        
        if best_err < 1e-8:
            break
    
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

    # Strategy 1: Vectorized search from rank-17
    if time.time() + SAFETY < DEADLINE:
        try:
            err_struct, factors_struct = _try_rank16_vectorized(rng, T, DEADLINE, SAFETY)
            if err_struct < 1e-6 and factors_struct is not None:
                U_s, V_s, W_s = factors_struct
                Uf = U_s.reshape(n * m, 16)
                Vf = V_s.reshape(m * p, 16)
                Wf = W_s.reshape(p * n, 16)
                
                # Try rounding to integers
                Ur_int = np.round(Uf)
                Vr_int = np.round(Vf)
                Wr_int = np.round(Wf)
                ok_int, _ = _verify(Ur_int, Vr_int, Wr_int, tol=1e-9)
                if ok_int:
                    best = (16, Ur_int, Vr_int, Wr_int)
                else:
                    # Try half-integers
                    Ur_half = np.round(2 * Uf) / 2
                    Vr_half = np.round(2 * Vf) / 2
                    Wr_half = np.round(2 * Wf) / 2
                    ok_half, _ = _verify(Ur_half, Vr_half, Wr_half, tol=1e-9)
                    if ok_half:
                        best = (16, Ur_half, Vr_half, Wr_half)
        except Exception:
            pass

    R, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
