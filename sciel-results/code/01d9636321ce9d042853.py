import numpy as np
import time

DIMENSIONS = [2, 3, 3]
DEADLINE_TOTAL = 135.0  # Safety margin before 150s kill

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
    T = np.ones((n, m, p))
    R = U.shape[1]
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    rec = np.einsum('ijr,jkr,pir->ijk', U3, V3, W3)
    diff = np.abs(rec - T).max()
    return diff < tol, diff


def _try_rank16_structured(rng, T, deadline, safety):
    """Try to find rank-16 via structured block-parallel search with vectorized ops."""
    n, m, p = 2, 3, 3
    R_target = 16
    U17, V17, W17 = _build_rank17()
    
    best_err = np.inf
    best_factors = None
    
    for drop_idx in range(17):
        if time.time() + safety >= deadline:
            break
        keep = [i for i in range(17) if i != drop_idx]
        
        U = U17[:, keep].reshape(n, m, R_target).copy()
        V = V17[:, keep].reshape(m, p, R_target).copy()
        W = W17[:, keep].reshape(p, n, R_target).copy()
        
        U += rng.normal(0, 0.05, U.shape)
        V += rng.normal(0, 0.05, V.shape)
        W += rng.normal(0, 0.05, W.shape)
        
        lr = 0.1
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        prev_err = np.inf
        
        for it in range(1500):
            if time.time() + safety >= deadline:
                break
            
            rec = np.einsum('ijr,jkr,pir->ijk', U, V, W)
            err = np.linalg.norm(rec - T)
            
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break
            
            if err > prev_err * 1.5:
                lr *= 0.5
                vU *= 0.3
                vV *= 0.3
                vW *= 0.3
            prev_err = err
            
            resid = rec - T
            dU = np.einsum('ijk,jkr,pkr->ijr', resid, V, W)
            dV = np.einsum('ijk,ijr,pkr->jkr', resid, U, W)
            dW = np.einsum('ijk,ijr,jkr->pir', resid, U, V)
            
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            U += vU
            V += vV
            W += vW
            
            if err < 0.01:
                lr = min(lr * 1.05, 0.15)
            elif err > 1.0:
                lr *= 0.7
        
        if best_err < 1e-8:
            break
    
    return best_err, best_factors


def _try_rank16_random(rng, T, deadline, safety):
    """Try to find rank-16 via random warm-start with vectorized Adam."""
    n, m, p = 2, 3, 3
    R = 16
    best_err = np.inf
    best_factors = None
    
    U17, V17, W17 = _build_rank17()
    
    for restart in range(15):
        if time.time() + safety >= deadline:
            break
        
        drop_idx = rng.integers(0, 17)
        keep = [i for i in range(17) if i != drop_idx]
        
        U = U17[:, keep].reshape(n, m, R).copy()
        V = V17[:, keep].reshape(m, p, R).copy()
        W = W17[:, keep].reshape(p, n, R).copy()
        
        scale = 0.2 + 0.1 * restart
        U += rng.normal(0, scale, U.shape)
        V += rng.normal(0, scale, V.shape)
        W += rng.normal(0, scale, W.shape)
        
        lr = 0.05
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        mU = np.zeros_like(U)
        mV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        t = 0
        
        for it in range(1000):
            if time.time() + safety >= deadline:
                break
            t += 1
            
            rec = np.einsum('ijr,jkr,pir->ijk', U, V, W)
            err = np.linalg.norm(rec - T)
            
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
            if err < 1e-10:
                break
            
            resid = rec - T
            dU = np.einsum('ijk,jkr,pkr->ijr', resid, V, W)
            dV = np.einsum('ijk,ijr,pkr->jkr', resid, U, W)
            dW = np.einsum('ijk,ijr,jkr->pir', resid, U, V)
            
            mU = beta1 * mU + (1 - beta1) * dU
            mV = beta1 * mV + (1 - beta1) * dV
            mW = beta1 * mW + (1 - beta1) * dW
            vU = beta2 * vU + (1 - beta2) * dU**2
            vV = beta2 * vV + (1 - beta2) * dV**2
            vW = beta2 * vW + (1 - beta2) * dW**2
            
            mU_hat = mU / (1 - beta1**t)
            mV_hat = mV / (1 - beta1**t)
            mW_hat = mW / (1 - beta1**t)
            vU_hat = vU / (1 - beta2**t)
            vV_hat = vV / (1 - beta2**t)
            vW_hat = vW / (1 - beta2**t)
            
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if err > 1.0:
                lr *= 0.8
            elif err < 0.01:
                lr = min(lr * 1.02, 0.1)
        
        if best_err < 1e-8:
            break
    
    return best_err, best_factors


def _polish(U, V, W, T, deadline, safety):
    """High-precision polish pass with vectorized gradient descent."""
    n, m, p = 2, 3, 3
    R = U.shape[2]
    
    best_err = np.inf
    best_factors = (U.copy(), V.copy(), W.copy())
    
    # Phase 1: Medium steps
    lr = 0.01
    for it in range(200):
        if time.time() + safety >= deadline:
            break
        
        rec = np.einsum('ijr,jkr,pir->ijk', U, V, W)
        resid = rec - T
        err = np.linalg.norm(resid)
        
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        if err < 1e-12:
            break
        
        dU = np.einsum('ijk,jkr,pkr->ijr', resid, V, W)
        dV = np.einsum('ijk,ijr,pkr->jkr', resid, U, W)
        dW = np.einsum('ijk,ijr,jkr->pir', resid, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    # Phase 2: Fine-tune
    lr = 0.001
    for it in range(1000):
        if time.time() + safety >= deadline:
            break
        rec = np.einsum('ijr,jkr,pir->ijk', U, V, W)
        err = np.linalg.norm(rec - T)
        if err < best_err:
            best_err = err
            best_factors = (U.copy(), V.copy(), W.copy())
        if err < 1e-14:
            break
        
        resid = rec - T
        dU = np.einsum('ijk,jkr,pkr->ijr', resid, V, W)
        dV = np.einsum('ijk,ijr,pkr->jkr', resid, U, W)
        dW = np.einsum('ijk,ijr,jkr->pir', resid, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return best_err, best_factors


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = DEADLINE_TOTAL
    SAFETY = 15.0

    # Baseline: rank-17 decomposition (verified exact, integer entries).
    U17, V17, W17 = _build_rank17()
    ok17, diff17 = _verify(U17, V17, W17)
    best = (17, U17, V17, W17)

    T = np.ones((n, m, p))
    rng = np.random.default_rng(42)

    def _try_round_and_verify(Uf, Vf, Wf, tol=1e-9):
        """Try rounding to integers, then half-integers. Return (ok, U, V, W) or None."""
        Ur_int = np.round(Uf)
        Vr_int = np.round(Vf)
        Wr_int = np.round(Wf)
        ok_int, _ = _verify(Ur_int, Vr_int, Wr_int, tol=tol)
        if ok_int:
            return (Ur_int, Vr_int, Wr_int)
        Ur_half = np.round(2 * Uf) / 2
        Vr_half = np.round(2 * Vf) / 2
        Wr_half = np.round(2 * Wf) / 2
        ok_half, _ = _verify(Ur_half, Vr_half, Wr_half, tol=tol)
        if ok_half:
            return (Ur_half, Vr_half, Wr_half)
        return None

    # Strategy 1: Structured search from rank-17
    if time.time() + SAFETY < DEADLINE:
        try:
            err_struct, factors_struct = _try_rank16_structured(rng, T, DEADLINE, SAFETY)
            if err_struct < 1e-6 and factors_struct is not None:
                U_s, V_s, W_s = factors_struct
                err_pol, factors_pol = _polish(U_s, V_s, W_s, T, DEADLINE, SAFETY)
                if err_pol < 1e-8:
                    U_p, V_p, W_p = factors_pol
                    Uf = U_p.reshape(n * m, 16)
                    Vf = V_p.reshape(m * p, 16)
                    Wf = W_p.reshape(p * n, 16)
                    result = _try_round_and_verify(Uf, Vf, Wf)
                    if result is not None:
                        best = (16, result[0], result[1], result[2])
        except Exception:
            pass

    # Strategy 2: Random warm-start search
    if best[0] == 17 and time.time() + SAFETY < DEADLINE:
        try:
            err_rand, factors_rand = _try_rank16_random(rng, T, DEADLINE, SAFETY)
            if err_rand < 1e-6 and factors_rand is not None:
                U_r, V_r, W_r = factors_rand
                err_pol, factors_pol = _polish(U_r, V_r, W_r, T, DEADLINE, SAFETY)
                if err_pol < 1e-8:
                    U_p, V_p, W_p = factors_pol
                    Uf = U_p.reshape(n * m, 16)
                    Vf = V_p.reshape(m * p, 16)
                    Wf = W_p.reshape(p * n, 16)
                    result = _try_round_and_verify(Uf, Vf, Wf)
                    if result is not None:
                        best = (16, result[0], result[1], result[2])
        except Exception:
            pass

    R, U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
