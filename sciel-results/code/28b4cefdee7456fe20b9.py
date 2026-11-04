"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p)
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]
        T_computed += tmp2
    
    T_expected = np.ones((n, m, p), dtype=np.float64)
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _build_4x4x4_laderman():
    """
    Laderman's rank-19 decomposition for 4x4 * 4x4 = 4x4.
    Returns U (16x19), V (16x19), W (16x19) as numpy arrays.
    """
    n = m = p = 4
    rank = 19
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Laderman's 19 products for 4x4 matrix multiplication
    # Using 1-indexed notation: A[i,j], B[j,k], C[i,k]
    # 0-indexed: A[i,j] -> U[i*m+j], B[j,k] -> V[j*p+k], C[i,k] -> W[k*n+i]
    
    # Define the 19 terms as (U_pattern, V_pattern, W_pattern)
    # Each pattern is a list of (index, coefficient) tuples
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22) -> C11, C22
    terms.append([(idx_u(0,0),1),(idx_u(1,1),1)], [(idx_v(0,0),1),(idx_v(1,1),1)], [(idx_w(0,0),1),(idx_w(1,1),1)])
    # M2 = (A33+A44)(B33+B44) -> C33, C44
    terms.append([(idx_u(2,2),1),(idx_u(3,3),1)], [(idx_v(2,2),1),(idx_v(3,3),1)], [(idx_w(2,2),1),(idx_w(3,3),1)])
    # M3 = (A11+A22+A33+A44)(B11+B22+B33+B44) -> C13,C14,C23,C24,C31,C32,C41,C42 (off-diagonal 2x2 blocks)
    terms.append([(idx_u(0,0),1),(idx_u(1,1),1),(idx_u(2,2),1),(idx_u(3,3),1)],
                 [(idx_v(0,0),1),(idx_v(1,1),1),(idx_v(2,2),1),(idx_v(3,3),1)],
                 [(idx_w(2,0),1),(idx_w(3,0),1),(idx_w(2,1),1),(idx_w(3,1),1),(idx_w(0,2),1),(idx_w(1,2),1),(idx_w(0,3),1),(idx_w(1,3),1)])
    # M4 = (A13+A24)(B13+B24) -> C11,C12,C21,C22
    terms.append([(idx_u(0,2),1),(idx_u(1,3),1)], [(idx_v(0,2),1),(idx_v(1,3),1)], [(idx_w(0,0),1),(idx_w(1,0),1),(idx_w(0,1),1),(idx_w(1,1),1)])
    # M5 = (A31+A42)(B31+B42) -> C33,C34,C43,C44
    terms.append([(idx_u(2,0),1),(idx_u(3,1),1)], [(idx_v(2,0),1),(idx_v(3,1),1)], [(idx_w(2,2),1),(idx_w(3,2),1),(idx_w(2,3),1),(idx_w(3,3),1)])
    # M6 = (A12+A34)(B12+B34) -> C14,C24,C31,C41 (wait, need to check)
    # Actually let me use the standard Laderman construction more carefully.
    
    # Let me use a verified construction. The standard Laderman rank-19 for 4x4:
    # I'll use the formulation from the literature.
    
    # Actually, let me construct it using 2x2 Strassen blocks combined.
    # This is complex. Let me use a simpler approach: just return None and rely on numerical methods.
    
    return None


def _try_lm(rank, n, m, p, max_iter=500, tol=1e-10, deadline=None, n_trials=10):
    """
    Try to find a rank-r decomposition via Levenberg-Marquardt style optimization.
    Uses structured initializations to improve convergence.
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build target tensor
    target = np.zeros((n_m, m_p, p_n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        
        rng = np.random.RandomState(trial * 7919 + rank * 31 + 42)
        
        # Structured initialization: start near schoolbook and perturb
        # Use a mix of random and structured patterns
        U = rng.randn(n_m, rank) * 0.3
        V = rng.randn(m_p, rank) * 0.3
        W = rng.randn(p_n, rank) * 0.3
        
        # Add some structure: make first few columns resemble schoolbook
        for c in range(min(rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            if c < n_m:
                U[i*m+j, c] += 1.0
            if c < m_p:
                V[j*p+k, c] += 1.0
            if c < p_n:
                W[k*n+i, c] += 1.0
        
        lr = 0.01
        lam = 0.01  # LM parameter
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute current tensor estimate
            est = np.einsum('ar,br,cr->abc', U, V, W)
            residual = est - target
            
            err = np.max(np.abs(residual))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            # Gradients
            dU = np.einsum('abc,br,cr->ar', residual, V, W)
            dV = np.einsum('abc,ar,cr->br', residual, U, W)
            dW = np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Hessian approximation (diagonal)
            gU2 = np.einsum('ar,ar->ar', dU, dU)
            gV2 = np.einsum('br,br->br', dV, dV)
            gW2 = np.einsum('cr,cr->cr', dW, dW)
            
            # LM step
            denom_U = gU2 + lam
            denom_V = gV2 + lam
            denom_W = gW2 + lam
            
            step_U = dU / np.maximum(denom_U, 1e-12)
            step_V = dV / np.maximum(denom_V, 1e-12)
            step_W = dW / np.maximum(denom_W, 1e-12)
            
            # Line search
            best_step_lr = lr
            best_step_err = err
            for trial_lr in [lr, lr*0.5, lr*0.25, lr*0.1, lr*0.05]:
                U_test = U - trial_lr * step_U
                V_test = V - trial_lr * step_V
                W_test = W - trial_lr * step_W
                
                est_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
                err_test = np.max(np.abs(est_test - target))
                
                if err_test < best_step_err:
                    best_step_err = err_test
                    best_step_lr = trial_lr
                else:
                    break
            
            U = U - best_step_lr * step_U
            V = V - best_step_lr * step_V
            W = W - best_step_lr * step_W
            
            # Adapt LR
            if best_step_lr < lr * 0.3:
                lr = max(best_step_lr, 1e-6)
            elif best_step_lr > lr * 1.5:
                lr = min(lr * 1.5, 0.1)
            
            # Adapt LM parameter
            if best_step_err < err * 0.9:
                lam = max(lam * 0.5, 1e-8)
            else:
                lam = min(lam * 2.0, 10.0)
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def _try_structured_4x4x5(deadline=None):
    """
    Try to construct a low-rank decomposition for 4x4x5 using structured approaches.
    Strategy: Use rank-20 to rank-25 with careful initialization.
    """
    n, m, p = 4, 4, 5
    
    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
        if deadline is not None and time.time() > deadline:
            break
        
        result = _try_lm(rank, n, m, p, max_iter=300, tol=1e-9, deadline=deadline, n_trials=8)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                return result
    
    return None


def _try_polish(result, n, m, p, deadline=None, max_iter=1000, tol=1e-12):
    """Polish an approximate solution to higher precision."""
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    rank = U.shape[1]
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        est = np.einsum('ar,br,cr->abc', U, V, W)
        residual = est - target
        err = np.max(np.abs(residual))
        
        if err < tol:
            break
        
        dU = np.einsum('abc,br,cr->ar', residual, V, W)
        dV = np.einsum('abc,ar,cr->br', residual, U, W)
        dW = np.einsum('abc,ar,br->cr', residual, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if lr < 1e-8:
            break
        lr *= 0.95
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-10):
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin for 150s limit
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try structured low-rank approaches
    if time.time() < deadline:
        result_struct = _try_structured_4x4x5(deadline)
        if result_struct is not None:
            best_result = result_struct
            best_rank = result_struct.get("rank", best_rank)
    
    # Try to polish the best result
    if time.time() < deadline and best_rank < 80:
        polished = _try_polish(best_result, n, m, p, deadline, max_iter=500, tol=1e-12)
        if polished is not None:
            best_result = polished
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
