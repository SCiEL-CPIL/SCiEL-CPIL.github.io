"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _try_rank(target_rank, n, m, p, T_flat, deadline):
    """Try to find a valid decomposition of given rank using ALS with quantization."""
    if time.time() > deadline - 15:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Try multiple random initializations
    for init in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random integer/half-integer initialization
        U = np.random.randint(-1, 2, size=(nm, target_rank))
        V = np.random.randint(-1, 2, size=(mp, target_rank))
        
        # Build design matrix B_flat: (rank, nm*mp)
        # B_flat[r] = outer(U[:,r], V[:,r]).flatten()
        B_flat = np.zeros((target_rank, nm * mp))
        for r in range(target_rank):
            B_flat[r] = np.outer(U[:, r], V[:, r]).flatten()
        
        # Solve for W: T_flat = W @ B_flat.T
        # i.e., B_flat.T @ W.T = T_flat.T
        try:
            W_sol, residuals, rank_sv, sv = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=1e-12)
        except:
            continue
        
        W_sol = W_sol.T  # shape (pn, target_rank)
        
        # Quantize to half-integers
        W_q = np.round(W_sol * 2) / 2.0
        
        # Check if quantized solution works
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W_q.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p, target_rank):
            return U_list, V_list, W_list
        
        # Try integer quantization too
        W_q_int = np.round(W_sol)
        W_list_int = W_q_int.tolist()
        if _verify(U_list, V_list, W_list_int, n, m, p, target_rank):
            return U_list, V_list, W_list_int
        
        # Try a second ALS iteration: fix W, solve for U and V
        # This is more complex; skip for now due to time constraints
    
    return None

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # Try ranks from 100 down to 85
    for target_rank in range(100, 84, -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank(target_rank, n, m, p, T_flat, deadline)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            break
    
    # If we found rank 100, try to go lower
    if best_rank > 90:
        for target_rank in range(min(best_rank - 1, 95), 84, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank(target_rank, n, m, p, T_flat, deadline)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result
# EVOLVE-BLOCK-END
