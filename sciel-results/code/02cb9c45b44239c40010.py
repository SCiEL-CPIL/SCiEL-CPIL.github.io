import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
# A blocks indexed 0..3: 00, 01, 10, 11
# B blocks indexed 0..3: 00, 01, 10, 11
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def _build_rank84():
    """
    Build rank-84 decomposition by splitting 4x4*4x6 into three 4x4*2 subproblems.
    Each subproblem is solved by 4 independent Strassen 2x2*2x2 products (rank 7 each).
    Total: 3 * 4 * 7 = 84.
    """
    n, m, p = 4, 4, 6
    R = 84
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    # Split B into three 4x2 blocks: columns 0-1, 2-3, 4-5
    for block in range(3):
        for a in range(2):  # A row block
            for prod_idx in range(2):  # 0 for X1*Y1, 1 for X2*Y2
                if prod_idx == 0:
                    x_col_offset = 0
                    y_row_offset = 0
                else:
                    x_col_offset = 2
                    y_row_offset = 2
                
                for s in range(7):
                    for ai in range(4):
                        if alpha[s][ai] != 0:
                            a_row, a_col = ai // 2, ai % 2
                            u_idx = (a * 2 + a_row) * 4 + (x_col_offset + a_col)
                            U[u_idx, col] += alpha[s][ai]
                    
                    for bi in range(4):
                        if beta[s][bi] != 0:
                            b_row, b_col = bi // 2, bi % 2
                            v_idx = (y_row_offset + b_row) * 6 + (2 * block + b_col)
                            V[v_idx, col] += beta[s][bi]
                    
                    for ca, cb, gamma in recon[s]:
                        w_idx = (2 * block + cb) * 4 + (a * 2 + ca)
                        W[w_idx, col] += gamma
                    
                    col += 1
    
    return U, V, W


def _build_schoolbook():
    n, m, p = 4, 4, 6
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    _t0 = time.time()
    deadline = _t0 + 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    def check(U, V, W, atol=1e-9):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)
    
    def err_of(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T - recon) ** 2)
    
    # Start with rank-84 construction
    U84, V84, W84 = _build_rank84()
    R84 = 84
    
    if not check(U84, V84, W84, atol=1e-6):
        U_sb, V_sb, W_sb = _build_schoolbook()
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    best_U, best_V, best_W = U84.copy(), V84.copy(), W84.copy()
    best_rank = R84
    
    # Try to reduce rank by checking if any column can be removed
    # and the remaining columns can be adjusted to still reconstruct T
    # Use a structured approach: for each column to remove, solve for
    # corrections to the remaining columns
    
    def try_remove_column(U, V, W, col_to_remove, target_rank):
        """Try to remove col_to_remove and adjust remaining columns."""
        if time.time() > deadline - 30:
            return None, None, None, float('inf')
        
        # Create matrices without the column to remove
        mask = [i for i in range(target_rank) if i != col_to_remove]
        U_rem = U[:, mask].copy()
        V_rem = V[:, mask].copy()
        W_rem = W[:, mask].copy()
        
        # Current reconstruction without this column
        recon_rem = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
        residual = T - recon_rem
        
        # We want to find small adjustments dU, dV, dW to the remaining columns
        # such that sum_r dU[:,r] * dV[:,r] * dW[:,r] ≈ residual
        # This is still nonlinear, but we can try a first-order approximation
        # or use a simple gradient-based approach
        
        # Try a simple approach: scale up the remaining columns slightly
        # or use a random search with structured perturbations
        
        best_err = err_of(U_rem, V_rem, W_rem)
        if best_err < 1e-10:
            return U_rem, V_rem, W_rem, best_err
        
        # Try adjusting each remaining column
        np.random.seed(col_to_remove)
        for it in range(20):
            if time.time() > deadline - 25:
                break
            if best_err < 1e-10:
                break
            improved = False
            for c in range(len(mask)):
                if time.time() > deadline - 25:
                    break
                Uo, Vo, Wo = U_rem[:, c].copy(), V_rem[:, c].copy(), W_rem[:, c].copy()
                for scale in [0.05, 0.02, 0.01]:
                    U_rem[:, c] = Uo + np.random.randn(n*m) * scale
                    V_rem[:, c] = Vo + np.random.randn(m*p) * scale
                    W_rem[:, c] = Wo + np.random.randn(p*n) * scale
                    new_err = err_of(U_rem, V_rem, W_rem)
                    if new_err < best_err:
                        best_err = new_err
                        improved = True
                        break
                    U_rem[:, c] = Uo
                    V_rem[:, c] = Vo
                    W_rem[:, c] = Wo
                if not improved:
                    break
            if not improved:
                break
        
        if best_err < 1e-10:
            return U_rem, V_rem, W_rem, best_err
        return None, None, None, best_err
    
    # Try removing columns one at a time from rank 84 down
    for target_rank in range(R84 - 1, 70, -1):
        if time.time() > deadline - 40:
            break
        
        U = best_U[:, :target_rank].copy()
        V = best_V[:, :target_rank].copy()
        W = best_W[:, :target_rank].copy()
        
        err = err_of(U, V, W)
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            continue
        
        # Try removing each column
        found = False
        for c in range(target_rank):
            if time.time() > deadline - 35:
                break
            U_new, V_new, W_new, new_err = try_remove_column(U, V, W, c, target_rank)
            if U_new is not None and new_err < 1e-10:
                best_rank = target_rank - 1
                best_U, best_V, best_W = U_new, V_new, W_new
                found = True
                break
        
        if not found:
            # Try a global coordinate descent on the current target_rank
            np.random.seed(target_rank * 1000)
            for it in range(30):
                if time.time() > deadline - 30:
                    break
                if err < 1e-10:
                    break
                improved = False
                for c in range(target_rank):
                    if time.time() > deadline - 30:
                        break
                    Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                    for scale in [0.1, 0.05, 0.01]:
                        U[:, c] = Uo + np.random.randn(n*m) * scale
                        V[:, c] = Vo + np.random.randn(m*p) * scale
                        W[:, c] = Wo + np.random.randn(p*n) * scale
                        new_err = err_of(U, V, W)
                        if new_err < err:
                            err = new_err
                            improved = True
                            break
                        U[:, c] = Uo
                        V[:, c] = Vo
                        W[:, c] = Wo
                    if not improved:
                        break
                if not improved:
                    break
            
            if err < 1e-10:
                best_rank = target_rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Final verification
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    new_row.append(int(round(x)))
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result
    
    if not check(best_U, best_V, best_W, atol=1e-6):
        U_sb, V_sb, W_sb = _build_schoolbook()
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
