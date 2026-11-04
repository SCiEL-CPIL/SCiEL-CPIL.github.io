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
        # B block columns: 2*block, 2*block+1
        # For each 4x4*2 subproblem:
        # A is 4x4, B_block is 4x2
        # Split A into 2x2 blocks, B_block into 2x2 blocks (each 2x2 since B_block is 4x2)
        # Wait, B_block is 4x2, so it's 2 blocks of 2x2 vertically
        # A is 4x4, so 4 blocks of 2x2
        # C_block is 4x2, so 2 blocks of 2x2
        
        # For C_ab (2x2) where a in {0,1} (A row block), b in {0,1} (B col block, but B is 4x2 so only b=0,1 in the 2x2 sense)
        # Actually B_block is 4x2. Let's index B_block rows as 0,1,2,3 and cols as 0,1.
        # B_block can be seen as two 2x2 blocks: B_block[0:2, 0:2] and B_block[2:4, 0:2]
        # A is four 2x2 blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
        # C_block is two 2x2 blocks: C_block[0:2, 0:2] and C_block[2:4, 0:2]
        
        # C_block[a, 0:2] = A[a,0:2] * B_block[0:2, 0:2] + A[a,2:4] * B_block[2:4, 0:2]
        # where a is 0 or 1 (row block of A)
        
        for a in range(2):  # A row block
            # For each a, we compute C_block[a*2:(a+1)*2, 0:2]
            # This is a 2x2 matrix
            # It equals A[a,0:2] * B_block[0:2,0:2] + A[a,2:4] * B_block[2:4,0:2]
            # Let X1 = A[a,0:2] (2x2), X2 = A[a,2:4] (2x2)
            # Let Y1 = B_block[0:2,0:2] (2x2), Y2 = B_block[2:4,0:2] (2x2)
            # C_block[a] = X1*Y1 + X2*Y2
            
            # Use Strassen for each product:
            # X1*Y1: 7 terms
            # X2*Y2: 7 terms
            
            for prod_idx in range(2):  # 0 for X1*Y1, 1 for X2*Y2
                # prod_idx 0: X = A[a, 0:2], Y = B_block[0:2, 0:2]
                # prod_idx 1: X = A[a, 2:4], Y = B_block[2:4, 0:2]
                if prod_idx == 0:
                    x_col_offset = 0  # columns 0-1 of A
                    y_row_offset = 0  # rows 0-1 of B_block
                else:
                    x_col_offset = 2  # columns 2-3 of A
                    y_row_offset = 2  # rows 2-3 of B_block
                
                for s in range(7):
                    # U part: A entries with alpha[s] coefficients
                    for ai in range(4):
                        if alpha[s][ai] != 0:
                            # ai indexes position in A: row=ai//2, col=ai%2
                            a_row, a_col = ai // 2, ai % 2
                            # Global A index: row = a*2 + a_row, col = x_col_offset + a_col
                            u_idx = (a * 2 + a_row) * 4 + (x_col_offset + a_col)
                            U[u_idx, col] += alpha[s][ai]
                    
                    # V part: B_block entries with beta[s] coefficients
                    for bi in range(4):
                        if beta[s][bi] != 0:
                            # bi indexes position in B_block: row=bi//2, col=bi%2
                            b_row, b_col = bi // 2, bi % 2
                            # Global B index: row = y_row_offset + b_row, col = 2*block + b_col
                            v_idx = (y_row_offset + b_row) * 6 + (2 * block + b_col)
                            V[v_idx, col] += beta[s][bi]
                    
                    # W part: output of Strassen
                    for ca, cb, gamma in recon[s]:
                        # ca, cb are positions in the 2x2 output
                        # This output is C_block[a*2:(a+1)*2, 0:2]
                        # Global C index: row = a*2 + ca, col = 2*block + cb
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
    deadline = _t0 + 140.0  # safety margin
    
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
    
    # Verify
    if not check(U84, V84, W84, atol=1e-6):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook()
        def to_list(mat):
            return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    best_U, best_V, best_W = U84.copy(), V84.copy(), W84.copy()
    best_rank = R84
    
    # Try to reduce rank using coordinate descent
    # Strategy: try removing columns one at a time
    
    for target_rank in range(R84 - 1, 60, -1):
        if time.time() > deadline - 20:
            break
        
        # Start from current best truncated to target_rank
        if target_rank < best_rank:
            U = best_U[:, :target_rank].copy()
            V = best_V[:, :target_rank].copy()
            W = best_W[:, :target_rank].copy()
        else:
            U = best_U.copy()
            V = best_V.copy()
            W = best_W.copy()
        
        err = err_of(U, V, W)
        
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            continue
        
        # Coordinate descent with small perturbations
        np.random.seed(target_rank)
        for it in range(50):
            if time.time() > deadline - 15:
                break
            if err < 1e-10:
                break
            improved = False
            for c in range(target_rank):
                if time.time() > deadline - 15:
                    break
                Uo, Vo, Wo = U[:, c].copy(), V[:, c].copy(), W[:, c].copy()
                # Try small random perturbations
                for scale in [0.1, 0.01, 0.001]:
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
        
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to polish the best solution
    if time.time() < deadline - 10:
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        err = err_of(U, V, W)
        
        if err > 1e-10:
            # Use scipy minimize to reduce error
            def objective(x):
                R = len(x) // (n*m + m*p + p*n)
                u_part = x[:R*(n*m)].reshape(R, n*m).T
                v_part = x[R*(n*m):R*(n*m + R*(m*p))].reshape(R, m*p).T
                w_part = x[R*(n*m + R*(m*p)):].reshape(R, p*n).T
                recon = np.einsum('ar,br,cr->abc', u_part, v_part, w_part)
                return np.sum((T - recon) ** 2)
            
            x0 = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
                if res.fun < err:
                    R = best_rank
                    U = res.x[:R*(n*m)].reshape(R, n*m).T
                    V = res.x[R*(n*m):R*(n*m + R*(m*p))].reshape(R, m*p).T
                    W = res.x[R*(n*m + R*(m*p)):].reshape(R, p*n).T
                    if check(U, V, W, atol=1e-8):
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            except:
                pass
    
    # Convert to lists with integer rounding where possible
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
    
    # Final verification
    if not check(best_U, best_V, best_W, atol=1e-6):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook()
        return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
