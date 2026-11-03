import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # 1. Start with schoolbook (rank 48) as fallback
    rank_sb = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # 2. Try Strassen-based decomposition (rank 44)
    if time.time() < deadline - 5:
        U_s, V_s, W_s = _strassen_based(n, m, p)
        if _check_exact(U_s, V_s, W_s, n, m, p):
            best_result = (U_s, V_s, W_s)
            best_rank = U_s.shape[1]

    # 3. Try to reduce rank further via optimization
    if best_rank > 44 and time.time() < deadline - 10:
        # This shouldn't happen, but keep as safety
        pass

    # 4. Attempt to find lower rank decompositions
    # Try ranks from 43 down to 20
    for rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 15:
            break
        # Give more time to lower ranks
        time_left = deadline - time.time() - 10
        if time_left < 3:
            break
        found = _try_find_rank(n, m, p, rank, max_restarts=8, time_limit=min(8, time_left), deadline=deadline)
        if found is not None:
            best_result = found
            best_rank = rank
            # If we find a very low rank, stop
            if rank <= 20:
                break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_based(n, m, p):
    """
    Use Strassen's algorithm for 2x2 blocks to reduce rank.
    Split 3x4 * 4x4 into:
    - 1x4 * 4x4 (schoolbook, rank 16)
    - 2x4 * 4x4 (use Strassen on 2x2 sub-blocks)
    
    For 2x4 * 4x4:
    Split into four 2x2 * 2x2 blocks.
    Each 2x2*2x2 uses Strassen (rank 7 instead of 8).
    Total: 4 * 7 = 28 for the 2x4 part.
    Total rank: 16 + 28 = 44.
    """
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: First row of A (1x4) times B (4x4) - schoolbook
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: Rows 1,2 of A (2x4) times B (4x4)
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    
    # Block positions:
    # A11: rows 1,2; cols 0,1 -> U indices: (1*4+0, 1*4+1, 2*4+0, 2*4+1)
    # A12: rows 1,2; cols 2,3 -> U indices: (1*4+2, 1*4+3, 2*4+2, 2*4+3)
    # A21: rows 3,4; cols 0,1 -> Wait, n=3, so rows are 0,1,2
    # Let me re-index:
    # A is 3x4, B is 4x4
    # Row 0: handled above
    # Rows 1,2: this is a 2x4 matrix
    # Split into 2x2 blocks:
    # A[1:3, 0:2] and A[1:3, 2:4]
    # B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    
    # Strassen for each 2x2 * 2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Block (i,j) where i in {1,2}, j in {0,1} for the 2x4 part
    # i=1 corresponds to row 1, i=2 corresponds to row 2
    # j=0 corresponds to cols 0,1, j=1 corresponds to cols 2,3
    
    for i_block in range(2):  # i_block: 0 for row 1, 1 for row 2
        for j_block in range(2):  # j_block: 0 for cols 0,1, 1 for cols 2,3
            # This 2x2 block of A is at rows [i_block+1, i_block+2], cols [j_block*2, j_block*2+2]
            # This 2x2 block of B is at rows [j_block*2, j_block*2+2], cols [i_block*2, i_block*2+2]
            # Wait, I need to be more careful.
            
            # Let's define the 2x2 sub-blocks of A (rows 1,2) and B:
            # A_sub[i_block, j_block] is a 2x2 block
            # A_sub[0,0]: rows 1,2; cols 0,1
            # A_sub[0,1]: rows 1,2; cols 2,3
            # A_sub[1,0]: rows 1,2; cols 0,1 -- No, that's wrong.
            
            # Actually, for 2x4 * 4x4:
            # A is 2x4, B is 4x4
            # Split A into two 2x2 blocks: A_left (cols 0,1), A_right (cols 2,3)
            # Split B into four 2x2 blocks:
            # B_TL (rows 0,1; cols 0,1), B_TR (rows 0,1; cols 2,3)
            # B_BL (rows 2,3; cols 0,1), B_BR (rows 2,3; cols 2,3)
            
            # C = A * B = [A_left * B_TL + A_right * B_BL, A_left * B_TR + A_right * B_BR]
            # Each term is 2x2 * 2x2 = 2x2
            # So C has two 2x2 blocks: C_left and C_right
            
            # For C_left = A_left * B_TL + A_right * B_BL:
            # Use Strassen for each product, but we can't directly combine.
            # Instead, compute each 2x2*2x2 with Strassen (rank 7 each).
            
            # So for the 2x4 * 4x4 part, we have 4 products of 2x2 * 2x2:
            # P1 = A_left * B_TL
            # P2 = A_left * B_TR
            # P3 = A_right * B_BL
            # P4 = A_right * B_BR
            # Each uses 7 Strassen terms, total 28.
            
            pass
    
    # Let me redo this more carefully.
    # The 2x4 * 4x4 part:
    # A_2x4 = [[a00, a01, a02, a03],
    #           [a10, a11, a12, a13]]
    # B_4x4 = [[b00, b01, b02, b03],
    #           [b10, b11, b12, b13],
    #           [b20, b21, b22, b23],
    #           [b30, b31, b32, b33]]
    # C_2x4 = A_2x4 * B_4x4
    
    # Split into 2x2 blocks:
    # A_L = [[a00, a01], [a10, a11]] (cols 0,1)
    # A_R = [[a02, a03], [a12, a13]] (cols 2,3)
    # B_TL = [[b00, b01], [b10, b11]] (rows 0,1; cols 0,1)
    # B_TR = [[b02, b03], [b12, b13]] (rows 0,1; cols 2,3)
    # B_BL = [[b20, b21], [b30, b31]] (rows 2,3; cols 0,1)
    # B_BR = [[b22, b23], [b32, b33]] (rows 2,3; cols 2,3)
    
    # C_L = A_L * B_TL + A_R * B_BL (2x2 result)
    # C_R = A_L * B_TR + A_R * B_BR (2x2 result)
    
    # For each 2x2*2x2 product, use Strassen (7 terms).
    # Total: 4 * 7 = 28 terms for the 2x4 part.
    
    # Strassen terms for X(2x2) * Y(2x2):
    # M1 = (X00 + X11) * (Y00 + Y11)
    # M2 = (X10 + X11) * Y00
    # M3 = X00 * (Y01 - Y11)
    # M4 = X11 * (Y10 - Y00)
    # M5 = (X00 + X01) * Y11
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each of the 4 products, add 7 terms.
    
    # Product 1: A_L * B_TL
    # X = A_L, Y = B_TL
    # X00 = a00, X01 = a01, X10 = a10, X11 = a11
    # Y00 = b00, Y01 = b01, Y10 = b10, Y11 = b11
    _add_strassen_block(U, V, W, n, m, p, col,
                        # X indices (row, col) in A_2x4 -> global (row+1, col) in A
                        (1, 0), (1, 1), (2, 0), (2, 1),  # X00, X01, X10, X11
                        # Y indices (row, col) in B -> global (row, col) in B
                        (0, 0), (0, 1), (1, 0), (1, 1),  # Y00, Y01, Y10, Y11
                        # Z indices (row, col) in C_2x4 -> global (row+1, col) in C
                        (1, 0), (1, 1), (2, 0), (2, 1))   # Z00, Z01, Z10, Z11
    col += 7
    
    # Product 2: A_L * B_TR
    # X = A_L, Y = B_TR
    _add_strassen_block(U, V, W, n, m, p, col,
                        (1, 0), (1, 1), (2, 0), (2, 1),  # X00, X01, X10, X11
                        (0, 2), (0, 3), (1, 2), (1, 3),  # Y00, Y01, Y10, Y11
                        (1, 2), (1, 3), (2, 2), (2, 3))   # Z00, Z01, Z10, Z11
    col += 7
    
    # Product 3: A_R * B_BL
    # X = A_R, Y = B_BL
    _add_strassen_block(U, V, W, n, m, p, col,
                        (1, 2), (1, 3), (2, 2), (2, 3),  # X00, X01, X10, X11
                        (2, 0), (2, 1), (3, 0), (3, 1),  # Y00, Y01, Y10, Y11
                        (1, 0), (1, 1), (2, 0), (2, 1))   # Z00, Z01, Z10, Z11
    col += 7
    
    # Product 4: A_R * B_BR
    # X = A_R, Y = B_BR
    _add_strassen_block(U, V, W, n, m, p, col,
                        (1, 2), (1, 3), (2, 2), (2, 3),  # X00, X01, X10, X11
                        (2, 2), (2, 3), (3, 2), (3, 3),  # Y00, Y01, Y10, Y11
                        (1, 2), (1, 3), (2, 2), (2, 3))   # Z00, Z01, Z10, Z11
    col += 7
    
    return U, V, W

def _add_strassen_block(U, V, W, n, m, p, start_col,
                         x00, x01, x10, x11,
                         y00, y01, y10, y11,
                         z00, z01, z10, z11):
    """
    Add 7 Strassen terms for a 2x2 * 2x2 product.
    x00, x01, x10, x11 are (row, col) indices in A (n x m)
    y00, y01, y10, y11 are (row, col) indices in B (m x p)
    z00, z01, z10, z11 are (row, col) indices in C (n x p)
    
    Wait, the indexing is confusing. Let me use explicit indices.
    U has shape (n*m, rank), indexed by [i*m + j, col] for A[i,j]
    V has shape (m*p, rank), indexed by [j*p + k, col] for B[j,k]
    W has shape (p*n, rank), indexed by [k*n + i, col] for C[i,k]
    """
    # M1 = (X00 + X11) * (Y00 + Y11)
    U[x00[0]*m + x00[1], start_col] = 1
    U[x11[0]*m + x11[1], start_col] = 1
    V[y00[0]*p + y00[1], start_col] = 1
    V[y11[0]*p + y11[1], start_col] = 1
    W[z00[0]*n + z00[1], start_col] = 1  # Z00 += M1
    W[z11[0]*n + z11[1], start_col] = 1  # Z11 += M1
    start_col += 1
    
    # M2 = (X10 + X11) * Y00
    U[x10[0]*m + x10[1], start_col] = 1
    U[x11[0]*m + x11[1], start_col] = 1
    V[y00[0]*p + y00[1], start_col] = 1
    W[z10[0]*n + z10[1], start_col] = 1  # Z10 += M2
    W[z11[0]*n + z11[1], start_col] = -1  # Z11 -= M2
    start_col += 1
    
    # M3 = X00 * (Y01 - Y11)
    U[x00[0]*m + x00[1], start_col] = 1
    V[y01[0]*p + y01[1], start_col] = 1
    V[y11[0]*p + y11[1], start_col] = -1
    W[z01[0]*n + z01[1], start_col] = 1  # Z01 += M3
    W[z11[0]*n + z11[1], start_col] = 1  # Z11 += M3
    start_col += 1
    
    # M4 = X11 * (Y10 - Y00)
    U[x11[0]*m + x11[1], start_col] = 1
    V[y10[0]*p + y10[1], start_col] = 1
    V[y00[0]*p + y00[1], start_col] = -1
    W[z00[0]*n + z00[1], start_col] = 1  # Z00 += M4
    W[z10[0]*n + z10[1], start_col] = 1  # Z10 += M4
    start_col += 1
    
    # M5 = (X00 + X01) * Y11
    U[x00[0]*m + x00[1], start_col] = 1
    U[x01[0]*m + x01[1], start_col] = 1
    V[y11[0]*p + y11[1], start_col] = 1
    W[z00[0]*n + z00[1], start_col] = -1  # Z00 -= M5
    W[z01[0]*n + z01[1], start_col] = 1   # Z01 += M5
    start_col += 1
    
    # M6 = (X10 - X00) * (Y00 + Y01)
    U[x10[0]*m + x10[1], start_col] = 1
    U[x00[0]*m + x00[1], start_col] = -1
    V[y00[0]*p + y00[1], start_col] = 1
    V[y01[0]*p + y01[1], start_col] = 1
    W[z11[0]*n + z11[1], start_col] = 1  # Z11 += M6
    start_col += 1
    
    # M7 = (X01 - X11) * (Y10 + Y11)
    U[x01[0]*m + x01[1], start_col] = 1
    U[x11[0]*m + x11[1], start_col] = -1
    V[y10[0]*p + y10[1], start_col] = 1
    V[y11[0]*p + y11[1], start_col] = 1
    W[z00[0]*n + z00[1], start_col] = 1  # Z00 += M7
    start_col += 1

def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10, deadline=None):
    start_time = time.time()

    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        if deadline is not None and time.time() > deadline - 5:
            break

        # Strategy A: prune from schoolbook
        # Strategy B: structured random init
        if restart % 2 == 0:
            U, V, W = _init_prune(n, m, p, rank)
        else:
            U, V, W = _init_random(n, m, p, rank)

        # ALS iterations
        U, V, W, max_err = _als(U, V, W, n, m, p,
                                 iters=40,
                                 start_time=start_time,
                                 time_limit=time_limit)

        # Check
        if max_err < 1e-6:
            # Polish with SciPy LM
            U2, V2, W2 = _polish_lm(U, V, W, n, m, p,
                                  time_left=time_limit - (time.time() - start_time))
            if U2 is not None:
                return U2, V2, W2
            else:
                # Try direct snap
                snapped, ok = _snap_all(U, V, W, 1e-4)
                if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
                    return snapped

    return None

def _init_prune(n, m, p, rank):
    """Start from schoolbook, keep 'rank' random terms, re-solve."""
    nmp = n * m * p
    U, V, W = _schoolbook(n, m, p)

    # Choose 'rank' random columns to keep
    cols = random.sample(range(nmp), rank)
    U = U[:, cols]
    V = V[:, cols]
    W = W[:, cols]

    # Add small perturbation to allow re-solvability
    U += np.random.randn(n * m, rank) * 0.05
    V += np.random.randn(m * p, rank) * 0.05
    W += np.random.randn(p * n, rank) * 0.05
    return U, V, W

def _init_random(n, m, p, rank):
    """Structured random init with some schoolbook structure."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.25:
                        U[i * m + j, r] = random.choice([-1.0, 1.0])
                        V[j * p + k, r] = random.choice([-1.0, 1.0])
                        W[k * n + i, r] = random.choice([-1.0, 1.0])

    # Add noise to break symmetry
    U += np.random.randn(n * m, rank) * 0.2
    V += np.random.randn(m * p, rank) * 0.2
    W += np.random.randn(p * n, rank) * 0.2
    return U, V, W

def _als(U, V, W, n, m, p, iters=60, start_time=None, time_limit=10):
    """Alternating least squares for tensor decomposition."""
    max_err = float('inf')
    prev_err = float('inf')

    for it in range(iters):
        if start_time is not None and time.time() > start_time + time_limit:
            break

        # Update U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                V_part = V[j * p:(j + 1) * p, :]
                W_part = W[np.arange(p) * n + i, :]
                A = V_part * W_part
                b = np.ones(p)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except Exception:
                    U_new[idx, :] = U[idx, :]
        U = U_new

        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                U_part = U[np.arange(n) * m + j, :]
                W_part = W[k * n:(k + 1) * n, :]
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except Exception:
                    V_new[idx, :] = V[idx, :]
        V = V_new

        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                U_part = U[i * m:(i + 1) * m, :]
                V_part = V[np.arange(m) * p + k, :]
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except Exception:
                    W_new[idx, :] = W[idx, :]
        W = W_new

        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e

        if max_err < 1e-8:
            break
        if abs(prev_err - max_err) < 1e-11:
            break
        prev_err = max_err

    return U, V, W, max_err

def _polish_lm(U, V, W, n, m, p, time_left=5):
    """Polish using Levenberg-Marquardt."""
    if time_left < 0.5:
        return None, None, None

    start = time.time()
    rank = U.shape[1]
    
    # Flatten parameters
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def residuals(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_p[i*m+j, :], V_p[j*p+k, :] * W_p[k*n+i, :])
                    res.append(1.0 - val)
        return np.array(res)

    try:
        result = least_squares(residuals, params, method='lm', max_nfev=100)
        if result.success or result.cost < 1e-8:
            U_new = result.x[:n*m*rank].reshape(n*m, rank)
            V_new = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_new = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Check and snap
            snapped, ok = _snap_all(U_new, V_new, W_new, 1e-3)
            if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
                return snapped
            else:
                return None, None, None
    except Exception:
        return None, None, None

def _snap_all(U, V, W, tol=1e-6):
    """Snap all three to half-integers if close."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    ok = (np.max(np.abs(U - U_s)) < tol and
          np.max(np.abs(V - V_s)) < tol and
          np.max(np.abs(W - W_s)) < tol)
    return (U_s, V_s, W_s), ok

def _check_exact(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
