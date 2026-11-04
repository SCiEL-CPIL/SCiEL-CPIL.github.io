"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 130  # safety margin before 150s kill


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c."""
    dim_u, dim_v, dim_w = n*m, m*p, p*n
    T_rec = np.zeros((dim_u, dim_v, dim_w))
    for r in range(U.shape[1]):
        T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    # Build target
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T_rec, T, atol=tol)


def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) where keys are (row, col) and values are coefficients.
    
    For X(2x2) * Y(2x2) = Z(2x2):
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11)(Y00)
    M3 = (X00)(Y01-Y11)
    M4 = (X11)(Y10-Y00)
    M5 = (X00+X01)(Y11)
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    """
    terms = [
        # M1: (X00+X11)(Y00+Y11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2: (X10+X11)(Y00)
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3: (X00)(Y01-Y11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4: (X11)(Y10-Y00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5: (X00+X01)(Y11)
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6: (X10-X00)(Y00+Y01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7: (X01-X11)(Y10+Y11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    return terms


def _build_strassen_44():
    """
    Build an explicit rank-44 decomposition for 3x4x4 matrix multiplication.
    
    Strategy:
    - Row 0 of A (1x4) * B (4x4) -> schoolbook, rank 16
    - Rows 1-2 of A (2x4) * B (4x4) -> split into 4 blocks of 2x2*2x2=2x2, Strassen rank 7 each = 28
    Total: 16 + 28 = 44
    
    Indexing convention:
    U row index: i*m + j where i in [0,3) is row of A, j in [0,4) is col of A
    V row index: j*p + k where j in [0,4) is row of B, k in [0,4) is col of B
    W row index: k*n + i where k in [0,4) is col of C, i in [0,3) is row of C
    """
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    U = np.zeros((dim_u, 44))
    V = np.zeros((dim_v, 44))
    W = np.zeros((dim_w, 44))
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) * B (4x4) = C row 0 (1x4)
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # Schoolbook: 4*4 = 16 terms
    for k in range(p):
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1
            col += 1
    
    # Part 2: Rows 1-2 of A (2x4) * B (4x4) = C rows 1-2 (2x4)
    # Split into 4 blocks:
    # A_L (cols 0-1) * B_top (rows 0-1) -> C_L (cols 0-1)
    # A_L (cols 0-1) * B_bot (rows 2-3) -> C_R (cols 2-3)  -- wait, need to be careful
    # Actually: C[1:3, k] = sum_j A[1:3, j] * B[j, k]
    # Split j into {0,1} and {2,3}:
    # C[1:3, 0:2] gets contributions from A[1:3, 0:2]*B[0:2, 0:2] and A[1:3, 2:3]*B[2:3, 0:2]
    # C[1:3, 2:4] gets contributions from A[1:3, 0:2]*B[0:2, 2:4] and A[1:3, 2:3]*B[2:3, 2:4]
    
    # Block 1: A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    # Block 2: A[1:3, 0:2] * B[0:2, 2:4] -> C[1:3, 2:4]
    # Block 3: A[1:3, 2:4] * B[2:4, 0:2] -> C[1:3, 0:2]
    # Block 4: A[1:3, 2:4] * B[2:4, 2:4] -> C[1:3, 2:4]
    
    strassen_terms = _strassen_2x2_terms()
    
    blocks = [
        # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
        (1, 0, 0, 0, 1, 0),  # A[1:3,0:2] * B[0:2,0:2] -> C[1:3,0:2]
        (1, 0, 0, 2, 1, 2),  # A[1:3,0:2] * B[0:2,2:4] -> C[1:3,2:4]
        (1, 2, 2, 0, 1, 0),  # A[1:3,2:4] * B[2:4,0:2] -> C[1:3,0:2]
        (1, 2, 2, 2, 1, 2),  # A[1:3,2:4] * B[2:4,2:4] -> C[1:3,2:4]
    ]
    
    for (ar, ac, br, bc, cr, cc) in blocks:
        for (x_coeffs, y_coeffs, z_coeffs) in strassen_terms:
            for (xi, xj), ux in x_coeffs.items():
                U[(ar + xi) * m + (ac + xj), col] += ux
            for (yi, yj), vy in y_coeffs.items():
                V[(br + yi) * p + (bc + yj), col] += vy
            for (zi, zj), wz in z_coeffs.items():
                W[(cc + zj) * n + (cr + zi), col] += wz
            col += 1
    
    return U, V, W


def _build_schoolbook_48():
    """Build the rank-48 schoolbook decomposition."""
    n, m, p = 3, 4, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, 48))
    V = np.zeros((dim_v, 48))
    W = np.zeros((dim_w, 48))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with Strassen rank-44
    U, V, W = _build_strassen_44()
    best_rank = 44
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def check(U, V, W, tol=1e-9):
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_rec, T, atol=tol)
    
    def compute_loss(U, V, W):
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = T_rec - T
        return np.sum(err**2)
    
    # Verify the Strassen decomposition
    if not check(U, V, W, tol=1e-8):
        # Fallback to schoolbook if Strassen is broken
        U, V, W = _build_schoolbook_48()
        best_rank = 48
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank using random restarts + gradient descent
    # Focus on ranks 40-43
    for R in range(43, 39, -1):
        if time.time() > DEADLINE - 30:
            break
        
        found = False
        for restart in range(5):
            if time.time() > DEADLINE - 25 or found:
                break
            
            # Initialize with Strassen structure but with some terms merged
            # Randomly assign the 44 Strassen terms to R slots
            U_r = np.zeros((dim_u, R))
            V_r = np.zeros((dim_v, R))
            W_r = np.zeros((dim_w, R))
            
            # Start from Strassen and randomly merge some columns
            if best_rank >= 44:
                assignments = np.random.randint(0, R, size=best_rank)
                for idx in range(best_rank):
                    r = assignments[idx]
                    U_r[:, r] += best_U[:, idx]
                    V_r[:, r] += best_V[:, idx]
                    W_r[:, r] += best_W[:, idx]
            else:
                assignments = np.random.randint(0, R, size=48)
                U_s, V_s, W_s = _build_schoolbook_48()
                for idx in range(48):
                    r = assignments[idx]
                    U_r[:, r] += U_s[:, idx]
                    V_r[:, r] += V_s[:, idx]
                    W_r[:, r] += W_s[:, idx]
            
            # Add small random perturbation
            U_r += np.random.randn(dim_u, R) * 0.05
            V_r += np.random.randn(dim_v, R) * 0.05
            W_r += np.random.randn(dim_w, R) * 0.05
            
            # Gradient descent with adaptive learning rate
            lr = 0.002
            prev_loss = np.inf
            for step in range(3000):
                if time.time() > DEADLINE - 20:
                    break
                
                T_rec = np.zeros((dim_u, dim_v, dim_w))
                for r in range(R):
                    T_rec += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                err = T_rec - T
                loss = np.sum(err**2)
                
                dU = 2 * np.einsum('abc,br,cr->ar', err, V_r, W_r)
                dV = 2 * np.einsum('abc,ar,cr->br', err, U_r, W_r)
                dW = 2 * np.einsum('abc,ar,br->cr', err, U_r, V_r)
                
                U_r -= lr * dU
                V_r -= lr * dV
                W_r -= lr * dW
                
                if loss < prev_loss * 0.99:
                    lr = min(lr * 1.05, 0.02)
                elif loss > prev_loss * 1.01:
                    lr *= 0.95
                prev_loss = loss
                
                if loss < 1e-12:
                    break
                if loss > 1e6:
                    break
            
            if check(U_r, V_r, W_r, tol=1e-8):
                best_rank = R
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
        
        if found:
            break
    
    # Try to round to half-integers
    def to_half_int(arr):
        return np.round(arr * 2) / 2
    
    U_h, V_h, W_h = to_half_int(best_U), to_half_int(best_V), to_half_int(best_W)
    if check(U_h, V_h, W_h, tol=1e-8):
        best_U, best_V, best_W = U_h, V_h, W_h
    
    # Final polish: try to clean up near-zero entries
    def clean_up(arr, tol=1e-6):
        arr = arr.copy()
        arr[np.abs(arr) < tol] = 0.0
        # Round to nearest 0.5
        arr = np.round(arr * 2) / 2
        return arr
    
    U_c, V_c, W_c = clean_up(best_U), clean_up(best_V), clean_up(best_W)
    if check(U_c, V_c, W_c, tol=1e-8):
        best_U, best_V, best_W = U_c, V_c, W_c
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
