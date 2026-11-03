import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank_schoolbook = n * m * p  # 40

    # Build the Strassen rank-36 decomposition (deterministic, fast)
    U, V, W = _build_strassen_36(n, m, p)
    best_rank = 36
    best_U, best_V, best_W = U, V, W

    # Precompute the tensor structure for efficient verification
    # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all (i,j,k) in [0,n) x [0,m) x [0,p)
    
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Strategy: Try to find rank < 36 by solving linear systems
    # For a given rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Approach: Fix U and V randomly (or structured), then solve for W linearly.
    # The system is linear in W: for each (i,j,k), sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    
    # We need the system to be exactly solvable with half-integer W.
    
    np.random.seed(12345)
    
    # Try ranks from 35 down to 20
    for R in range(35, 19, -1):
        if time.time() > deadline:
            break
            
        # Number of trials - be aggressive for small ranks
        # For R=35, try many; for smaller R, fewer but smarter
        trials = 5000 if R == 35 else 3000
        
        found = False
        for trial in range(trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V with small integer entries
            # Use entries from {-1, 0, 1} for simplicity
            U_try = np.random.randint(-1, 2, size=(n * m, R))
            V_try = np.random.randint(-1, 2, size=(m * p, R))
            
            # Build linear system for W
            # n_eq = n * m * p = 40 equations
            # n_var = p * n * R = 10 * R variables
            n_eq = n * m * p
            n_var = p * n * R
            
            A_mat = np.zeros((n_eq, n_var), dtype=np.float64)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * (m * p) + j * p + k
                        var_base = (k * n + i) * R
                        # A[eq_idx, var_base + r] = U[i*m+j, r] * V[j*p+k, r]
                        A_mat[eq_idx, var_base:var_base + R] = (
                            U_try[i * m + j, :] * V_try[j * p + k, :]
                        )
            
            B_vec = np.ones(n_eq)
            
            try:
                W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
            except:
                continue
            
            if residuals is None or len(residuals) == 0:
                continue
            if residuals[0] > 1e-10:
                continue
            
            # Round to half-integers
            W_rounded = np.round(W_flat * 2) / 2.0
            
            # Verify exactness
            W_mat = W_rounded.reshape((p * n, R))
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        sum_val = np.dot(
                            U_try[i * m + j, :],
                            V_try[j * p + k, :] * W_mat[k * n + i, :]
                        )
                        if abs(sum_val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_try.tolist()
                best_V = V_try.tolist()
                best_W = W_mat.tolist()
                found = True
                break
        
        if found:
            break  # Found a solution at this rank, stop
    
    # If we found rank 35, try to go lower
    if best_rank == 35 and time.time() < deadline:
        for R in range(34, 19, -1):
            if time.time() > deadline:
                break
            trials = 2000
            found = False
            for trial in range(trials):
                if time.time() > deadline:
                    break
                U_try = np.random.randint(-1, 2, size=(n * m, R))
                V_try = np.random.randint(-1, 2, size=(m * p, R))
                
                n_eq = n * m * p
                n_var = p * n * R
                A_mat = np.zeros((n_eq, n_var), dtype=np.float64)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * (m * p) + j * p + k
                            var_base = (k * n + i) * R
                            A_mat[eq_idx, var_base:var_base + R] = (
                                U_try[i * m + j, :] * V_try[j * p + k, :]
                            )
                
                B_vec = np.ones(n_eq)
                
                try:
                    W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, B_vec, rcond=None)
                except:
                    continue
                
                if residuals is None or len(residuals) == 0:
                    continue
                if residuals[0] > 1e-10:
                    continue
                
                W_rounded = np.round(W_flat * 2) / 2.0
                W_mat = W_rounded.reshape((p * n, R))
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            sum_val = np.dot(
                                U_try[i * m + j, :],
                                V_try[j * p + k, :] * W_mat[k * n + i, :]
                            )
                            if abs(sum_val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = R
                    best_U = U_try.tolist()
                    best_V = V_try.tolist()
                    best_W = W_mat.tolist()
                    found = True
                    break
            
            if found:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5 matrix multiplication."""
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Four Strassen 2x2x2 blocks + one schoolbook 2x4x1
    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
