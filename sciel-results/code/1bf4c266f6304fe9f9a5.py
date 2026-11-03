import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T has dimensions (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u = i*m+j, v = j*p+k, w = k*n+i for some i, j, k
    # Otherwise 0.
    
    # The schoolbook solution has rank = n*m*p = 105.
    # We want to find a lower rank decomposition with integer/half-integer entries.
    
    # Strategy:
    # 1. Use the known lower bound and known results for small matrix multiplication.
    #    For 3x5x7, the Strassen-like algorithms might reduce the rank.
    #    However, 3, 5, 7 are odd.
    #    Let's check if a block decomposition helps.
    #    n=3, m=5, p=7.
    #    If we pad to 4x8x8, we can use Strassen for 4x4? No, Strassen is 2x2.
    #    Pad to 4x8x8:
    #    A (4x8) = [A (3x5) | 0 (3x3); 0 (1x5) | 0 (1x3)]
    #    B (8x8) = [B (5x7) | 0 (5x1); 0 (3x7) | 0 (3x1)]
    #    C (4x8) = [C (3x7) | 0 (3x1); 0 (1x7) | 0 (1x1)]
    #    This doesn't seem to simplify immediately.
    
    #    Let's try a direct search for rank 35, 40, ..., 104.
    #    Since random search is unlikely to find it, we need a structured approach.
    #    
    #    One known technique is to use the "Laser Method" or "Capset" constructions, but those are for large N.
    #    For small N, exact algorithms are used.
    #    
    #    Let's try to solve the linear system for a specific rank R.
    #    T = sum_{r=1}^R u_r v_r^T w_r^T
    #    This is a system of (n*m)*(m*p)*(p*n) equations for R * (n*m + m*p + p*n) unknowns.
    #    15*35*21 = 11025 equations.
    #    Unknowns: R * (15 + 35 + 21) = 71 R.
    #    If R=35, 2485 unknowns. Underdetermined.
    #    If R=105, 7455 unknowns.
    #    
    #    We can fix U and V randomly (or as identity-like) and solve for W?
    #    T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    #    For fixed U, V, this is linear in W.
    #    Let M_r[u, v] = U[u, r] V[v, r].
    #    Then T[u, v, w] = sum_r M_r[u, v] W[w, r].
    #    This is a linear system A * W_flat = T_flat.
    #    A is (11025) x (21*R).
    #    If we fix U and V, we can solve for W using least squares.
    #    Then we check if the error is 0.
    #    If not, we update U and V using gradient descent or ALS.
    
    #    Let's implement an ALS (Alternating Least Squares) approach.
    #    1. Initialize U, V, W randomly.
    #    2. Fix V, W. Solve for U.
    #       U[u, r] appears in equations for all v, w.
    #       T[u, v, w] = sum_r U[u, r] (V[v, r] W[w, r])
    #       Let C_r[v, w] = V[v, r] W[w, r].
    #       Then for fixed u, we have a system for U[u, :].
    #       Actually, for fixed u, the equations are:
    #       T[u, v, w] = sum_r U[u, r] C_r[v, w]
    #       This is a linear system for the vector U[u, :] of size R.
    #       We have (m*p)*(p*n) = 35*21 = 735 equations for R unknowns.
    #       If R < 735, we can solve in least squares sense.
    #       But we have 15 such systems (one for each u).
    #       So we solve 15 independent least squares problems.
    #    3. Similarly for V and W.
    #    4. Repeat until convergence.
    #    5. Check if exact solution is found.
    #    6. Try to snap to half-integers.
    
    #    This might be slow, but let's try.
    
    #    First, let's define the target tensor T as a 3D array.
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    #    We will try ranks from a lower bound up to 105.
    #    Lower bound: max(nm, mp, np) = 35.
    #    Also, rank >= max(n, m, p) * something?
    #    For matrix multiplication, rank >= n*m*p / (something)?
    #    Actually, rank >= n*m*p is not true, Strassen reduces it.
    #    But for 3x5x7, maybe rank is close to 105?
    #    Let's check known results.
    #    Actually, I don't know the exact rank for 3x5x7.
    #    Let's start with rank 35 and go up.
    
    #    To speed up, we can use a sparse representation of T.
    #    But ALS requires solving linear systems, which is easier with dense matrices.
    #    15x35x21 is small enough for dense.
    
    def check_exact(U, V, W, rank):
        # U: (15, R), V: (35, R), W: (21, R)
        # Compute T_decomp and compare with T.
        # T_decomp[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        # This can be computed efficiently.
        # T_decomp = sum_r U[:, r, None, None] * V[None, :, r, None] * W[None, None, :, r]
        # Or:
        # M_r = U[:, r] * V[:, r] (15x35)
        # T_decomp[u, v, w] = sum_r M_r[u, v] W[w, r]
        
        T_decomp = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for r in range(rank):
            M_r = np.outer(U[:, r], V[:, r])
            T_decomp += M_r[:, :, None] * W[:, r][None, None, :]
        
        # Check if T_decomp is close to T
        # We only care about the non-zero entries of T being 1 and zero entries being 0.
        # But for exactness, we need all entries to match.
        if np.allclose(T_decomp, T, atol=1e-6):
            return True
        return False

    def snap_to_half_integers(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    def als_decompose(T, n, m, p, rank, max_iter=100, tol=1e-8):
        # Initialize U, V, W
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = np.inf
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            # For each u, solve for U[u, :]
            # T[u, v, w] = sum_r U[u, r] (V[v, r] W[w, r])
            # Let A_u be a matrix of size (m*p*p*n) x rank
            # A_u[v, w, r] = V[v, r] W[w, r]
            # b_u = T[u, :, :].flatten()
            # Solve A_u U[u, :] = b_u
            
            for u in range(n * m):
                A_u = np.zeros((m * p * p * n, rank))
                b_u = T[u, :, :].flatten()
                idx = 0
                for v in range(m * p):
                    for w in range(p * n):
                        A_u[idx, :] = V[v, :] * W[w, :]
                        idx += 1
                # Solve least squares
                U[u, :], residuals, rank_u, sv = np.linalg.lstsq(A_u, b_u, rcond=None)
            
            # Update V
            for v in range(m * p):
                A_v = np.zeros((n * m * p * n, rank))
                b_v = T[:, v, :].flatten()
                idx = 0
                for u in range(n * m):
                    for w in range(p * n):
                        A_v[idx, :] = U[u, :] * W[w, :]
                        idx += 1
                V[v, :], residuals, rank_v, sv = np.linalg.lstsq(A_v, b_v, rcond=None)
            
            # Update W
            for w in range(p * n):
                A_w = np.zeros((n * m * m * p, rank))
                b_w = T[:, :, w].flatten()
                idx = 0
                for u in range(n * m):
                    for v in range(m * p):
                        A_w[idx, :] = U[u, :] * V[v, :]
                        idx += 1
                W[w, :], residuals, rank_w, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            
            # Compute error
            # T_decomp = sum_r U[:, r] * V[:, r] * W[:, r]
            T_decomp = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for r in range(rank):
                M_r = np.outer(U[:, r], V[:, r])
                T_decomp += M_r[:, :, None] * W[:, r][None, None, :]
            
            error = np.linalg.norm(T_decomp - T)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
        
        return U, V, W, error

    # Try ranks from 35 to 105
    best_rank = 105
    best_U = None
    best_V = None
    best_W = None
    
    # We will try a few ranks and see if we can find a solution.
    # Given the time limit, we can't try all.
    # Let's try rank 35, 40, 50, 60, 70, 80, 90, 100, 105.
    ranks_to_try = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            U, V, W, error = als_decompose(T, n, m, p, rank, max_iter=50, tol=1e-6)
            
            if error < 1e-6:
                # Check exactness
                if check_exact(U, V, W, rank):
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
                    if check_exact(U_snap, V_snap, W_snap, rank):
                        best_rank = rank
                        best_U = U_snap
                        best_V = V_snap
                        best_W = W_snap
                        found = True
                        break
                    else:
                        # If snapping fails, keep the float solution?
                        # The problem requires integer/half-integer entries.
                        # So we must find a snapped solution.
                        # If we can't snap, we might need to search more.
                        pass
            
            # If error is small but not exact, try to polish
            if error < 1e-3:
                # Try to snap and check
                U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)
                if check_exact(U_snap, V_snap, W_snap, rank):
                    best_rank = rank
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    found = True
                    break
        
        if found:
            break
    
    # If no solution found, return schoolbook
    if best_U is None:
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
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
