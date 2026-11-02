import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def verify(U, V, W, R, tol=1e-9):
        """Verify tensor decomposition using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_true = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_true[i * m + j, j * p + k, k * n + i] = 1.0
        return np.max(np.abs(T_recon - T_true)) < tol

    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    def snap_to_half_integer(mat, tol=1e-6):
        snapped = np.round(mat * 2) / 2
        if np.max(np.abs(mat - snapped)) < tol:
            return snapped
        return mat

    def build_hybrid_strassen():
        """Build hybrid Strassen/schoolbook decomposition.
        
        Split A (2x3) = [A1 (2x2) | A2 (2x1)]
        Split B (3x7) = [B1 (2x7); B2 (1x7)]
        C = A1*B1 + A2*B2
        
        For A1*B1 (2x2 * 2x7): Strassen on 2x2 column blocks + schoolbook for last col
        For A2*B2 (2x1 * 1x7): schoolbook
        """
        U_list = []
        V_list = []
        W_list = []
        
        # U indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3, A[0,2]=4, A[1,2]=5
        # V indices: B[0,0..6]=0..6, B[1,0..6]=7..13, B[2,0..6]=14..20
        # W indices: C[0,0..6]=0..6, C[1,0..6]=7..13
        
        def add_strassen_2x2_block(c0, c1):
            """Strassen for A1 * B1[:, c0:c1+1] where B1 is 2x2."""
            # B indices for this block:
            # B[0,c0] = c0, B[0,c1] = c1, B[1,c0] = 7+c0, B[1,c1] = 7+c1
            # C indices: C[0,c0]=c0, C[0,c1]=c1, C[1,c0]=7+c0, C[1,c1]=7+c1
            
            b00, b01 = c0, c1
            b10, b11 = 7 + c0, 7 + c1
            c00, c01 = c0, c1
            c10, c11 = 7 + c0, 7 + c1
            
            # M1 = (A00+A11)*(B00+B11)
            U_list.append([1, 0, 0, 1, 0, 0])
            V_list.append([0]*21); V_list[-1][b00] = 1; V_list[-1][b11] = 1
            W_list.append([0]*14); W_list[-1][c00] = 1; W_list[-1][c11] = 1
            
            # M2 = (A10+A11)*B00
            U_list.append([0, 0, 1, 1, 0, 0])
            V_list.append([0]*21); V_list[-1][b00] = 1
            W_list.append([0]*14); W_list[-1][c10] = 1
            
            # M3 = A00*(B01-B11)
            U_list.append([1, 0, 0, 0, 0, 0])
            V_list.append([0]*21); V_list[-1][b01] = 1; V_list[-1][b11] = -1
            W_list.append([0]*14); W_list[-1][c00] = 1
            
            # M4 = A11*(B00+B10)
            U_list.append([0, 0, 0, 1, 0, 0])
            V_list.append([0]*21); V_list[-1][b00] = 1; V_list[-1][b10] = 1
            W_list.append([0]*14); W_list[-1][c00] = -1; W_list[-1][c10] = 1
            
            # M5 = (A00+A10)*B11
            U_list.append([1, 0, 1, 0, 0, 0])
            V_list.append([0]*21); V_list[-1][b11] = 1
            W_list.append([0]*14); W_list[-1][c01] = 1
            
            # M6 = (A11-A10)*(B00+B01)
            U_list.append([0, 0, -1, 1, 0, 0])
            V_list.append([0]*21); V_list[-1][b00] = 1; V_list[-1][b01] = 1
            W_list.append([0]*14); W_list[-1][c10] = 1; W_list[-1][c11] = -1
            
            # M7 = (A01-A11)*(B10+B11)
            U_list.append([0, 1, 0, -1, 0, 0])
            V_list.append([0]*21); V_list[-1][b10] = 1; V_list[-1][b11] = 1
            W_list.append([0]*14); W_list[-1][c00] = -1; W_list[-1][c01] = 1
        
        # Apply Strassen to column pairs (0,1), (2,3), (4,5)
        add_strassen_2x2_block(0, 1)
        add_strassen_2x2_block(2, 3)
        add_strassen_2x2_block(4, 5)
        
        # Column 6: schoolbook for A1 * B1[:,6]
        # C[0,6] = A00*B06 + A01*B16
        # C[1,6] = A10*B06 + A11*B16
        U_list.append([1, 0, 0, 0, 0, 0])
        V_list.append([0]*21); V_list[-1][6] = 1
        W_list.append([0]*14); W_list[-1][6] = 1
        
        U_list.append([0, 1, 0, 0, 0, 0])
        V_list.append([0]*21); V_list[-1][13] = 1
        W_list.append([0]*14); W_list[-1][6] = 1
        
        U_list.append([0, 0, 1, 0, 0, 0])
        V_list.append([0]*21); V_list[-1][6] = 1
        W_list.append([0]*14); W_list[-1][13] = 1
        
        U_list.append([0, 0, 0, 1, 0, 0])
        V_list.append([0]*21); V_list[-1][13] = 1
        W_list.append([0]*14); W_list[-1][13] = 1
        
        # Part 2: A2 (2x1) * B2 (1x7) schoolbook
        # C[0,k] += A[0,2]*B[2,k], C[1,k] += A[1,2]*B[2,k]
        for k in range(7):
            U_list.append([0, 0, 0, 0, 1, 0])
            V_list.append([0]*21); V_list[-1][14+k] = 1
            W_list.append([0]*14); W_list[-1][k] = 1
            
            U_list.append([0, 0, 0, 0, 0, 1])
            V_list.append([0]*21); V_list[-1][14+k] = 1
            W_list.append([0]*14); W_list[-1][7+k] = 1
        
        R = len(U_list)
        U_mat = np.array(U_list).T
        V_mat = np.array(V_list).T
        W_mat = np.array(W_list).T
        
        if verify(U_mat, V_mat, W_mat, R):
            return U_mat, V_mat, W_mat, R
        return None

    def continuous_optimize(R, attempt, deadline):
        """Optimize a rank-R decomposition."""
        if time.time() > deadline - 5:
            return None
        rng = np.random.default_rng(attempt * 7919 + R * 104729)
        
        vals = [-1, -0.5, 0, 0.5, 1]
        U = rng.choice(vals, size=(n * m, R)).astype(float)
        V = rng.choice(vals, size=(m * p, R)).astype(float)
        W = rng.choice(vals, size=(p * n, R)).astype(float)
        
        def flatten(U, V, W):
            return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def unflatten(x):
            u_size = n * m * R
            v_size = m * p * R
            U = x[:u_size].reshape(n * m, R)
            V = x[u_size:u_size+v_size].reshape(m * p, R)
            W = x[u_size+v_size:].reshape(p * n, R)
            return U, V, W
        
        def residuals(x):
            U, V, W = unflatten(x)
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            T_true = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_true[i * m + j, j * p + k, k * n + i] = 1.0
            return (T_recon - T_true).flatten()
        
        x0 = flatten(U, V, W)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000, ftol=1e-14, xtol=1e-14)
        except Exception:
            return None
        
        if sol.cost < 1e-12:
            U_sol, V_sol, W_sol = unflatten(sol.x)
            U_snapped = snap_to_half_integer(U_sol, tol=1e-5)
            V_snapped = snap_to_half_integer(V_sol, tol=1e-5)
            W_snapped = snap_to_half_integer(W_sol, tol=1e-5)
            
            if verify(U_snapped, V_snapped, W_snapped, R):
                return U_snapped, V_snapped, W_snapped
            
            all_close = all(
                np.max(np.abs(mat - snap_to_half_integer(mat, tol=1e-4))) < 1e-4
                for mat in [U_sol, V_sol, W_sol]
            )
            if all_close:
                return U_snapped, V_snapped, W_snapped
        
        return None

    # Build schoolbook as fallback
    U_sb, V_sb, W_sb, R_sb = schoolbook()
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = R_sb

    # Try hybrid Strassen first
    result = build_hybrid_strassen()
    if result is not None:
        U_sol, V_sol, W_sol, R_sol = result
        if verify(U_sol, V_sol, W_sol, R_sol):
            best_rank = R_sol
            best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

    # Try continuous optimization for ranks below current best
    min_rank_to_try = max(n * m, m * p, p * n)  # 21
    max_attempts_per_rank = 2
    
    for R in range(min_rank_to_try, best_rank):
        if time.time() > deadline - 15:
            break
        for attempt in range(max_attempts_per_rank):
            if time.time() > deadline - 15:
                break
            result = continuous_optimize(R, attempt, deadline)
            if result is not None:
                U_sol, V_sol, W_sol = result
                if verify(U_sol, V_sol, W_sol, R):
                    best_rank = R
                    best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    break

    return best_result
