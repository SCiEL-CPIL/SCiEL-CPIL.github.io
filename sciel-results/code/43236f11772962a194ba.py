import numpy as np
import time
import sys
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 135.0  # 15s safety margin from 150s limit

    # Schoolbook rank: 2*3*7 = 42
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2 rank-7
    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        # M3 = A00(B10-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        # M4 = A11(B01-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        # M6 = (A10-A00)(B00+B10)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        # M7 = (A01-A11)(B11+B01)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    # 2x1x2 rank-4 (optimal: 2*1*2 = 4)
    def get_2x1x2_rank4():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        return U, V, W

    # 2x3x1 rank-6 (optimal: 2*3*1 = 6)
    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    # 2x3x2 rank-11: Strassen(7) + 2x1x2(4)
    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        # Strassen handles cols 0,1 of B and cols 0,1 of C (the 2x2 part)
        U[0, 0:7] = U_s[0, :]  # A[0,0]
        U[1, 0:7] = U_s[1, :]  # A[0,1]
        U[3, 0:7] = U_s[2, :]  # A[1,0]
        U[4, 0:7] = U_s[3, :]  # A[1,1]
        V[0, 0:7] = V_s[0, :]  # B[0,0]
        V[1, 0:7] = V_s[1, :]  # B[0,1]
        V[3, 0:7] = V_s[2, :]  # B[1,0]
        V[4, 0:7] = V_s[3, :]  # B[1,1]
        W[0, 0:7] = W_s[0, :]  # C[0,0]
        W[1, 0:7] = W_s[1, :]  # C[0,1]
        W[2, 0:7] = W_s[2, :]  # C[1,0]
        W[3, 0:7] = W_s[3, :]  # C[1,1]
        # 2x1x2 handles col 2 of B and col 2 of C
        U[2, 7:11] = U_212[0, :]  # A[0,2]
        U[5, 7:11] = U_212[1, :]  # A[1,2]
        V[6, 7:11] = V_212[0, :]  # B[0,2]
        V[7, 7:11] = V_212[1, :]  # B[1,2]
        W[0, 7:11] = W_212[0, :]  # C[0,2]
        W[1, 7:11] = W_212[1, :]  # C[1,2]
        W[2, 7:11] = W_212[2, :]  # C[2,0]
        W[3, 7:11] = W_212[3, :]  # C[2,1]
        return U, V, W

    # Build 2x3x7 as blocks: 2x3x2 + 2x3x2 + 2x3x2 + 2x3x1 = 11+11+11+6 = 39
    def build_2x3x7():
        U11_a, V11_a, W11_a = get_2x3x2_rank11()
        U11_b, V11_b, W11_b = get_2x3x2_rank11()
        U11_c, V11_c, W11_c = get_2x3x2_rank11()
        U6, V6, W6 = get_2x3x1_rank6()
        R_total = 39
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        # Block 1: B cols 0-1, C cols 0-1
        U_full[:, 0:11] = U11_a
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:11] = V11_a[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:11] = W11_a[k*2+i, :]
        # Block 2: B cols 2-3, C cols 2-3
        U_full[:, 11:22] = U11_b
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), 11:22] = V11_b[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, 11:22] = W11_b[k*2+i, :]
        # Block 3: B cols 4-5, C cols 4-5
        U_full[:, 22:33] = U11_c
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 22:33] = V11_c[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 22:33] = W11_c[k*2+i, :]
        # Block 4: B col 6, C col 6
        U_full[:, 33:39] = U6
        for j in range(3):
            V_full[j*7+6, 33:39] = V6[j, :]
        for i in range(2):
            W_full[6*2+i, 33:39] = W6[i, :]
        return U_full, V_full, W_full

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    def round_to_half_int(matrix):
        """Round values to nearest half-integer."""
        return np.round(matrix * 2) / 2

    def try_reduce_rank(U_init, V_init, W_init, target_rank, max_nfev=500):
        """Try to find a rank-reduced decomposition by zeroing some columns and optimizing."""
        if time.time() > deadline - 10:
            return None
        
        current_rank = U_init.shape[1]
        if target_rank >= current_rank:
            return None
        
        # Try different subsets of columns to keep
        # We'll try to remove (current_rank - target_rank) columns
        num_to_remove = current_rank - target_rank
        
        # Precompute target tensor
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try a few random subsets of columns to keep
        import itertools
        import random
        
        # For efficiency, try a limited number of subsets
        num_trials = min(20, max(1, int((deadline - 10 - time.time()) / 0.5)))
        
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Choose which columns to keep
            keep_cols = random.sample(range(current_rank), target_rank)
            keep_cols.sort()
            
            U_sub = U_init[:, keep_cols].copy()
            V_sub = V_init[:, keep_cols].copy()
            W_sub = W_init[:, keep_cols].copy()
            
            # Flatten for optimization
            U_flat = U_sub.flatten()
            V_flat = V_sub.flatten()
            W_flat = W_sub.flatten()
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            def residual(x):
                U_new = x[:N_U*target_rank].reshape(N_U, target_rank)
                V_new = x[N_U*target_rank:N_U*target_rank + N_V*target_rank].reshape(N_V, target_rank)
                W_new = x[N_U*target_rank + N_V*target_rank:].reshape(N_W, target_rank)
                T_calc = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                return (T_calc - T_target).flatten()
            
            try:
                result = least_squares(residual, x0, method='lm', max_nfev=max_nfev)
                
                if result.success or np.linalg.norm(result.fun) < 1e-8:
                    U_opt = result.x[:N_U*target_rank].reshape(N_U, target_rank)
                    V_opt = result.x[N_U*target_rank:N_U*target_rank + N_V*target_rank].reshape(N_V, target_rank)
                    W_opt = result.x[N_U*target_rank + N_V*target_rank:].reshape(N_W, target_rank)
                    
                    # Round to half-integers
                    U_rounded = round_to_half_int(U_opt)
                    V_rounded = round_to_half_int(V_opt)
                    W_rounded = round_to_half_int(W_opt)
                    
                    # Verify rounded solution
                    T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T_target, atol=1e-9):
                        return U_rounded, V_rounded, W_rounded
                        
            except Exception:
                continue
        
        return None

    def polish(U, V, W, rank):
        """High-precision polish pass with tighter tolerance."""
        if time.time() > deadline - 5:
            return U, V, W
        
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0

        def residual(x):
            U_new = x[:N_U*rank].reshape(N_U, rank)
            V_new = x[N_U*rank:N_U*rank + N_V*rank].reshape(N_V, rank)
            W_new = x[N_U*rank + N_V*rank:].reshape(N_W, rank)
            T_calc = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
            return (T_calc - T_target).flatten()

        try:
            result = least_squares(residual, x0, method='lm', max_nfev=200)
            if result.success and np.linalg.norm(result.fun) < 1e-10:
                U_polished = result.x[:N_U*rank].reshape(N_U, rank)
                V_polished = result.x[N_U*rank:N_U*rank + N_V*rank].reshape(N_V, rank)
                W_polished = result.x[N_U*rank + N_V*rank:].reshape(N_W, rank)
                return U_polished, V_polished, W_polished
        except Exception:
            pass
        return U, V, W

    # Build the rank-39 decomposition
    U_full, V_full, W_full = build_2x3x7()
    current_rank = 39
    
    # Apply polish pass
    U_full, V_full, W_full = polish(U_full, V_full, W_full, current_rank)
    
    if not verify(U_full, V_full, W_full):
        # Fallback to schoolbook
        return get_schoolbook()
    
    # Try to reduce rank from 39 down
    # Theoretical lower bound considerations:
    # - 2x3x7 has tensor rank at least max(2*3, 3*7, 7*2) = 21
    # - But practical lower bounds are higher
    # Try ranks 38, 37, 36, 35, 34, 33, 32, 31, 30
    best_solution = {
        "U": U_full.tolist(),
        "V": V_full.tolist(),
        "W": W_full.tolist()
    }
    
    # Try to reduce rank
    for target_rank in range(current_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_reduce_rank(U_full, V_full, W_full, target_rank, max_nfev=300)
        
        if result is not None:
            U_full, V_full, W_full = result
            current_rank = target_rank
            best_solution = {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            # Continue trying to reduce further
        else:
            # If we can't reduce from this rank, try one more step down
            # but don't spend too much time
            if time.time() > deadline - 20:
                break
    
    return best_solution
