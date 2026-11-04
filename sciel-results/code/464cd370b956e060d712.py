import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    schoolbook_rank = n * m * p  # 36
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = schoolbook_rank
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
        return U, V, W, rank

    def try_rank_als(rank, max_iters=300, trials=3):
        """Try to find a decomposition of given rank using ALS with better initialization."""
        for trial in range(trials):
            if time.time() > deadline - 5:
                return None
                
            np.random.seed(2000 + trial * 7)
            
            # Initialize W with structured random values
            W = np.random.choice([-1.0, 0.0, 1.0], size=(p*n, rank))
            # Ensure each row has at least one non-zero
            for i in range(p*n):
                if all(W[i] == 0):
                    W[i][trial % rank] = 1.0
            
            # Initialize U and V with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            
            prev_error = float('inf')
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for V given U and W
                A_v = np.zeros((n*m*p, m*p*rank))
                b_v = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m*p, rank)
                
                # Solve for U given V and W
                A_u = np.zeros((n*m*p, n*m*rank))
                b_u = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n*m, rank)
                
                # Check error periodically
                if iteration % 5 == 0 or iteration == max_iters - 1:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        # Try rounding to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
                        # Also try integer rounding
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W_int.tolist()
                    
                    # Early stop if error is not improving
                    if max_err > prev_error * 1.01 and iteration > 50:
                        break
                    prev_error = max_err
                    
        return None

    def try_rank_fixed_W(rank, max_iters=200, num_W_configs=10):
        """Try fixed W patterns and solve for U, V."""
        for w_trial in range(num_W_configs):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(5000 + w_trial * 13)
            
            # Generate a fixed W pattern
            W = np.zeros((p*n, rank))
            for c in range(rank):
                # Assign random non-zero entries to each column
                num_nonzero = np.random.randint(1, max(2, rank//4) + 1)
                rows = np.random.choice(p*n, size=num_nonzero, replace=False)
                for r in rows:
                    W[r, c] = np.random.choice([-1.0, 1.0])
            
            # Check if W is valid (each row should have some structure)
            if np.all(W == 0):
                continue
            
            # Solve for U and V using least squares
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            
            for iteration in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for V
                A_v = np.zeros((n*m*p, m*p*rank))
                b_v = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    v_idx = (j*p + k) * rank + c
                                    A_v[eq_idx, v_idx] = U[i*m+j][c] * w_val
                            eq_idx += 1
                V_sol, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_sol.reshape(m*p, rank)
                
                # Solve for U
                A_u = np.zeros((n*m*p, n*m*rank))
                b_u = np.ones(n*m*p)
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(rank):
                                w_val = W[k*n+i][c]
                                if abs(w_val) > 1e-12:
                                    u_idx = (i*m + j) * rank + c
                                    A_u[eq_idx, u_idx] = V[j*p+k][c] * w_val
                            eq_idx += 1
                U_sol, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_sol.reshape(n*m, rank)
                
                # Check error
                if iteration % 10 == 0:
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(rank):
                                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                                max_err = max(max_err, abs(s - 1.0))
                    
                    if max_err < 1e-10:
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        if verify_decomposition(U_snap.tolist(), V_snap.tolist(), W.tolist(), rank):
                            return U_snap.tolist(), V_snap.tolist(), W.tolist()
                        
                        U_int = np.round(U)
                        V_int = np.round(V)
                        if verify_decomposition(U_int.tolist(), V_int.tolist(), W.tolist(), rank):
                            return U_int.tolist(), V_int.tolist(), W.tolist()
        
        return None

    # Try ranks from 25 to 35
    for rank in range(25, 36):
        if time.time() > deadline - 15:
            break
        
        # First try ALS with random initialization
        res = try_rank_als(rank, max_iters=200, trials=2)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
        
        # Then try fixed W patterns
        if time.time() > deadline - 10:
            break
        res = try_rank_fixed_W(rank, max_iters=100, num_W_configs=5)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
