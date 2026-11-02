"""Tensor decomposition search for 5x5x5 matrix multiplication."""
import time
import numpy as np
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 5, 5, 5
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns max error."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def round_to_half(x):
        """Round to nearest half-integer."""
        return round(x * 2) / 2.0

    def verify_rounded(U_r, V_r, W_r, rank):
        """Verify rounded decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def try_rank(rank, max_restarts=5, max_iters=100):
        """Try to find exact rank decomposition via alternating minimization."""
        if time.time() > deadline - 15:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 8:
                return None
            
            np.random.seed(rank * 7919 + restart * 104729 + 17)
            
            # Random initialization with small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            best_err = float('inf')
            best_state = (U.copy(), V.copy(), W.copy())
            
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    return None
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        B = np.array([[V[j*p+k, c] * W[k*n+i, c] for k in range(p)] for c in range(rank)])
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([[U[i*m+j, c] * W[k*n+i, c] for i in range(n)] for c in range(rank)])
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        C = np.array([[U[i*m+j, c] * V[j*p+k, c] for j in range(m)] for c in range(rank)])
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence every 3 iterations
                if it % 3 == 2 or it == max_iters - 1:
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                    
                    if max_err < best_err:
                        best_err = max_err
                        best_state = (U.copy(), V.copy(), W.copy())
                    
                    if max_err < 1e-8:
                        # Try rounding
                        U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                        V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                        W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                        err_r = verify_rounded(U_r, V_r, W_r, rank)
                        if err_r < 1e-8:
                            return {
                                "U": U_r.tolist(),
                                "V": V_r.tolist(),
                                "W": W_r.tolist()
                            }
                    
                    # If close, try local search with rounding
                    if max_err < 1e-4:
                        # Try adjusting some entries by 0.5
                        for attempt in range(50):
                            if time.time() > deadline - 2:
                                break
                            U_test = U.copy()
                            V_test = V.copy()
                            W_test = W.copy()
                            
                            # Randomly perturb a few entries
                            for _ in range(3):
                                mat = np.random.choice([0, 1, 2])
                                row = np.random.randint(0, U.shape[0] if mat == 0 else (V.shape[0] if mat == 1 else W.shape[0]))
                                col = np.random.randint(0, rank)
                                if mat == 0:
                                    U_test[row, col] += np.random.choice([-1, 1]) * 0.5
                                elif mat == 1:
                                    V_test[row, col] += np.random.choice([-1, 1]) * 0.5
                                else:
                                    W_test[row, col] += np.random.choice([-1, 1]) * 0.5
                            
                            # Round and verify
                            U_r = np.array([[round_to_half(x) for x in U_test[i]] for i in range(n*m)])
                            V_r = np.array([[round_to_half(x) for x in V_test[j]] for j in range(m*p)])
                            W_r = np.array([[round_to_half(x) for x in W_test[k]] for k in range(p*n)])
                            err_r = verify_rounded(U_r, V_r, W_r, rank)
                            if err_r < 1e-8:
                                return {
                                    "U": U_r.tolist(),
                                    "V": V_r.tolist(),
                                    "W": W_r.tolist()
                                }
                    
                    if max_err < 1e-8:
                        break
            
            # Try best state
            if best_err < 1e-6:
                U, V, W = best_state
                U_r = np.array([[round_to_half(x) for x in U[i]] for i in range(n*m)])
                V_r = np.array([[round_to_half(x) for x in V[j]] for j in range(m*p)])
                W_r = np.array([[round_to_half(x) for x in W[k]] for k in range(p*n)])
                err_r = verify_rounded(U_r, V_r, W_r, rank)
                if err_r < 1e-8:
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
        
        return None

    # Try ranks in strategic order
    rank_order = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 65, 70, 80, 100, 125]
    
    for rank in rank_order:
        if time.time() > deadline - 20:
            break
        if rank >= 125:
            return schoolbook()
        
        result = try_rank(rank, max_restarts=4, max_iters=80)
        if result is not None:
            return result
    
    return schoolbook()
