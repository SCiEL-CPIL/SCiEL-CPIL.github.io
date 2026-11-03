"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    # T[a, b, c] = 1 if indices match multiplication term
    # a = k*n + i, b = i*m + j, c = j*p + k
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0

    def check_solution(U, V, W):
        """Check if U, V, W exactly decompose T."""
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        return np.allclose(T_rec, T, atol=1e-10)

    def try_round_and_check(U, V, W, scale=2):
        """Try to round U, V, W to nearest 1/scale and check."""
        U_r = np.round(U * scale) / scale
        V_r = np.round(V * scale) / scale
        W_r = np.round(W * scale) / scale
        
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U_r.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W_r[:, rr], U_r[:, rr], V_r[:, rr])
        
        if np.allclose(T_rec, T, atol=1e-10):
            return U_r, V_r, W_r
        return None

    # Try to find low rank decompositions
    # Theoretical min rank for 2x3x4 is 19 (based on known results for small matrix multiplication)
    # Schoolbook is 24. Let's try 19 to 23.
    
    best_result = None
    best_rank = 24
    
    # Strategy: Use ALS (Alternating Least Squares) to find a decomposition
    # We'll iterate over ranks from 19 to 23.
    # For each rank, we'll do a few random restarts.
    
    max_restarts = 5
    max_iters = 20
    
    for r in range(19, 24):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize U, V, W randomly
            # Use small random values to start
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # Normalize to avoid overflow/underflow issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            converged = False
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                for b in range(dim_U):
                    X = np.zeros((dim_W * dim_V, r))
                    y = T[:, b, :].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], V[:, rr])
                    
                    try:
                        u_b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        U[b, :] = u_b
                    except:
                        pass
                        
                # Update V
                for c in range(dim_V):
                    X = np.zeros((dim_W * dim_U, r))
                    y = T[:, :, c].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], U[:, rr])
                    
                    try:
                        v_c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        V[c, :] = v_c
                    except:
                        pass
                        
                # Update W
                for a in range(dim_W):
                    X = np.zeros((dim_U * dim_V, r))
                    y = T[a, :, :].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(U[:, rr], V[:, rr])
                    
                    try:
                        w_a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        W[a, :] = w_a
                    except:
                        pass
                
                # Check convergence
                T_rec = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
                
                error = np.linalg.norm(T - T_rec)
                if error < 1e-8:
                    converged = True
                    break
                    
            if converged:
                # Try to round to nearest 0.5
                rounded = try_round_and_check(U, V, W, scale=2)
                if rounded is not None:
                    U_r, V_r, W_r = rounded
                    best_result = (U_r, V_r, W_r, r)
                    found = True
                    break
                else:
                    # If rounding fails, check if float solution is "exact" enough
                    # The prompt asks for integer/half-integer, so we might need to be strict.
                    # However, if the evaluator allows small tolerance, we could return floats.
                    # But the prompt says "exactly decompose".
                    # Let's try rounding to integer (scale=1) as well, just in case.
                    rounded_int = try_round_and_check(U, V, W, scale=1)
                    if rounded_int is not None:
                        U_r, V_r, W_r = rounded_int
                        best_result = (U_r, V_r, W_r, r)
                        found = True
                        break
                    else:
                        # If no rounded solution, we might still return the float solution
                        # if it's very close, but the prompt is strict.
                        # Let's assume we need exact integer/half-integer.
                        # If we can't round, we skip this rank.
                        pass

        if found:
            break

    if best_result is not None:
        U_r, V_r, W_r, r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
