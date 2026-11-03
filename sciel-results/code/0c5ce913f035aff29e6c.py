import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill

    size_U = n * m  # 6
    size_V = m * p  # 21
    size_W = p * n  # 14

    # Build the list of (a, b, c) triples
    # a = i*m + j, b = j*p + k, c = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_constraints = len(triples)  # 42

    # Precompute indices for Block Coordinate Descent
    # For W[c], we need (a, b) pairs
    indices_W = [[] for _ in range(size_W)]
    # For V[b], we need (a, c) pairs
    indices_V = [[] for _ in range(size_V)]
    # For U[a], we need (b, c) pairs
    indices_U = [[] for _ in range(size_U)]
    
    for a, b, c in triples:
        indices_W[c].append((a, b))
        indices_V[b].append((a, c))
        indices_U[a].append((b, c))

    def schoolbook():
        rank = n * m * p  # 42
        U = np.zeros((size_U, rank))
        V = np.zeros((size_V, rank))
        W = np.zeros((size_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb

    def compute_residual(U, V, W):
        """Compute residual sum of squares."""
        res = 0.0
        for a, b, c in triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            res += (val - 1.0) ** 2
        return res

    def verify(U, V, W, tol=1e-8):
        """Verify the decomposition is exact."""
        for (a, b, c) in triples:
            val = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(val - 1.0) > tol:
                return False
        return True

    def bcd_step(U, V, W, R):
        """
        Perform one step of Block Coordinate Descent.
        Updates W, then V, then U using least squares.
        """
        # Update W
        # For each c, solve A_c w = 1
        for c in range(size_W):
            pairs = indices_W[c]
            if not pairs:
                continue
            num_eqs = len(pairs)
            A_c = np.zeros((num_eqs, R))
            for idx, (a, b) in enumerate(pairs):
                A_c[idx, :] = U[a, :] * V[b, :]
            y = np.ones(num_eqs)
            # Solve least squares
            try:
                w_sol, residuals, rank_mat, sv = np.linalg.lstsq(A_c, y, rcond=None)
                W[c, :] = w_sol
            except np.linalg.LinAlgError:
                pass

        # Update V
        # For each b, solve A_b v = 1
        for b in range(size_V):
            pairs = indices_V[b]
            if not pairs:
                continue
            num_eqs = len(pairs)
            A_b = np.zeros((num_eqs, R))
            for idx, (a, c) in enumerate(pairs):
                A_b[idx, :] = U[a, :] * W[c, :]
            y = np.ones(num_eqs)
            try:
                v_sol, residuals, rank_mat, sv = np.linalg.lstsq(A_b, y, rcond=None)
                V[b, :] = v_sol
            except np.linalg.LinAlgError:
                pass

        # Update U
        # For each a, solve A_a u = 1
        for a in range(size_U):
            pairs = indices_U[a]
            if not pairs:
                continue
            num_eqs = len(pairs)
            A_a = np.zeros((num_eqs, R))
            for idx, (b, c) in enumerate(pairs):
                A_a[idx, :] = V[b, :] * W[c, :]
            y = np.ones(num_eqs)
            try:
                u_sol, residuals, rank_mat, sv = np.linalg.lstsq(A_a, y, rcond=None)
                U[a, :] = u_sol
            except np.linalg.LinAlgError:
                pass

    def try_rank_bcd(R, max_iter=500, max_restarts=50):
        """
        Try to find a decomposition of rank R using BCD.
        """
        if R >= best_rank:
            return None
            
        best_res = np.inf
        best_U_sol = None
        best_V_sol = None
        best_W_sol = None
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE - 5:
                return None
            
            # Initialize with random values
            # Scale is important. 1/sqrt(R) is a common choice.
            scale = 1.0 / np.sqrt(R)
            U = np.random.randn(size_U, R) * scale
            V = np.random.randn(size_V, R) * scale
            W = np.random.randn(size_W, R) * scale
            
            # Run BCD iterations
            for it in range(max_iter):
                if time.time() - start_time > DEADLINE - 5:
                    return None
                
                bcd_step(U, V, W, R)
                
                # Check residual periodically to save time
                if it % 10 == 0 or it == max_iter - 1:
                    res = compute_residual(U, V, W)
                    if res < best_res:
                        best_res = res
                        best_U_sol = U.copy()
                        best_V_sol = V.copy()
                        best_W_sol = W.copy
                        
                    if res < 1e-12:
                        return U.copy(), V.copy(), W.copy()
                        
        if best_U_sol is not None and best_res < 1e-10:
            if verify(best_U_sol, best_V_sol, best_W_sol):
                return best_U_sol, best_V_sol, best_W_sol
        return None

    # Try ranks from 41 down to 21
    # BCD is fast, so we can try more ranks or more restarts
    for R in range(rank_fb - 1, 20, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Adjust restarts based on rank
        # Lower ranks are harder, but BCD is fast.
        if R >= 35:
            max_restarts = 100
            max_iter = 200
        elif R >= 25:
            max_restarts = 50
            max_iter = 300
        else:
            max_restarts = 20
            max_iter = 500
            
        result = try_rank_bcd(R, max_restarts=max_restarts, max_iter=max_iter)
        
        if result is not None:
            U_try, V_try, W_try = result
            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, R
            # Continue searching for even lower rank
        else:
            # If failed, continue to lower rank
            pass

    # Final verification of best result
    if best_rank < rank_fb:
        if not verify(best_U, best_V, best_W, tol=1e-7):
            # Fallback to schoolbook
            best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
