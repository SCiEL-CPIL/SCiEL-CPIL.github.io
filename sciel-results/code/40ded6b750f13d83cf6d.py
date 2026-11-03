"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def make_schoolbook():
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

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy."""
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        U_t = Ua.reshape(n, m, rank)
        V_t = Va.reshape(m, p, rank)
        W_t = Wa.reshape(p, n, rank)
        
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            for i in range(n):
                U_i = U_t[i, :, c]
                V_c = V_t[:, :, c]
                W_ic = W_t[:, i, c]
                T[i, :, :] += (U_i[:, None] * V_c) * W_ic[None, :]
        
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def verify_exact(U, V, W):
        """Exact verification using fractions."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True

    def to_half_integer_matrix(M):
        """Convert a numpy matrix to a list of lists of floats that are half-integers."""
        # Try to snap values to nearest half-integer
        M_snapped = np.round(M * 2) / 2
        # Check if snapping is close enough
        if np.allclose(M_snapped, M, atol=1e-6):
            return [[float(x) for x in row] for row in M_snapped]
        else:
            return [[float(x) for x in row] for row in M]

    def try_random_rank(r, max_attempts=200):
        """Try to find a rank-r decomposition via random U,V and solving for W."""
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U and V with entries from a small set to encourage structure
            # Use {-1, 0, 1} or {-1, 1} or continuous
            if attempt % 3 == 0:
                U = np.random.choice([-1, 0, 1], size=(n * m, r)).astype(float)
                V = np.random.choice([-1, 0, 1], size=(m * p, r)).astype(float)
            elif attempt % 3 == 1:
                U = np.random.choice([-1, 1], size=(n * m, r)).astype(float)
                V = np.random.choice([-1, 1], size=(m * p, r)).astype(float)
            else:
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
            
            # Compute A_{ik,c} = sum_j U[i*m+j,c] * V[j*p+k,c]
            # Vectorize this computation
            A = np.zeros((n, p, r))
            for i in range(n):
                for k in range(p):
                    for c in range(r):
                        s = 0.0
                        for j in range(m):
                            s += U[i*m+j, c] * V[j*p+k, c]
                        A[i, k, c] = s
            
            # Now solve for W: W has shape (p*n, r)
            # For each (i,k): sum_c W[k*n+i, c] * A[i,k,c] = 1
            # This is 15 equations, 15r unknowns
            # Stack into a linear system: M * w = b
            
            M = np.zeros((n * p, p * n * r))
            for i in range(n):
                for k in range(p):
                    eq_idx = k * n + i
                    for c in range(r):
                        var_idx = (k * n + i) * r + c
                        M[eq_idx, var_idx] = A[i, k, c]
            
            b = np.ones(n * p)
            
            # Solve minimum norm solution
            try:
                w_sol, residuals, rank_M, sv = np.linalg.lstsq(M, b, rcond=None)
            except:
                continue
            
            # Check if the solution satisfies the equations
            if not np.allclose(M @ w_sol, b, atol=1e-9):
                continue
            
            # Reshape W
            W = w_sol.reshape(p * n, r)
            
            # Convert to lists and verify
            U_list = to_half_integer_matrix(U)
            V_list = to_half_integer_matrix(V)
            W_list = to_half_integer_matrix(W)
            
            if verify_fast(U_list, V_list, W_list, tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_grad_descent(r, max_iters=5000):
        """Try gradient descent to find a rank-r decomposition."""
        if time.time() > deadline:
            return None
        
        # Initialize with schoolbook-like structure
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        lr = 0.01
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            T = np.zeros((n, m, p), dtype=float)
            for c in range(r):
                for i in range(n):
                    U_i = U[i*m:(i+1)*m, c]  # (m,)
                    V_c = V.reshape(m, p, r)[:, :, c]  # (m, p)
                    W_ic = W.reshape(p, n, r)[:, i, c]  # (p,)
                    T[i, :, :] += (U_i[:, None] * V_c) * W_ic[None, :]
            
            loss = np.sum((T - 1.0)**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(r):
                for i in range(n):
                    U_i = U[i*m:(i+1)*m, c]
                    V_c = V.reshape(m, p, r)[:, :, c]
                    W_ic = W.reshape(p, n, r)[:, i, c]
                    term = (U_i[:, None] * V_c) * W_ic[None, :]
                    diff = term - 1.0/n*m*p  # approximate target
                    # dT/dU_i[j] = V_c[j,k] * W_ic[k] for each k
                    # dT/dV_c[j,k] = U_i[j] * W_ic[k]
                    # dT/dW_ic[k] = U_i[j] * V_c[j,k] for each j
                    
                    dU[i*m:(i+1)*m, c] += np.sum(diff * V_c * W_ic[None, :], axis=1)
                    dV.reshape(m, p, r)[:, :, c] += np.sum(diff * U_i[:, None] * W_ic[None, :], axis=0)
                    dW.reshape(p, n, r)[:, i, c] += np.sum(diff * U_i[:, None] * V_c, axis=1)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.5
            else:
                lr = min(lr * 1.05, 0.05)
        
        if best_loss < 1e-9:
            U_list = to_half_integer_matrix(best_U)
            V_list = to_half_integer_matrix(best_V)
            W_list = to_half_integer_matrix(best_W)
            if verify_fast(U_list, V_list, W_list, tol=1e-9):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Start with schoolbook
    best = make_schoolbook()
    best_rank = rank_schoolbook

    # Try decreasing ranks with random search
    for r in range(rank_schoolbook - 1, 15, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_random_rank(r, max_attempts=100)
        if result is not None:
            best = result
            best_rank = r
            break
    
    # If we have time, try gradient descent for a few more ranks
    if time.time() < deadline - 20:
        for r in range(min(best_rank - 1, 30), 15, -1):
            if time.time() > deadline - 5:
                break
            result = try_grad_descent(r, max_iters=2000)
            if result is not None:
                best = result
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
