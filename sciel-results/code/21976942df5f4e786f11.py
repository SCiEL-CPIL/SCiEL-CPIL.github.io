import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = np.array(U[i * m + j])
                v_vec = np.array(V[j * p + k])
                w_vec = np.array(W[k * n + i])
                prod = np.sum(u_vec * v_vec * w_vec)
                if abs(prod - 1.0) > tol:
                    return False
    return True

def construct_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) -> 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen -> 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen -> 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def search_lower_rank(n, m, p, target_rank, time_budget=10, max_iter=5000):
    """
    Try to find a decomposition of the given rank using vectorized gradient descent with Adam.
    """
    start_time = time.time()
    
    # Precompute index arrays for vectorized operations
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    
    u_idx_arr = np.array(u_indices)
    v_idx_arr = np.array(v_indices)
    w_idx_arr = np.array(w_indices)
    
    def objective_and_grads(U, V, W):
        """Fully vectorized computation of objective and gradients."""
        # Get all needed rows
        U_sel = U[u_idx_arr]  # (30, rank)
        V_sel = V[v_idx_arr]  # (30, rank)
        W_sel = W[w_idx_arr]  # (30, rank)
        
        # Compute val for each triplet: sum over rank dimension
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (30,)
        errs = vals - 1.0  # (30,)
        obj = np.sum(errs * errs)
        
        # Gradients
        # d(obj)/dU[u_idx, r] = 2 * err * V_sel * W_sel
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # For each triplet t, add 2*errs[t] * V_sel[t] * W_sel[t] to grad_U[u_idx_arr[t]]
        # This requires scatter-add
        for t in range(len(u_idx_arr)):
            grad_U[u_idx_arr[t]] += 2 * errs[t] * V_sel[t] * W_sel[t]
            grad_V[v_idx_arr[t]] += 2 * errs[t] * U_sel[t] * W_sel[t]
            grad_W[w_idx_arr[t]] += 2 * errs[t] * U_sel[t] * V_sel[t]
        
        return obj, grad_U, grad_V, grad_W
    
    # Multiple random restarts
    best_obj = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization with varied scales
        scale = 0.3 + 0.4 * np.random.random()
        U = np.random.randn(n * m, target_rank) * scale
        V = np.random.randn(m * p, target_rank) * scale
        W = np.random.randn(p * n, target_rank) * scale
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        m_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_U = np.zeros_like(U)
        v_V = np.zeros_like(V)
        v_W = np.zeros_like(W)
        t = 0
        
        for it in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            t += 1
            obj, gU, gV, gW = objective_and_grads(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if obj < 1e-10:
                break
            
            # Adam update
            m_U = beta1 * m_U + (1 - beta1) * gU
            m_V = beta1 * m_V + (1 - beta1) * gV
            m_W = beta1 * m_W + (1 - beta1) * gW
            v_U = beta2 * v_U + (1 - beta2) * gU ** 2
            v_V = beta2 * v_V + (1 - beta2) * gV ** 2
            v_W = beta2 * v_W + (1 - beta2) * gW ** 2
            
            m_U_hat = m_U / (1 - beta1 ** t)
            m_V_hat = m_V / (1 - beta1 ** t)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            
            U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
            V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            # Slow decay
            lr *= 0.999999
    
    if best_obj < 1e-8 and best_U is not None:
        # Try to round to nice values
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        if verify_decomposition(n, m, p, 
                                [[float(x) for x in row] for row in U_rounded],
                                [[float(x) for x in row] for row in V_rounded],
                                [[float(x) for x in row] for row in W_rounded]):
            return {
                "U": [[float(x) for x in row] for row in U_rounded],
                "V": [[float(x) for x in row] for row in V_rounded],
                "W": [[float(x) for x in row] for row in W_rounded]
            }
        
        # Return unrounded if it's very close
        if best_obj < 1e-6:
            return {
                "U": [[float(x) for x in row] for row in best_U],
                "V": [[float(x) for x in row] for row in best_V],
                "W": [[float(x) for x in row] for row in best_W]
            }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with known rank 28 solution
    best_solution = construct_rank_28()
    best_rank = 28
    
    # Try to find rank 27
    if time.time() < deadline - 20:
        sol = search_lower_rank(n, m, p, 27, time_budget=25, max_iter=20000)
        if sol is not None:
            best_solution = sol
            best_rank = 27
    
    # Try rank 26 if time permits
    if best_rank > 26 and time.time() < deadline - 15:
        sol = search_lower_rank(n, m, p, 26, time_budget=15, max_iter=15000)
        if sol is not None:
            best_solution = sol
            best_rank = 26
    
    # Try rank 25 if time permits
    if best_rank > 25 and time.time() < deadline - 10:
        sol = search_lower_rank(n, m, p, 25, time_budget=10, max_iter=10000)
        if sol is not None:
            best_solution = sol
            best_rank = 25
    
    return {"U": best_solution["U"], "V": best_solution["V"], "W": best_solution["W"]}
