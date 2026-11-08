import time
import sys
import random
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
    
    # Part 1: A2 B2 (j=2, k=0..4) -> 2*5=10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) -> 2*2=4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen -> 7 terms
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen -> 7 terms
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def snap_to_rational(U, V, W, denom=2):
    """Snap values to nearest multiple of 1/denom."""
    U_snap = np.round(U * denom) / denom
    V_snap = np.round(V * denom) / denom
    W_snap = np.round(W * denom) / denom
    return U_snap, V_snap, W_snap

def objective_smooth(U, V, W, n, m, p, rank):
    """Smooth objective: sum of squared errors."""
    obj = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                obj += (val - 1.0) ** 2
    return obj

def search_lower_rank(n, m, p, start_rank=27, end_rank=20, max_iter=200, time_budget=10):
    """
    Try to find a decomposition of rank < 28 using gradient-based search
    initialized from the rank-28 solution.
    """
    start_time = time.time()
    
    # Get the rank-28 solution as reference
    ref = construct_rank_28()
    ref_U = np.array(ref["U"], dtype=float)
    ref_V = np.array(ref["V"], dtype=float)
    ref_W = np.array(ref["W"], dtype=float)
    
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() - start_time > time_budget:
            break
        
        # Initialize by taking first 'rank' terms from rank-28 solution
        # and adding small random perturbation
        U = ref_U[:, :rank].copy()
        V = ref_V[:, :rank].copy()
        W = ref_W[:, :rank].copy()
        
        # Add small perturbation to escape local minima
        U += np.random.randn(n * m, rank) * 0.01
        V += np.random.randn(m * p, rank) * 0.01
        W += np.random.randn(p * n, rank) * 0.01
        
        # Gradient-based optimization
        lr = 0.01
        best_obj = objective_smooth(U, V, W, n, m, p, rank)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                        err = 2.0 * (val - 1.0)
                        dU[i * m + j] += err * V[j * p + k] * W[k * n + i]
                        dV[j * p + k] += err * U[i * m + j] * W[k * n + i]
                        dW[k * n + i] += err * U[i * m + j] * V[j * p + k]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            obj = objective_smooth(U, V, W, n, m, p, rank)
            
            if obj < best_obj:
                best_obj = obj
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_obj < 1e-10:
                # Found a near-exact solution, try to snap to rationals
                U_snap, V_snap, W_snap = snap_to_rational(best_U, best_V, best_W)
                obj_snap = objective_smooth(U_snap, V_snap, W_snap, n, m, p, rank)
                if obj_snap < 1e-9:
                    U_list = [[round(x, 4) for x in row] for row in U_snap]
                    V_list = [[round(x, 4) for x in row] for row in V_snap]
                    W_list = [[round(x, 4) for x in row] for row in W_snap]
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
                break
        
        # If we got close but not exact, try polishing
        if best_obj < 1e-6:
            U_snap, V_snap, W_snap = snap_to_rational(best_U, best_V, best_W)
            obj_snap = objective_smooth(U_snap, V_snap, W_snap, n, m, p, rank)
            if obj_snap < 1e-9:
                U_list = [[round(x, 4) for x in row] for row in U_snap]
                V_list = [[round(x, 4) for x in row] for row in V_snap]
                W_list = [[round(x, 4) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin
    
    # First, try the known rank 28 solution
    best_solution = construct_rank_28()
    best_rank = 28
    
    # Try to find a lower rank solution
    if time.time() < deadline - 30:
        lower_solution = search_lower_rank(n, m, p, start_rank=27, end_rank=22, max_iter=300, time_budget=20)
        if lower_solution is not None:
            U = lower_solution["U"]
            V = lower_solution["V"]
            W = lower_solution["W"]
            if verify_decomposition(n, m, p, U, V, W):
                best_solution = lower_solution
                best_rank = lower_solution["rank"]
    
    # If we still have time, try more restarts
    if time.time() < deadline - 10 and best_rank == 28:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            sol = search_lower_rank(n, m, p, start_rank=27, end_rank=27, max_iter=100, time_budget=5)
            if sol is not None:
                U = sol["U"]
                V = sol["V"]
                W = sol["W"]
                if verify_decomposition(n, m, p, U, V, W):
                    best_solution = sol
                    best_rank = sol["rank"]
                    break
    
    return {"U": best_solution["U"], "V": best_solution["V"], "W": best_solution["W"]}
