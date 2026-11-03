"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import numpy as np
import time
import random
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition using float64."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k] * W[k*n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact_frac(U, V, W, n, m, p):
    """Verify exact decomposition using exact rational arithmetic."""
    Uf = [[Fraction(x).limit_denominator(10**6) for x in row] for row in U]
    Vf = [[Fraction(x).limit_denominator(10**6) for x in row] for row in V]
    Wf = [[Fraction(x).limit_denominator(10**6) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(len(Uf[0])):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _snap_to_half_int(A):
    """Snap array to nearest half-integer."""
    return np.round(A * 2) / 2.0

def _snap_to_int(A):
    """Snap array to nearest integer."""
    return np.round(A)

def _try_rank_exact(n, m, p, target_rank, time_budget, seed_offset=0):
    """Try to find exact rank decomposition with target_rank."""
    start = time.time()
    rng = np.random.default_rng(seed_offset + target_rank * 1000)
    
    # We'll use a structured approach:
    # 1. Random init with structured patterns
    # 2. Alternating minimization / gradient descent
    # 3. Snap to half-integers and verify
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_trials = 50
    steps_per_trial = 300
    
    for trial in range(num_trials):
        if time.time() - start > time_budget:
            break
        
        # Different initialization strategies
        if trial % 3 == 0:
            # Random small init
            U = rng.normal(0, 0.5, (n*m, target_rank))
            V = rng.normal(0, 0.5, (m*p, target_rank))
            W = rng.normal(0, 0.5, (p*n, target_rank))
        elif trial % 3 == 1:
            # Sparse init with some structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for c in range(target_rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            # Add noise
            U += rng.normal(0, 0.1, (n*m, target_rank))
            V += rng.normal(0, 0.1, (m*p, target_rank))
            W += rng.normal(0, 0.1, (p*n, target_rank))
        else:
            # Random binary-like init
            U = rng.choice([-1.0, 0.0, 1.0], (n*m, target_rank))
            V = rng.choice([-1.0, 0.0, 1.0], (m*p, target_rank))
            W = rng.choice([-1.0, 0.0, 1.0], (p*n, target_rank))
        
        # Gradient descent with adaptive learning rate
        lr = 0.05
        for step in range(steps_per_trial):
            if time.time() - start > time_budget:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m + j]
                        v_row = V[j*p + k]
                        w_row = W[k*n + i]
                        s = np.dot(u_row, v_row * w_row)
                        err_term = 2.0 * (s - 1.0)
                        dU[i*m + j] += err_term * v_row * w_row
                        dV[j*p + k] += err_term * u_row * w_row
                        dW[k*n + i] += err_term * u_row * v_row
            
            # Adaptive learning rate
            err = np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2)
            if err > 0:
                lr = min(0.1, max(0.001, 0.01 / np.sqrt(err)))
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check error periodically
            if step % 20 == 0:
                # Quick error check
                err_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err_val += (s - 1.0)**2
                if err_val < best_err:
                    best_err = err_val
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                # Try snapping if error is small
                if err_val < 1e-4:
                    U_snap = _snap_to_half_int(U)
                    V_snap = _snap_to_half_int(V)
                    W_snap = _snap_to_half_int(W)
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, 1e-8):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    
                    U_int = _snap_to_int(U)
                    V_int = _snap_to_int(V)
                    W_int = _snap_to_int(W)
                    if _verify_exact(U_int, V_int, W_int, n, m, p, 1e-8):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _try_alternating_min(n, m, p, target_rank, time_budget, seed_offset=0):
    """Alternating minimization approach."""
    start = time.time()
    rng = np.random.default_rng(seed_offset + 7777)
    
    for trial in range(20):
        if time.time() - start > time_budget:
            break
        
        U = rng.normal(0, 1.0, (n*m, target_rank))
        V = rng.normal(0, 1.0, (m*p, target_rank))
        W = rng.normal(0, 1.0, (p*n, target_rank))
        
        for it in range(200):
            if time.time() - start > time_budget:
                break
            
            # Fix V and W, optimize U
            for i in range(n):
                for j in range(m):
                    # Minimize sum_k ||U[i*m+j] - target_k||^2 where target depends on V,W
                    # Actually, for each (i,j), we want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                    # This is a linear system in U[i*m+j,:]
                    A = np.zeros((p, target_rank))
                    b = np.ones(p)
                    for k in range(p):
                        for c in range(target_rank):
                            A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve least squares
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i*m+j] = sol
                    except:
                        pass
            
            # Fix U and W, optimize V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, target_rank))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(target_rank):
                            A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j*p+k] = sol
                    except:
                        pass
            
            # Fix U and V, optimize W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, target_rank))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(target_rank):
                            A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k*n+i] = sol
                    except:
                        pass
            
            # Check error
            if it % 10 == 0:
                err_val = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                            err_val += (s - 1.0)**2
                if err_val < 1e-6:
                    U_snap = _snap_to_half_int(U)
                    V_snap = _snap_to_half_int(V)
                    W_snap = _snap_to_half_int(W)
                    if _verify_exact(U_snap, V_snap, W_snap, n, m, p, 1e-8):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                    U_int = _snap_to_int(U)
                    V_int = _snap_to_int(V)
                    W_int = _snap_to_int(W)
                    if _verify_exact(U_int, V_int, W_int, n, m, p, 1e-8):
                        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    schoolbook = _schoolbook_decomposition(n, m, p)
    schoolbook_rank = n * m * p  # 75
    
    # Try ranks from 74 down to ~55
    # For 3x5x5, known bounds: rank <= 75 (schoolbook), lower bounds exist
    # Let's try a range of ranks
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 25, 50), -1):
        if time.time() > deadline - 5.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        # Allocate time based on how many ranks left
        ranks_left = schoolbook_rank - target_rank
        time_per_rank = min(remaining / max(1, ranks_left), 20.0)
        
        # Try gradient-based search
        result = _try_rank_exact(n, m, p, target_rank, time_per_rank, seed_offset=0)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_exact(np.array(U), np.array(V), np.array(W), n, m, p, 1e-8):
                return {"U": U, "V": V, "W": W}
        
        # Try alternating minimization
        if time.time() < deadline - 5.0:
            result2 = _try_alternating_min(n, m, p, target_rank, min(10.0, deadline - time.time() - 5.0), seed_offset=42)
            if result2 is not None:
                U, V, W = result2
                if _verify_exact(np.array(U), np.array(V), np.array(W), n, m, p, 1e-8):
                    return {"U": U, "V": V, "W": W}
    
    return schoolbook

# EVOLVE-BLOCK-END
