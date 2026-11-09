"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=3, m=3, p=8, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _round_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2

def _solve_W(U, V, n, m, p):
    """Solve for W given U and V using least squares."""
    R = U.shape[1]
    W = np.zeros((p * n, R))
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, R))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            if A.size > 0:
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k*n+i, :] = w
    return W

def _solve_V(U, W, n, m, p):
    """Solve for V given U and W using least squares."""
    R = U.shape[1]
    V = np.zeros((m * p, R))
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, R))
            for i in range(n):
                A[i, :] = U[i*m+j, :] * W[k*n+i, :]
            if A.size > 0:
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V[j*p+k, :] = v
    return V

def _solve_U(V, W, n, m, p):
    """Solve for U given V and W using least squares."""
    R = V.shape[1]
    U = np.zeros((n * m, R))
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, R))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            if A.size > 0:
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j, :] = u
    return U

def _alternating_minimize(U, V, W, n, m, p, max_iter=50, tol=1e-9, deadline=None):
    """Alternating minimization with half-integer rounding."""
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        W = _solve_W(U, V, n, m, p)
        V = _solve_V(U, W, n, m, p)
        U = _solve_U(V, W, n, m, p)
        
        U_r = _round_half(U)
        V_r = _round_half(V)
        W_r = _round_half(W)
        
        if _validate(U_r, V_r, W_r, n, m, p, tol):
            return U_r, V_r, W_r, True
    return U_r, V_r, W_r, False

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Leave 5s margin before 120s limit
    
    # Start with schoolbook rank 72
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Strategy 1: Try to reduce rank by merging schoolbook columns
    # For each pair of columns that share structure, try to merge
    # Key insight: for fixed i, the columns (i,j,k) for different j,k
    # might be mergeable if we use linear combinations
    
    # Try greedy elimination: remove one column at a time, re-solve
    current_rank = rank
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    
    for col_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Try removing column col_to_remove
        mask = np.ones(current_rank, dtype=bool)
        mask[col_to_remove] = False
        U_trial = U_cur[:, mask]
        V_trial = V_cur[:, mask]
        W_trial = W_cur[:, mask]
        
        # Refine with alternating minimization
        U_r, V_r, W_r, success = _alternating_minimize(
            U_trial, V_trial, W_trial, n, m, p, 
            max_iter=30, tol=1e-9, deadline=deadline
        )
        
        if success:
            U_cur, V_cur, W_cur = U_r, V_r, W_r
            current_rank -= 1
            if current_rank < best_rank:
                best_rank = current_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
    
    # Strategy 2: Try lower ranks from scratch with structured initialization
    # For 3x3 x 3x8, try ranks from 64 down
    for target_rank in range(min(64, best_rank - 1), 40, -1):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            random.seed(restart * 1000 + target_rank)
            np.random.seed(restart * 1000 + target_rank)
            
            # Structured initialization: start with schoolbook and randomly
            # zero out some entries, then refine
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            # Initialize with random sparse pattern based on schoolbook
            for c in range(target_rank):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                U_init[i * m + j, c] = 1.0
                V_init[j * p + k, c] = 1.0
                W_init[k * n + i, c] = 1.0
                
                # Add some shared structure
                if random.random() < 0.4:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    U_init[i2 * m + j2, c] += random.choice([1.0, -1.0])
                    V_init[j2 * p + k2, c] += random.choice([1.0, -1.0])
                    W_init[k2 * n + i2, c] += random.choice([1.0, -1.0])
            
            # Alternating minimization
            U_r, V_r, W_r, success = _alternating_minimize(
                U_init, V_init, W_init, n, m, p,
                max_iter=100, tol=1e-9, deadline=deadline
            )
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    # Final polish: try to squeeze out more rank reduction
    if time.time() < deadline - 10:
        current_rank = best_rank
        U_cur, V_cur, W_cur = best_U.copy(), best_V.copy(), best_W.copy()
        
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for col_to_remove in range(current_rank):
                if time.time() > deadline - 5:
                    break
                
                mask = np.ones(current_rank, dtype=bool)
                mask[col_to_remove] = False
                U_trial = U_cur[:, mask]
                V_trial = V_cur[:, mask]
                W_trial = W_cur[:, mask]
                
                U_r, V_r, W_r, success = _alternating_minimize(
                    U_trial, V_trial, W_trial, n, m, p,
                    max_iter=20, tol=1e-9, deadline=deadline
                )
                
                if success:
                    U_cur, V_cur, W_cur = U_r, V_r, W_r
                    current_rank -= 1
                    improved = True
                    break
            
            if not improved:
                break
        
        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_r.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
