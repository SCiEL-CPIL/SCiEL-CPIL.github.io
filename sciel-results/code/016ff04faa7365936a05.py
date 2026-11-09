"""
Rank-minimal tensor decomposition for 3x5x7 matrix multiplication.
Uses structured initialization from schoolbook, column merging heuristics,
and alternating optimization with half-integer rounding.
"""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N


def _verify_halfint(U, V, W, n, m, p, tol=1e-6):
    """Verify exact decomposition with half-integer entries."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _loss(U, V, W, n, m, p):
    """Compute squared loss for decomposition."""
    rank = U.shape[1]
    loss = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                val = np.dot(u_row, V[j * p + k] * W[k * n + i])
                loss += (val - 1.0) ** 2
    return loss


def _als_step(U, V, W, n, m, p):
    """One ALS iteration: solve for each factor given the others."""
    rank = U.shape[1]
    
    # Solve for U
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k] = V[j * p + k] * W[k * n + i]
            b = np.ones(p)
            try:
                U[i * m + j] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
    
    # Solve for V
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, rank))
            for i in range(n):
                A[i] = U[i * m + j] * W[k * n + i]
            b = np.ones(n)
            try:
                V[j * p + k] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass
    
    # Solve for W
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, rank))
            for j in range(m):
                A[j] = U[i * m + j] * V[j * p + k]
            b = np.ones(m)
            try:
                W[k * n + i] = np.linalg.lstsq(A, b, rcond=None)[0]
            except:
                pass


def _grad_step(U, V, W, n, m, p, lr=0.01):
    """One gradient descent step."""
    rank = U.shape[1]
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            for k in range(p):
                v_row = V[j * p + k]
                w_row = W[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                err = val - 1.0
                grad_U[i * m + j] += err * (v_row * w_row)
                grad_V[j * p + k] += err * (u_row * w_row)
                grad_W[k * n + i] += err * (u_row * v_row)
    
    U -= lr * grad_U
    V -= lr * grad_V
    W -= lr * grad_W


def _round_halfint(arr):
    """Round to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _build_schoolbook(n, m, p):
    """Build the rank-105 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_rank_from_schoolbook(target_rank, n, m, p, deadline):
    """
    Try to find a rank-target_rank decomposition by merging schoolbook columns.
    Uses structured random initialization based on schoolbook structure.
    """
    if time.time() > deadline:
        return None
    
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    school_rank = n * m * p
    
    best_result = None
    
    # Strategy 1: Random column merging with ALS refinement
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(attempt * 7919 + target_rank)
        
        # Create random mapping from schoolbook columns to target columns
        # Each schoolbook column gets assigned to one of target_rank columns
        col_map = rng.randint(0, target_rank, size=school_rank)
        
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        for c in range(school_rank):
            tc = col_map[c]
            U[:, tc] += U_sb[:, c]
            V[:, tc] += V_sb[:, c]
            W[:, tc] += W_sb[:, c]
        
        # Add small random perturbation to escape degeneracy
        U += rng.randn(n * m, target_rank) * 0.01
        V += rng.randn(m * p, target_rank) * 0.01
        W += rng.randn(p * n, target_rank) * 0.01
        
        # ALS refinement
        for it in range(30):
            if time.time() > deadline - 5:
                break
            _als_step(U, V, W, n, m, p)
            loss = _loss(U, V, W, n, m, p)
            if loss < 1e-10:
                break
        
        # Gradient polish
        for it in range(50):
            if time.time() > deadline - 5:
                break
            _grad_step(U, V, W, n, m, p, lr=0.005)
            loss = _loss(U, V, W, n, m, p)
            if loss < 1e-10:
                break
        
        loss = _loss(U, V, W, n, m, p)
        if loss < 1e-6:
            # Round to half-integers and verify
            U_h = _round_halfint(U)
            V_h = _round_halfint(V)
            W_h = _round_halfint(W)
            
            U_list = U_h.tolist()
            V_list = V_h.tolist()
            W_list = W_h.tolist()
            
            if _verify_halfint(U_list, V_list, W_list, n, m, p, tol=1e-4):
                return U_list, V_list, W_list
    
    return None


def _try_random_rank(target_rank, n, m, p, deadline):
    """Try random initialization for a given target rank."""
    if time.time() > deadline:
        return None
    
    for attempt in range(15):
        if time.time() > deadline - 5:
            break
        
        rng = np.random.RandomState(attempt * 104729 + target_rank * 31)
        
        # Mix of initialization strategies
        if attempt % 3 == 0:
            # Dense random
            U = rng.randn(n * m, target_rank) * 0.3
            V = rng.randn(m * p, target_rank) * 0.3
            W = rng.randn(p * n, target_rank) * 0.3
        elif attempt % 3 == 1:
            # Sparse random
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            U[rng.rand(n * m, target_rank) < 0.4] = 0
            V[rng.rand(m * p, target_rank) < 0.4] = 0
            W[rng.rand(p * n, target_rank) < 0.4] = 0
        else:
            # Half-integer random
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(n * m)])
            V = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(m * p)])
            W = np.array([[rng.choice(vals) for _ in range(target_rank)] for _ in range(p * n)])
        
        # ALS
        for it in range(40):
            if time.time() > deadline - 5:
                break
            _als_step(U, V, W, n, m, p)
            loss = _loss(U, V, W, n, m, p)
            if loss < 1e-10:
                break
        
        # Gradient
        for it in range(80):
            if time.time() > deadline - 5:
                break
            _grad_step(U, V, W, n, m, p, lr=0.003)
            loss = _loss(U, V, W, n, m, p)
            if loss < 1e-10:
                break
        
        loss = _loss(U, V, W, n, m, p)
        if loss < 1e-6:
            U_h = _round_halfint(U)
            V_h = _round_halfint(V)
            W_h = _round_halfint(W)
            
            U_list = U_h.tolist()
            V_list = V_h.tolist()
            W_list = W_h.tolist()
            
            if _verify_halfint(U_list, V_list, W_list, n, m, p, tol=1e-4):
                return U_list, V_list, W_list
    
    return None


def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    
    # Try ranks from just below schoolbook downward
    # Focus on the most likely improvements first
    candidate_ranks = list(range(best_rank - 1, max(best_rank - 20, 80), -1))
    
    for r in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # First try structured column merging (more likely to succeed near schoolbook)
        result = _try_rank_from_schoolbook(r, n, m, p, deadline)
        
        if result is None:
            # Then try random initialization
            result = _try_random_rank(r, n, m, p, deadline)
        
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {"U": U, "V": V, "W": W}
            
            # Continue trying lower ranks
            if best_rank > 80:
                candidate_ranks = list(range(best_rank - 1, max(best_rank - 15, 80), -1))
            else:
                break
    
    return best_result
