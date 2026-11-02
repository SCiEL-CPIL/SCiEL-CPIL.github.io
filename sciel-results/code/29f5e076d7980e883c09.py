"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact using exact arithmetic.
    """
    rank = len(U[0]) if U else 0
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i][c]) for c in range(rank)] for i in range(n * m)]
    V_frac = [[Fraction(V[j][c]) for c in range(rank)] for j in range(m * p)]
    W_frac = [[Fraction(W[k][c]) for c in range(rank)] for k in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = Fraction(0)
                for c in range(rank):
                    s += U_frac[ui][c] * V_frac[vj][c] * W_frac[wk][c]
                if s != 1:
                    return False
    return True


def _verify_exact_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float verification for pruning."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[ui][c] * V[vj][c] * W[wk][c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_structured_search(n, m, p, deadline):
    """
    Try to find a low-rank decomposition using structured approaches.
    """
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    best_rank = n * m * p
    
    # Try various rank values from low to high
    # Lower bound: for matrix mult, rank >= max(n*m, m*p, p*n) in some cases
    # For 3x4x7: lower bound is at least max(12, 28, 21) = 28
    # But practical lower bounds are higher
    
    # Try rank values that might work
    candidate_ranks = list(range(28, 84))
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with small random values
            Ur = np.random.randn(nm, target_rank) * 0.1
            Vr = np.random.randn(mp, target_rank) * 0.1
            Wr = np.random.randn(pn, target_rank) * 0.1
            
            # Use iterative optimization
            success = False
            for iter in range(100):
                if time.time() > deadline:
                    break
                    
                # Compute current tensor reconstruction
                S = np.zeros((nm, mp, pn))
                for r in range(target_rank):
                    S += np.outer(np.outer(Ur[:, r], Vr[:, r]), Wr[:, r]).reshape(nm, mp, pn)
                
                # Check if we're close
                error = np.sum((S - T) ** 2)
                if error < 1e-12:
                    success = True
                    break
                
                # Gradient computation (simplified)
                # For each factor, compute gradient and update
                for r in range(target_rank):
                    # Gradient for Ur[:, r]
                    grad_U = np.zeros(nm)
                    grad_V = np.zeros(mp)
                    grad_W = np.zeros(pn)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_u = i*m+j
                                idx_v = j*p+k
                                idx_w = k*n+i
                                diff = S[idx_u, idx_v, idx_w] - T[idx_u, idx_v, idx_w]
                                grad_U[idx_u] += diff * Vr[idx_v, r] * Wr[idx_w, r]
                                grad_V[idx_v] += diff * Ur[idx_u, r] * Wr[idx_w, r]
                                grad_W[idx_w] += diff * Ur[idx_u, r] * Vr[idx_v, r]
                    
                    # Update with small step
                    lr = 0.01 / (1 + iter * 0.1)
                    Ur[:, r] -= lr * grad_U
                    Vr[:, r] -= lr * grad_V
                    Wr[:, r] -= lr * grad_W
            
            if success:
                # Verify with exact arithmetic
                U_list = [[float(Ur[i][c]) for c in range(target_rank)] for i in range(nm)]
                V_list = [[float(Vr[j][c]) for c in range(target_rank)] for j in range(mp)]
                W_list = [[float(Wr[k][c]) for c in range(target_rank)] for k in range(pn)]
                
                if _verify_exact_float(U_list, V_list, W_list, n, m, p):
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        best_rank = target_rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    return None


def _try_local_search(n, m, p, deadline):
    """
    Start from schoolbook and try to merge terms to reduce rank.
    """
    start = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Get schoolbook
    school = _schoolbook(n, m, p)
    U = np.array(school["U"], dtype=float)
    V = np.array(school["V"], dtype=float)
    W = np.array(school["W"], dtype=float)
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce by combining similar terms
    # This is heuristic - try random subsets of columns
    for target_rank in range(best_rank - 1, 60, -1):
        if time.time() > deadline:
            break
        
        # Try to find target_rank columns that span the tensor
        # Use random projection approach
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Randomly select target_rank columns and try to fit
            cols = np.random.choice(best_rank, target_rank, replace=False)
            Ur = best_U[:, cols].copy()
            Vr = best_V[:, cols].copy()
            Wr = best_W[:, cols].copy()
            
            # Try to optimize
            for iter in range(50):
                if time.time() > deadline:
                    break
                
                # Compute error
                S = np.zeros((nm, mp, pn))
                for r in range(target_rank):
                    S += np.outer(np.outer(Ur[:, r], Vr[:, r]), Wr[:, r]).reshape(nm, mp, pn)
                
                error = np.sum((S - 1.0) ** 2)  # Target is 1 for non-zero entries
                if error < 1e-10:
                    # Verify
                    U_list = [[float(Ur[i][c]) for c in range(target_rank)] for i in range(nm)]
                    V_list = [[float(Vr[j][c]) for c in range(target_rank)] for j in range(mp)]
                    W_list = [[float(Wr[k][c]) for c in range(target_rank)] for k in range(pn)]
                    
                    if _verify_exact_float(U_list, V_list, W_list, n, m, p):
                        if _verify_exact(U_list, V_list, W_list, n, m, p):
                            best_rank = target_rank
                            best_U, best_V, best_W = Ur, Vr, Wr
                            break
                else:
                    # Simple gradient step
                    lr = 0.001
                    for r in range(target_rank):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    idx_u = i*m+j
                                    idx_v = j*p+k
                                    idx_w = k*n+i
                                    diff = S[idx_u, idx_v, idx_w] - 1.0
                                    Ur[idx_u, r] -= lr * diff * Vr[idx_v, r] * Wr[idx_w, r]
                                    Vr[idx_v, r] -= lr * diff * Ur[idx_u, r] * Wr[idx_w, r]
                                    Wr[idx_w, r] -= lr * diff * Ur[idx_u, r] * Vr[idx_v, r]
            
            if best_rank == target_rank:
                break
        
        if best_rank == target_rank:
            break
    
    if best_rank < n * m * p:
        U_list = [[float(best_U[i][c]) for c in range(best_rank)] for i in range(nm)]
        V_list = [[float(best_V[j][c]) for c in range(best_rank)] for j in range(mp)]
        W_list = [[float(best_W[k][c]) for c in range(best_rank)] for k in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill

    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try structured search
    try:
        improved = _try_structured_search(n, m, p, deadline)
        if improved:
            r = len(improved["U"][0])
            if r < best_rank:
                best_rank = r
                result = improved
    except Exception:
        pass

    # Try local search if time remains
    if time.time() < deadline - 10:
        try:
            improved2 = _try_local_search(n, m, p, deadline)
            if improved2:
                r = len(improved2["U"][0])
                if r < best_rank:
                    best_rank = r
                    result = improved2
        except Exception:
            pass

    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
