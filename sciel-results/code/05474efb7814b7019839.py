import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0]) if U else 0
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def _round_to_half_integer(arr):
    """Round array values to nearest half-integer (multiple of 0.5)."""
    return np.round(arr * 2) / 2

def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=5000):
    """Try to reduce rank to target_rank by finding linear combinations.
    Uses a greedy approach: try to express one term as a combination of others."""
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Build the full tensor from current decomposition
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    
    # The target tensor is: T[i*m+j, jp*p+k, kp*n+ip] = 1 iff i==ip, j==jp, k==kp
    # Otherwise 0
    
    # Strategy: try to find a linear dependency among the rank-1 terms
    # Term r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # If sum_r alpha_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0, we can eliminate terms
    
    # Alternative: try to merge two terms into one
    # If we can find r1, r2 and a scalar c such that:
    # U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = (1/c)*W[:,r2]
    # Then we can merge them
    
    # More practically: try random linear combinations and check if we can drop a term
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    current_rank = rank
    
    for it in range(max_iters):
        if current_rank <= target_rank:
            break
        
        # Pick a random term to try to eliminate
        r_elim = random.randint(0, current_rank - 1)
        
        # Try to express term r_elim as a linear combination of other terms
        # This is a hard problem, so use a heuristic:
        # Try swapping: replace term r_elim with a combination of two other terms
        
        if current_rank - r_elim < 2:
            continue
        
        r_a = random.randint(0, current_rank - 1)
        if r_a == r_elim:
            r_a = (r_a + 1) % current_rank
        r_b = random.randint(0, current_rank - 1)
        if r_b == r_elim or r_b == r_a:
            r_b = (r_b + 1) % current_rank
            if r_b == r_elim:
                r_b = (r_b + 1) % current_rank
        
        # Try to find coefficients alpha, beta such that:
        # U[:,r_elim] ≈ alpha*U[:,r_a] + beta*U[:,r_b]
        # V[:,r_elim] ≈ alpha*V[:,r_a] + beta*V[:,r_b]  
        # W[:,r_elim] ≈ alpha*W[:,r_a] + beta*W[:,r_b]
        
        # Solve least squares for U
        U_a = best_U[:, r_a]
        U_b = best_U[:, r_b]
        U_e = best_U[:, r_elim]
        
        V_a = best_V[:, r_a]
        V_b = best_V[:, r_b]
        V_e = best_V[:, r_elim]
        
        W_a = best_W[:, r_a]
        W_b = best_W[:, r_b]
        W_e = best_W[:, r_elim]
        
        # Solve for alpha, beta from U
        A_U = np.column_stack([U_a, U_b])
        if np.linalg.matrix_rank(A_U) < 2:
            continue
        try:
            coeff_U, _, _, _ = np.linalg.lstsq(A_U, U_e, rcond=None)
        except:
            continue
        alpha, beta = coeff_U
        
        # Check if these coefficients work for V and W
        V_pred = alpha * V_a + beta * V_b
        W_pred = alpha * W_a + beta * W_b
        
        if np.linalg.norm(V_pred - V_e) > 1e-6 or np.linalg.norm(W_pred - W_e) > 1e-6:
            continue
        
        # Great! We can eliminate term r_elim
        # Remove it by shifting columns
        new_U = np.delete(best_U, r_elim, axis=1)
        new_V = np.delete(best_V, r_elim, axis=1)
        new_W = np.delete(best_W, r_elim, axis=1)
        
        best_U, best_V, best_W = new_U, new_V, new_W
        current_rank -= 1
    
    if current_rank <= target_rank and _verify_exact(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    # Initialize with schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(dim_U)]
    V = [[0] * rank_schoolbook for _ in range(dim_V)]
    W = [[0] * rank_schoolbook for _ in range(dim_W)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Verify schoolbook is correct
    if not _verify_exact(U, V, W, n, m, p):
        # Should not happen, but return anyway
        return best_result
    
    # Try to reduce rank
    # Try target ranks from 23 down to some minimum
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # But actual minimum is likely higher
    
    for target_rank in range(rank_schoolbook - 1, 11, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for each target rank
        max_restarts = 5 if target_rank > 20 else 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Start from schoolbook and try to reduce
            result = _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=2000)
            if result is not None:
                U_r, V_r, W_r = result
                if _verify_exact(U_r, V_r, W_r, n, m, p):
                    actual_rank = len(U_r[0])
                    if actual_rank < best_rank:
                        best_rank = actual_rank
                        best_result = {"U": U_r, "V": V_r, "W": W_r}
                    break  # Found a valid decomposition at this rank
        
        # If we found a rank better than target_rank, continue to lower targets
        if best_rank <= target_rank:
            continue
        else:
            # Try a few more iterations
            pass
    
    # Final verification
    if _verify_exact(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    else:
        # Fall back to schoolbook
        return {"U": U, "V": V, "W": W}
