import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matmul."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are integers or half-integers."""
    for val in arr:
        doubled = 2.0 * val
        if abs(doubled - round(doubled)) > tol:
            return False
    return True

def _snap_to_half_int(arr, tol=1e-8):
    """Snap values to nearest half-integer if close."""
    result = []
    for val in arr:
        doubled = 2.0 * val
        nearest = round(doubled)
        if abs(doubled - nearest) < tol:
            result.append(nearest / 2.0)
        else:
            result.append(val)
    return result

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vals = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                s = np.sum(vals)
                error += (s - 1.0) ** 2
    return error

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W, rank

def _try_rank_r(r, n, m, p, max_iters=500, seed=42, deadline=None):
    """
    Try to find a decomposition of rank r using gradient descent 
    followed by half-integer snapping and local refinement.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    
    np.random.seed(seed)
    
    best_error = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    # Multiple restarts
    for restart in range(10):
        if deadline and time.time() > deadline - 5:
            break
            
        # Initialize with small random values
        U = np.random.randn(nm, r) * 0.5
        V = np.random.randn(mp, r) * 0.5
        W = np.random.randn(pn, r) * 0.5
        
        current_error = _compute_error(U, V, W, n, m, p)
        
        # Gradient Descent
        lr = 0.05
        for it in range(max_iters):
            if deadline and time.time() > deadline - 5:
                break
                
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        vals = U[u_idx, :] * V[v_idx, :] * W[w_idx, :]
                        s = np.sum(vals)
                        err = s - 1.0
                        
                        # d/dU = err * V * W
                        grad_U[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                        grad_V[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                        grad_W[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
                        
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_error = _compute_error(U, V, W, n, m, p)
            
            # Simple adaptive learning rate
            if new_error > current_error * 1.01:
                lr *= 0.5
                U += 0.5 * lr * grad_U
                V += 0.5 * lr * grad_V
                W += 0.5 * lr * grad_W
                new_error = _compute_error(U, V, W, n, m, p)
            else:
                # Occasionally increase lr if stuck? Keep simple for now
                pass
            
            current_error = new_error
            
            if current_error < best_error:
                best_error = current_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            if current_error < 1e-10:
                break
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-8:
        # Snap to half-integers
        best_U = np.array([_snap_to_half_int(row) for row in best_U])
        best_V = np.array([_snap_to_half_int(row) for row in best_V])
        best_W = np.array([_snap_to_half_int(row) for row in best_W])
        
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            if _is_half_integer(best_U) and _is_half_integer(best_V) and _is_half_integer(best_W):
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        # If snapping failed, try local search around the continuous solution
        # Convert to integers (doubled) for discrete search
        U_int = (2 * best_U).astype(int)
        V_int = (2 * best_V).astype(int)
        W_int = (2 * best_W).astype(int)
        
        # Perform a few steps of local search in the integer grid
        for step in range(20):
            if deadline and time.time() > deadline - 2:
                break
            improved = False
            # Try perturbing a random entry by +/- 1 (in doubled space, i.e. +/- 0.5)
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                idx = np.random.randint(nm)
                col = np.random.randint(r)
                old_val = U_int[idx, col]
                for delta in [-1, 1]:
                    U_int[idx, col] = old_val + delta
                    U_check = U_int / 2.0
                    if _verify_decomposition(U_check, best_V, best_W, n, m, p):
                        improved = True
                        break
                    U_int[idx, col] = old_val
            elif mat_choice == 1:
                idx = np.random.randint(mp)
                col = np.random.randint(r)
                old_val = V_int[idx, col]
                for delta in [-1, 1]:
                    V_int[idx, col] = old_val + delta
                    V_check = V_int / 2.0
                    if _verify_decomposition(best_U, V_check, best_W, n, m, p):
                        improved = True
                        break
                    V_int[idx, col] = old_val
            else:
                idx = np.random.randint(pn)
                col = np.random.randint(r)
                old_val = W_int[idx, col]
                for delta in [-1, 1]:
                    W_int[idx, col] = old_val + delta
                    W_check = W_int / 2.0
                    if _verify_decomposition(best_U, best_V, W_check, n, m, p):
                        improved = True
                        break
                    W_int[idx, col] = old_val
            
            if improved:
                best_U = U_int / 2.0
                best_V = V_int / 2.0
                best_W = W_int / 2.0
                if _verify_decomposition(best_U, best_V, best_W, n, m, p):
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
            else:
                break

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    U_fb, V_fb, W_fb, rank_schoolbook = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Start from the theoretical lower bound
    lower_bound = max(nm, mp, pn)
    
    # We prioritize lower ranks
    for r in range(lower_bound, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_r(r, n, m, p, max_iters=200, seed=42 + r, deadline=deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Return schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
