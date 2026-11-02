import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]
_SEARCH_TIMEOUT = 110.0  # 10s margin before 120s limit

def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact and entries are half-integers."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    
    if not np.allclose(T_calc, np.ones((n, m, p)), atol=tol):
        return False
        
    def is_half_int(arr):
        return np.allclose(arr * 2, np.round(arr * 2), atol=1e-9)
        
    return is_half_int(U) and is_half_int(V) and is_half_int(W)

def _residual(U, V, W, n, m, p):
    """Compute the residual norm."""
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    W_perm = np.transpose(W3, (1, 0, 2))
    T_calc = np.einsum('ijr,jkr,ikr->ijk', U3, V3, W_perm, optimize=True)
    return np.linalg.norm(T_calc - 1.0)

def _schoolbook():
    """Return the schoolbook decomposition with rank 36."""
    n, m, p = DIMENSIONS
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
    return U, V, W

def _try_rank_structured(n, m, p, R, deadline, seed=42):
    """
    Try to find a rank-R decomposition using structured search.
    Works in the space of half-integers directly.
    """
    np.random.seed(seed)
    
    # Half-integer values to try: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    # But we can be smarter: since we need sum of products = 1, 
    # and each product is a multiple of 0.125, we need reasonable bounds.
    # Let's try values in {-1, -0.5, 0, 0.5, 1} first, then expand if needed.
    
    half_int_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    
    def random_half_int_matrix(rows, cols, density=0.3):
        """Generate a random matrix with half-integer entries, mostly zeros."""
        mat = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.rand() < density:
                    mat[i, j] = np.random.choice(half_int_values)
        return mat
    
    # Try multiple random initializations
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    max_iters = 2000
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Initialize
        U = random_half_int_matrix(n*m, R, density=0.25)
        V = random_half_int_matrix(m*p, R, density=0.25)
        W = random_half_int_matrix(p*n, R, density=0.25)
        
        res = _residual(U, V, W, n, m, p)
        
        # Local search: try changing one entry at a time
        for step in range(100):
            if time.time() > deadline:
                break
            if res < 1e-9:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
                break
                
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Pick a random matrix and entry to modify
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U; rows = n*m
            elif mat_choice == 1:
                mat = V; rows = m*p
            else:
                mat = W; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            old_val = mat[row, col]
            
            # Try all half-integer values in a small range
            candidates = []
            for v in half_int_values:
                if abs(v - old_val) <= 1.0:  # Only try nearby values
                    candidates.append(v)
            # Also try 0
            if 0.0 not in candidates:
                candidates.append(0.0)
                
            best_candidate = old_val
            best_candidate_res = res
            
            for v in candidates:
                if v == old_val:
                    continue
                mat[row, col] = v
                new_res = _residual(U, V, W, n, m, p)
                if new_res < best_candidate_res:
                    best_candidate_res = new_res
                    best_candidate = v
                if time.time() > deadline:
                    break
            mat[row, col] = best_candidate
            res = best_candidate_res
            
            if res < 1e-9:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
                break
    
    # Check best found
    if best_U is not None:
        if _verify_exact(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
            
    return None

def _try_rank_extended(n, m, p, R, deadline, seed=42):
    """
    Extended search with larger value range and more aggressive search.
    """
    np.random.seed(seed)
    
    # Extended half-integer values
    half_int_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    def random_half_int_matrix(rows, cols, density=0.2):
        mat = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if np.random.rand() < density:
                    mat[i, j] = np.random.choice(half_int_values)
        return mat
    
    best_res = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # More restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        U = random_half_int_matrix(n*m, R, density=0.2)
        V = random_half_int_matrix(m*p, R, density=0.2)
        W = random_half_int_matrix(p*n, R, density=0.2)
        
        res = _residual(U, V, W, n, m, p)
        
        for step in range(50):
            if time.time() > deadline:
                break
            if res < 1e-9:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W
                break
                
            if res < best_res:
                best_res = res
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            # Try multiple random changes
            for _ in range(10):
                if time.time() > deadline:
                    break
                    
                mat_choice = np.random.randint(0, 3)
                if mat_choice == 0:
                    mat = U; rows = n*m
                elif mat_choice == 1:
                    mat = V; rows = m*p
                else:
                    mat = W; rows = p*n
                    
                row = np.random.randint(0, rows)
                col = np.random.randint(0, R)
                
                old_val = mat[row, col]
                new_val = np.random.choice(half_int_values)
                
                mat[row, col] = new_val
                new_res = _residual(U, V, W, n, m, p)
                
                # Accept if better or with small probability
                if new_res < res or (np.random.rand() < 0.1 and new_res < res * 1.5):
                    res = new_res
                else:
                    mat[row, col] = old_val
                    
                if res < 1e-9:
                    if _verify_exact(U, V, W, n, m, p):
                        return U, V, W
                    break
    
    if best_U is not None:
        if _verify_exact(best_U, best_V, best_W, n, m, p):
            return best_U, best_V, best_W
            
    return None

def _try_rank_gradient(n, m, p, R, deadline, seed=42):
    """
    Gradient-based search followed by snapping to half-integers.
    """
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    lr = 0.01
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(5000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-8:
            # Snap to half-integers
            U_s = np.round(U * 2) / 2
            V_s = np.round(V * 2) / 2
            W_s = np.round(W * 2) / 2
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
            break
        
        # Compute gradients using finite differences (expensive but simple)
        # For efficiency, use a few random finite difference steps
        for _ in range(3):
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U; rows = n*m
            elif mat_choice == 1:
                mat = V; rows = m*p
            else:
                mat = W; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            eps = 0.01
            old_val = mat[row, col]
            mat[row, col] = old_val + eps
            res_plus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val - eps
            res_minus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val
            
            grad = (res_plus - res_minus) / (2 * eps)
            mat[row, col] = old_val - lr * grad
            
            # Keep values bounded
            mat[row, col] = np.clip(mat[row, col], -2.0, 2.0)
            
        if time.time() > deadline:
            break
    
    # Final snap
    U_s = np.round(best_U * 2) / 2
    V_s = np.round(best_V * 2) / 2
    W_s = np.round(best_W * 2) / 2
    if _verify_exact(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
        
    return None

def _try_rank_hybrid(n, m, p, R, deadline, seed=42):
    """
    Hybrid approach: start with gradient descent, then switch to half-integer local search.
    """
    np.random.seed(seed)
    
    # Phase 1: Gradient descent to get close
    U = np.random.randn(n*m, R) * 0.5
    V = np.random.randn(m*p, R) * 0.5
    W = np.random.randn(p*n, R) * 0.5
    
    lr = 0.01
    best_res = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Quick gradient phase
    for step in range(2000):
        if time.time() > deadline:
            break
            
        res = _residual(U, V, W, n, m, p)
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if res < 1e-6:
            break
        
        # Simple gradient step
        for _ in range(5):
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U; rows = n*m
            elif mat_choice == 1:
                mat = V; rows = m*p
            else:
                mat = W; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            eps = 0.01
            old_val = mat[row, col]
            mat[row, col] = old_val + eps
            res_plus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val - eps
            res_minus = _residual(U, V, W, n, m, p)
            mat[row, col] = old_val
            
            grad = (res_plus - res_minus) / (2 * eps)
            mat[row, col] = old_val - lr * grad
            mat[row, col] = np.clip(mat[row, col], -2.0, 2.0)
            
        if time.time() > deadline:
            break
    
    # Phase 2: Snap and do half-integer local search
    U_s = np.round(best_U * 2) / 2
    V_s = np.round(best_V * 2) / 2
    W_s = np.round(best_W * 2) / 2
    
    res = _residual(U_s, V_s, W_s, n, m, p)
    
    half_int_values = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    
    for step in range(2000):
        if time.time() > deadline:
            break
        if res < 1e-9:
            if _verify_exact(U_s, V_s, W_s, n, m, p):
                return U_s, V_s, W_s
            break
            
        # Try changing a few entries
        for _ in range(5):
            if time.time() > deadline:
                break
                
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                mat = U_s; rows = n*m
            elif mat_choice == 1:
                mat = V_s; rows = m*p
            else:
                mat = W_s; rows = p*n
                
            row = np.random.randint(0, rows)
            col = np.random.randint(0, R)
            
            old_val = mat[row, col]
            new_val = np.random.choice(half_int_values)
            
            mat[row, col] = new_val
            new_res = _residual(U_s, V_s, W_s, n, m, p)
            
            if new_res < res or (np.random.rand() < 0.1 and new_res < res * 1.2):
                res = new_res
            else:
                mat[row, col] = old_val
                
            if res < 1e-9:
                if _verify_exact(U_s, V_s, W_s, n, m, p):
                    return U_s, V_s, W_s
                break
    
    if _verify_exact(U_s, V_s, W_s, n, m, p):
        return U_s, V_s, W_s
        
    return None

def construct():
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up
    # Lower bound for M_{3,3,4}: at least 12 (trivial bound)
    # We know 36 works (schoolbook)
    # Try promising ranks first, then fill in
    
    # Strategy: Try ranks in order, but with time slicing
    # Ranks to try: 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
    
    # Time budget per rank: ~4 seconds for 24 ranks in 110 seconds
    
    best_result = None
    
    # First, try a few key ranks with more time
    key_ranks = [12, 15, 18, 20, 23, 25, 28, 30, 32, 34]
    
    # Try key ranks first with more time each
    for R in key_ranks:
        if time.time() > hard_deadline:
            break
            
        rank_deadline = min(time.time() + 8.0, hard_deadline)
        
        # Try different search strategies
        for seed in [42, 123, 456, 789]:
            if time.time() > rank_deadline:
                break
                
            result = _try_rank_hybrid(n, m, p, R, rank_deadline, seed=seed)
            if result is not None:
                best_result = result
                break
                
        if best_result is not None:
            break
    
    # If not found, try remaining ranks with less time
    if best_result is None:
        remaining_ranks = [R for R in range(12, 36) if R not in key_ranks]
        
        for R in remaining_ranks:
            if time.time() > hard_deadline:
                break
                
            rank_deadline = min(time.time() + 3.0, hard_deadline)
            
            for seed in [42, 123]:
                if time.time() > rank_deadline:
                    break
                    
                result = _try_rank_structured(n, m, p, R, rank_deadline, seed=seed)
                if result is not None:
                    best_result = result
                    break
                    
            if best_result is not None:
                break
    
    # If still not found, use schoolbook
    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
