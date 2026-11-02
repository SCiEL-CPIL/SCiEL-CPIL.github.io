"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_factors(U, V, W, T, tol=1e-9):
    """Check if factors exactly decompose the tensor."""
    T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.max(np.abs(T_decomp - T)) < tol

def _is_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    doubled = 2 * arr
    rounded = np.round(doubled)
    return np.max(np.abs(doubled - rounded)) < tol

def _round_to_half(arr):
    """Round to nearest half-integer."""
    return np.round(2 * arr) / 2.0

def _optimize_rank(n, m, p, rank, max_iters=3000, lr_init=0.1, n_restarts=5, seed=42):
    """Optimize factors for given rank using gradient descent with decay."""
    T = _build_tensor(n, m, p)
    rng = np.random.RandomState(seed)
    
    best_loss = np.inf
    best_factors = None
    
    for restart in range(n_restarts):
        U = rng.randn(n * m, rank) * 0.5
        V = rng.randn(m * p, rank) * 0.5
        W = rng.randn(p * n, rank) * 0.5
        
        lr = lr_init
        for it in range(max_iters):
            T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
            diff = T_decomp - T
            loss = np.sum(diff ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_factors = (U.copy(), V.copy(), W.copy())
            
            if loss < 1e-14:
                break
            
            # Compute gradients
            grad_U = 2 * np.einsum('ijk,jc,kc->ic', diff, V, W)
            grad_V = 2 * np.einsum('ijk,ic,kc->jc', diff, U, W)
            grad_W = 2 * np.einsum('ijk,ic,jc->kc', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            if it % 500 == 0 and it > 0:
                lr *= 0.5
        
        if best_loss < 1e-10:
            break
    
    if best_loss > 1e-8:
        return None
    
    U_opt, V_opt, W_opt = best_factors
    
    # Try rounding strategies
    candidates = []
    
    # Strategy 1: all half-integers
    candidates.append((_round_to_half(U_opt), _round_to_half(V_opt), _round_to_half(W_opt)))
    
    # Strategy 2: all integers
    candidates.append((np.round(U_opt), np.round(V_opt), np.round(W_opt)))
    
    # Strategy 3: mixed
    for ru in ['half', 'int']:
        for rv in ['half', 'int']:
            for rw in ['half', 'int']:
                Ur = _round_to_half(U_opt) if ru == 'half' else np.round(U_opt)
                Vr = _round_to_half(V_opt) if rv == 'half' else np.round(V_opt)
                Wr = _round_to_half(W_opt) if rw == 'half' else np.round(W_opt)
                candidates.append((Ur, Vr, Wr))
    
    for Ur, Vr, Wr in candidates:
        if _verify_factors(Ur, Vr, Wr, T):
            return Ur, Vr, Wr
    
    # If no exact match, try to refine with local search on rounded values
    # Try small perturbations around the rounded values
    best_rounded_loss = np.inf
    best_rounded = None
    for Ur, Vr, Wr in candidates:
        loss = np.sum((np.einsum('ic,jc,kc->ijk', Ur, Vr, Wr) - T) ** 2)
        if loss < best_rounded_loss:
            best_rounded_loss = loss
            best_rounded = (Ur, Vr, Wr)
    
    if best_rounded_loss < 1e-6:
        # Very close, try to fix with exact arithmetic
        Ur, Vr, Wr = best_rounded
        # This shouldn't normally be reached if _verify_factors passed
        return Ur, Vr, Wr
    
    return None

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _to_list(arr):
    """Convert numpy array to list with int where possible."""
    result = []
    for row in arr:
        r = []
        for x in row:
            if abs(x - round(x)) < 1e-9:
                r.append(int(round(x)))
            elif abs(2*x - round(2*x)) < 1e-9:
                r.append(float(x))  # half-integer
            else:
                r.append(float(x))
        result.append(r)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try increasing ranks
    for rank in [10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 28, 30, 35, 40, 45, 50]:
        if time.time() > deadline - 15:
            break
        
        result = _optimize_rank(n, m, p, rank, max_iters=2000, lr_init=0.1, n_restarts=3, seed=42)
        if result is not None:
            U, V, W = result
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
