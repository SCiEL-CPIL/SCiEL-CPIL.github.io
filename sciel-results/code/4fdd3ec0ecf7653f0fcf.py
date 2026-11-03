import time
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_target_tensor(n, m, p):
    """Build the matmul tensor as a list of (a, b, c) indices where T=1."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    return triples

def _verify(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    triples = _build_target_tensor(n, m, p)
    for (a, b, c) in triples:
        val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
        if abs(val - 1.0) > tol:
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    U_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U]
    V_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V]
    W_frac = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W]
    
    triples = _build_target_tensor(n, m, p)
    for (a, b, c) in triples:
        val = sum(U_frac[a][r] * V_frac[b][r] * W_frac[c][r] for r in range(len(U_frac[0])))
        if val != 1:
            return False
    return True

def _snap_to_half_integers(arr, tol=0.01):
    """Snap values to nearest half-integer if close enough."""
    result = np.copy(arr)
    for idx in np.ndindex(result.shape):
        val = result[idx]
        nearest_half = round(val * 2) / 2.0
        if abs(val - nearest_half) < tol:
            result[idx] = nearest_half
    return result

def _schoolbook(n, m, p):
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

def _try_kronecker_structured(n, m, p, r, time_budget=20.0):
    """
    Try a structured decomposition exploiting the Kronecker-like structure
    of 2x3x6 = (2x3x3) x (1x3x2) in a specific way.
    """
    if _time_left() < time_budget:
        return None
    
    # Initialize with a structured pattern based on the Kronecker decomposition
    # k = 3*k2 + k1 where k1 in {0,1,2}, k2 in {0,1}
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strategy: use r = 2 * (rank of 2x3x3 slice)
    # For 2x3x3, try rank 3 per k2 value
    # Total: 2 * 3 * 2 = 12? Or try other structures
    
    # Let's try a direct structured approach
    # For each (k1, k2), use some terms
    # k1 in 0..2, k2 in 0..1
    
    # Initialize with a pattern that might work
    np.random.seed(42)
    
    # Try multiple initialization strategies
    for strategy in range(5):
        if _time_left() < time_budget:
            break
        
        U_init = np.zeros((n * m, r))
        V_init = np.zeros((m * p, r))
        W_init = np.zeros((p * n, r))
        
        if strategy == 0:
            # Kronecker-like: separate k1 and k2
            for k2 in range(2):
                for k1 in range(3):
                    k = k2 * 3 + k1
                    base = k2 * 3 + k1  # 0..5
                    # Use 2 terms per k
                    for term in range(min(2, r // 6)):
                        r_idx = base * 2 + term
                        if r_idx < r:
                            if term == 0:
                                U_init[0 * m + 0, r_idx] = 1
                                U_init[1 * m + 0, r_idx] = 1
                                V_init[0 * p + k, r_idx] = 1
                                W_init[k * n + 0, r_idx] = 1
                                W_init[k * n + 1, r_idx] = 1
                            else:
                                U_init[0 * m + 1, r_idx] = 1
                                U_init[0 * m + 2, r_idx] = 1
                                U_init[1 * m + 1, r_idx] = 1
                                U_init[1 * m + 2, r_idx] = 1
                                V_init[1 * p + k, r_idx] = 0.5
                                V_init[2 * p + k, r_idx] = 0.5
                                W_init[k * n + 0, r_idx] = 1
                                W_init[k * n + 1, r_idx] = 1
        elif strategy == 1:
            # Random initialization with structure
            U_init = np.random.randn(n * m, r) * 0.1
            V_init = np.random.randn(m * p, r) * 0.1
            W_init = np.random.randn(p * n, r) * 0.1
            # Add some structure
            for k in range(p):
                for j in range(m):
                    r_idx = (k * m + j) % r
                    U_init[:, r_idx] += 0.1
                    V_init[j * p + k, r_idx] += 0.1
                    W_init[k * n + 0, r_idx] += 0.1
                    W_init[k * n + 1, r_idx] += 0.1
        elif strategy == 2:
            # Use the fact that we can share U factors across k values
            # U depends only on (i,j), not k
            for j in range(m):
                for i in range(n):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        r_idx = (a * p + k) % r
                        U_init[a, r_idx] = 1.0
                        V_init[b, r_idx] = 1.0
                        W_init[c, r_idx] = 1.0
        elif strategy == 3:
            # Block structure
            for block in range(r):
                i = block % n
                j = (block // n) % m
                k = (block // (n * m)) % p
                U_init[i * m + j, block] = 1.0
                V_init[j * p + k, block] = 1.0
                W_init[k * n + i, block] = 1.0
                # Add some mixing
                if block + 1 < r:
                    U_init[i * m + j, block + 1] = 0.5
                    V_init[j * p + k, block + 1] = 0.5
                    W_init[k * n + i, block + 1] = 0.5
        else:
            # Fully random
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
        
        # Gradient descent refinement
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        triples = _build_target_tensor(n, m, p)
        best_loss = np.inf
        best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
        
        max_iters = 50
        lr = 0.02
        
        for it in range(max_iters):
            if _time_left() < time_budget:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U_curr)
            grad_V = np.zeros_like(V_curr)
            grad_W = np.zeros_like(W_curr)
            
            for (a, b, c) in triples:
                current = np.dot(U_curr[a, :], np.multiply(V_curr[b, :], W_curr[c, :]))
                error = current - 1.0
                loss += error * error
                
                # Gradients
                dU = error * np.multiply(V_curr[b, :], W_curr[c, :])
                dV = error * np.multiply(U_curr[a, :], W_curr[c, :])
                dW = error * np.multiply(U_curr[a, :], V_curr[b, :])
                
                grad_U[a, :] += dU
                grad_V[b, :] += dV
                grad_W[c, :] += dW
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
            
            if loss < 1e-12:
                break
            
            # Adaptive learning rate
            effective_lr = lr / (1 + it * 0.05)
            U_curr -= effective_lr * grad_U
            V_curr -= effective_lr * grad_V
            W_curr -= effective_lr * grad_W
        
        # Check best solution
        if best_loss < 1e-10:
            U_s = _snap_to_half_integers(best_U)
            V_s = _snap_to_half_integers(best_V)
            W_s = _snap_to_half_integers(best_W)
            
            if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            if _verify(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p, tol=1e-6):
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _try_direct_search(n, m, p, r, time_budget=15.0):
    """
    Try to find a rank-r decomposition using more aggressive search.
    """
    if _time_left() < time_budget:
        return None
    
    triples = _build_target_tensor(n, m, p)
    np.random.seed(123)
    
    best_result = None
    best_loss = np.inf
    
    for attempt in range(10):
        if _time_left() < time_budget:
            break
        
        U = np.random.randn(n * m, r) * 0.3
        V = np.random.randn(m * p, r) * 0.3
        W = np.random.randn(p * n, r) * 0.3
        
        loss = np.inf
        for it in range(100):
            if _time_left() < time_budget:
                break
            
            # Compute loss and gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            loss = 0.0
            
            for (a, b, c) in triples:
                current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                error = current - 1.0
                loss += error * error
                
                dU = error * np.multiply(V[b, :], W[c, :])
                dV = error * np.multiply(U[a, :], W[c, :])
                dW = error * np.multiply(U[a, :], V[b, :])
                
                grad_U[a, :] += dU
                grad_V[b, :] += dV
                grad_W[c, :] += dW
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-10:
                    U_s = _snap_to_half_integers(U)
                    V_s = _snap_to_half_integers(V)
                    W_s = _snap_to_half_integers(W)
                    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-6):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
            
            if loss < 1e-12:
                break
            
            lr = 0.05 / (1 + it * 0.1)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    n, m, p = 2, 3, 6
    
    # Start with schoolbook as fallback
    schoolbook = _schoolbook(n, m, p)
    best = schoolbook
    best_rank = n * m * p
    
    # Try ranks from 34 down to 12
    for r in range(34, 11, -1):
        if _time_left() < 15:
            break
        
        # Try structured approach
        result = _try_kronecker_structured(n, m, p, r, time_budget=10.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-6):
                best = result
                best_rank = r
                break
        
        # Try direct search
        result = _try_direct_search(n, m, p, r, time_budget=8.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-6):
                best = result
                best_rank = r
                break
    
    return best

