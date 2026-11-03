import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_rank_18_attempt(n, m, p):
    """
    Try to construct a rank-18 decomposition.
    
    For each (j,k), we need:
      T[j, j*p+k, k*n] = 1
      T[j+m, j*p+k, k*n+1] = 1
    
    With 18 terms (one per (j,k)), we need to handle cross-terms.
    Let's use the structure:
      U[j, t] = 1, U[j+m, t] = 1
      V[j*p+k, t] = 1
      W[k*n, t] = a, W[k*n+1, t] = b
      
    Then:
      T[j, j*p+k, k*n] = 1*1*a = a = 1 => a=1
      T[j+m, j*p+k, k*n+1] = 1*1*b = b = 1 => b=1
      But T[j, j*p+k, k*n+1] = 1*1*b = 1 ≠ 0 (bad)
      And T[j+m, j*p+k, k*n] = 1*1*a = 1 ≠ 0 (bad)
    
    So rank 18 with this simple structure doesn't work.
    Let's try a different approach: use 2 terms per j (not per (j,k)).
    For fixed j, we have p=6 k values. We need to encode 6 independent B[j,k] values
    into 2 terms. This seems hard.
    
    Alternative: use the fact that the problem has a special structure.
    Let me try rank 24: 4 terms per j (3 j values * 4 = 12... no, that's 12).
    Actually, let's just try optimization for various ranks.
    """
    return None

def _try_rank_with_optimization(n, m, p, r, max_iters=2000, lr=0.01, seed=42):
    """Try to find rank-r decomposition using optimization with multiple restarts."""
    if _time_left() < 15:
        return None
    
    np.random.seed(seed)
    
    # Build list of target triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    best_result = None
    best_loss = float('inf')
    
    for attempt in range(3):
        if _time_left() < 15:
            break
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        current_loss = float('inf')
        patience = 0
        max_patience = 200
        
        for it in range(max_iters):
            if _time_left() < 15:
                break
            
            # Compute loss and gradients
            loss = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for (a, b, c) in triples:
                current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                error = current - 1.0
                loss += error * error
                
                for rr in range(r):
                    dU = error * V[b, rr] * W[c, rr]
                    dV = error * U[a, rr] * W[c, rr]
                    dW = error * U[a, rr] * V[b, rr]
                    grad_U[a, rr] += dU
                    grad_V[b, rr] += dV
                    grad_W[c, rr] += dW
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm > 1e-10:
                step = lr / (1.0 + it * 0.005)
                U -= step * grad_U
                V -= step * grad_V
                W -= step * grad_W
            
            if loss < current_loss * 0.99:
                current_loss = loss
                patience = 0
            else:
                patience += 1
            
            if loss < 1e-12:
                break
            
            if patience > max_patience:
                break
        
        # Try to snap to half-integers
        def snap(arr):
            result = np.copy(arr)
            for idx in np.ndindex(result.shape):
                val = result[idx]
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_half) < 0.01:
                    result[idx] = nearest_half
                # Also try integers
                nearest_int = round(val)
                if abs(val - nearest_int) < 0.001:
                    result[idx] = nearest_int
            return result
        
        U_s = snap(U)
        V_s = snap(V)
        W_s = snap(W)
        
        if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, tol=1e-9):
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if loss < best_loss:
            best_loss = loss
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

def _try_structured_low_rank(n, m, p):
    """
    Try structured decompositions that exploit the matmul structure.
    
    Key insight: For multiplying A (n×m) by B (m×p), the tensor is:
    T[i*m+j, j*p+k, k*n+i] = 1
    
    For n=2, m=3, p=6:
    - 6 U rows, 18 V rows, 12 W rows
    - 36 target entries
    
    Strategy: Group by j. For each j, we have:
    - U rows: j and 3+j
    - V rows: j*6, j*6+1, ..., j*6+5
    - W rows: 0,1,2,3,4,5,6,7,8,9,10,11 (all, since k ranges 0-5 and i ranges 0-1)
    
    For fixed j, the sub-tensor is:
    T[j, j*6+k, 2k] = 1 for k=0..5
    T[3+j, j*6+k, 2k+1] = 1 for k=0..5
    
    This is a 2×6×12 sub-tensor with 12 non-zero entries.
    The rank of this sub-tensor is at most 6 (one per k).
    
    If we can do 6 terms per j, total rank = 3*6 = 18.
    But we showed rank 18 doesn't work with simple structure.
    
    Let me try: for each j, use 6 terms, one per k, but with a twist.
    Term t_k for (j,k):
      U[j, t_k] = 1, U[3+j, t_k] = 0
      V[j*6+k, t_k] = 1
      W[2k, t_k] = 1, W[2k+1, t_k] = 0
    
    And term t_k' for (j,k):
      U[j, t_k'] = 0, U[3+j, t_k'] = 1
      V[j*6+k, t_k'] = 1
      W[2k, t_k'] = 0, W[2k+1, t_k'] = 1
    
    This gives 12 terms per j, total 36. That's schoolbook again.
    
    To reduce, we need to share terms across j values.
    """
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # Try structured approaches first
    # Then optimization for lower ranks
    # Fallback to schoolbook
    
    # Try rank 18 with optimization
    for r in [18, 20, 22, 24, 27, 30, 33]:
        if _time_left() < 20:
            break
        result = _try_rank_with_optimization(n, m, p, r, max_iters=1000, lr=0.01, seed=42)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result
    
    # Fallback to schoolbook rank 36
    return _construct_rank_36(n, m, p)
