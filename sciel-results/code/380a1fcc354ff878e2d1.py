"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U @ diag @ V^T reconstructs the multiplication tensor contracted with W."""
    # The tensor is T[i, j, k] = delta. We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Actually the decomposition says: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: M_{i,j,k} = A_{i,j} * B_{j,k}
    # Decomposition: M_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # Wait, that's not right. The standard decomposition is:
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}
    # where T_{i,j,k} = delta_{i,j,k} is the multiplication tensor.
    # A rank-r decomposition of T means T_{i,j,k} = sum_{r=1}^r T1_{i,j,r} T2_{j,k,r} T3_{k,i,r}
    # Here U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # So T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify by checking a few random entries
    # Actually for exact verification, we check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True


def _try_rank(n, m, p, rank, seed=42, max_iter=500, deadline=None):
    """Try to find a rank decomposition using iterative optimization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random entries
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Gradient descent on the objective: sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        loss = 0.0
        for i, j, k in indices:
            ui = U[i*m+j, :]
            vj = V[j*p+k, :]
            wk = W[k*n+i, :]
            prod = ui * vj * wk
            s = np.sum(prod)
            diff = s - 1.0
            loss += diff * diff
            # d/dU: 2*diff * vj * wk
            grad_U[i*m+j, :] += 2 * diff * vj * wk
            grad_V[j*p+k, :] += 2 * diff * ui * wk
            grad_W[k*n+i, :] += 2 * diff * ui * vj
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Decay learning rate
        lr *= 0.999
    
    # Check if we found a valid solution
    if best_loss < 1e-9:
        # Try to round to integers/half-integers
        U_round = np.round(best_U * 2) / 2
        V_round = np.round(best_V * 2) / 2
        W_round = np.round(best_W * 2) / 2
        
        # Verify rounded version
        ok = True
        for i, j, k in indices:
            val = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
            if abs(val - 1.0) > 1e-6:
                ok = False
                break
        if ok:
            return U_round.astype(np.int64) * 2, V_round.astype(np.int64) * 2, W_round.astype(np.int64) * 2, rank
    
    return None


def _try_integer_search(n, m, p, rank, seed=42, max_attempts=200, deadline=None):
    """Try random integer/half-integer initializations with coordinate descent."""
    rng = np.random.RandomState(seed)
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline:
            break
        
        # Random small integer initialization
        U = rng.randint(-2, 3, size=(n*m, rank))
        V = rng.randint(-2, 3, size=(m*p, rank))
        W = rng.randint(-2, 3, size=(p*n, rank))
        
        # Coordinate descent: update one matrix at a time
        for it in range(100):
            if deadline and time.time() > deadline:
                break
            
            # Update U
            for r in range(rank):
                # For each (i,j), optimal U[i*m+j, r] given V and W
                # sum_k W[k*n+i, r] * V[j*p+k, r] should relate to target
                pass  # This is complex; skip for now
        
        # Just check if this random init works (unlikely but fast)
        ok = True
        for i, j, k in indices:
            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
            if abs(val - 1.0) > 1e-6:
                ok = False
                break
        if ok:
            return U, V, W, rank
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U0, V0, W0)
    
    # Try to find lower rank decompositions
    # Known bounds: for 4x4 * 4x9, the rank is at most n*m*p = 144
    # Lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # Actually the trivial lower bound is max(n*m, m*p, n*p) but tighter bounds exist
    
    # Try ranks from a reasonable lower bound upward
    # For 4x4x9, let's try from 36 up to 144
    lower_bound = max(n*m, m*p, n*p)  # 36
    
    # Try a few specific ranks
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Only try every few ranks to save time
        if rank < 50:
            continue  # Skip very low ranks, unlikely to work
        
        result = _try_rank(n, m, p, rank, seed=rank, max_iter=200, deadline=deadline)
        if result is not None:
            U_r, V_r, W_r, r = result
            # Scale back (we multiplied by 2)
            U_r = U_r / 2
            V_r = V_r / 2
            W_r = W_r / 2
            
            # Verify
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_result = (U_r, V_r, W_r)
                break  # Found a valid decomposition, use it
    
    # If we didn't find anything better, return schoolbook
    U, V, W = best_result
    
    # Convert to lists as required
    U_list = U.tolist() if hasattr(U, 'tolist') else [[float(x) for x in row] for row in U]
    V_list = V.tolist() if hasattr(V, 'tolist') else [[float(x) for x in row] for row in V]
    W_list = W.tolist() if hasattr(W, 'tolist') else [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
