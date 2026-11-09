"""Matrix multiplication 3,3,7 - rank-minimal tensor decomposition search."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [3, 3, 7]

def _tensor_entries():
    """Generate the (i,j,k) -> (a,b,c) mapping for the multiplication tensor."""
    n, m, p = DIMENSIONS
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # row in U: (i,j)
                b = j * p + k  # row in V: (j,k)
                c = k * n + i  # row in W: (k,i) - trace convention
                entries.append((a, b, c))
    return entries

def _verify(U, V, W, tol=1e-10):
    """Check if U, V, W exactly decompose the tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Build the tensor from U, V, W
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    entries = _tensor_entries()
    # Create sparse check
    for (a, b, c) in entries:
        val = 0.0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if abs(val - 1.0) > tol:
            return False
    # Check non-entries are zero
    nU = n * m
    nV = m * p
    nW = p * n
    entry_set = set(entries)
    for a in range(nU):
        for b in range(nV):
            for c in range(nW):
                if (a, b, c) not in entry_set:
                    val = 0.0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val) > tol:
                        return False
    return True

def _verify_fast(U_np, V_np, W_np, tol=1e-10):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    nU = n * m
    nV = m * p
    nW = p * n
    rank = U_np.shape[1]
    # Compute full tensor
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    # Check entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > tol:
                    return False
    # Check non-entries
    entry_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                entry_set.add((a, b, c))
    for a in range(nU):
        for b in range(nV):
            for c in range(nW):
                if (a, b, c) not in entry_set:
                    if abs(T[a, b, c]) > tol:
                        return False
    return True

def _make_schoolbook():
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

def _try_optimize(rank, max_iters=5000, lr=0.01, seed=42):
    """Try to find a rank-rank decomposition via gradient descent."""
    n, m, p = DIMENSIONS
    nU = n * m
    nV = m * p
    nW = p * n
    np.random.seed(seed)
    
    U = np.random.randn(nU, rank) * 0.1
    V = np.random.randn(nV, rank) * 0.1
    W = np.random.randn(nW, rank) * 0.1
    
    entries = _tensor_entries()
    entry_set = set(entries)
    all_indices = [(a, b, c) for a in range(nU) for b in range(nV) for c in range(nW)]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Compute T
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Loss
        loss = 0.0
        for (a, b, c) in entries:
            loss += (T[a, b, c] - 1.0) ** 2
        for (a, b, c) in all_indices:
            if (a, b, c) not in entry_set:
                loss += T[a, b, c] ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for (a, b, c) in all_indices:
            target = 1.0 if (a, b, c) in entry_set else 0.0
            err = T[a, b, c] - target
            dU[a] += 2 * err * (V[b] * W[c])
            dV[b] += 2 * err * (U[a] * W[c])
            dW[c] += 2 * err * (U[a] * V[b])
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return best_U, best_V, best_W, best_loss

def _try_random_search(rank, n_trials=100, seed=0):
    """Random search for exact decomposition."""
    n, m, p = DIMENSIONS
    nU = n * m
    nV = m * p
    nW = p * n
    np.random.seed(seed)
    
    for trial in range(n_trials):
        U = np.random.randn(nU, rank)
        V = np.random.randn(nV, rank)
        W = np.random.randn(nW, rank)
        if _verify_fast(U, V, W):
            return U, V, W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try known lower bounds and work up
    # For 3x3x3, rank is 23. For 3x3 x 3x7, try various ranks
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But tensor rank is likely higher
    
    # Try ranks from 21 upward
    for rank in range(21, best_rank + 1):
        if time.time() > deadline - 30:
            break
        
        # Try optimization with a few seeds
        found = False
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            U, V, W, loss = _try_optimize(rank, max_iters=2000, lr=0.05, seed=seed)
            
            if loss < 1e-8:
                # Verify with tight tolerance
                if _verify_fast(U, V, W, tol=1e-10):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # Try to go lower? No, we're going up from lower bound
            break
    
    # Convert to lists
    U_list = [[best_U[i][j] for j in range(best_rank)] for i in range(len(best_U))]
    V_list = [[best_V[i][j] for j in range(best_rank)] for i in range(len(best_V))]
    W_list = [[best_W[i][j] for j in range(best_rank)] for i in range(len(best_W))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
