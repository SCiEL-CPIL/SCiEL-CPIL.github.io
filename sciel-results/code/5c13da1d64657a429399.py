"""Find a rank-minimal exact tensor decomposition for multiplying a 3x3 by a 3x7 matrix."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should equal 1 for all i,j,k.
    """
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_random(n, m, p, rank, n_iter=500, seed=42, deadline=None):
    """
    Try to find a rank-`rank` decomposition using random initialization and iterative refinement.
    Returns dict with U, V, W if found, else None.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = rng.randn(n*m, rank) * 0.5
        V = rng.randn(m*p, rank) * 0.5
        W = rng.randn(p*n, rank) * 0.5
        
        # Iterative refinement using alternating least squares
        for inner in range(300):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Target: for all k, sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
                    # Fix V and W, solve for U[i,j,:]
                    # This is a linear system: for each k, sum_r U[i,j,r] * (V[j,k,r]*W[k,i,r]) = 1
                    # So A @ U[i,j,:] = b where A[k,r] = V[j,k,r]*W[k,i,r], b[k] = 1
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-14:
                break
        
        if best_err < 1e-14:
            break
    
    if best_err < 1e-10:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_rank_gd(n, m, p, rank, n_iter=100, seed=42, deadline=None):
    """
    Try gradient descent to find a rank-`rank` decomposition.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    lr = 0.01
    for it in range(n_iter):
        if time.time() > deadline - 10:
            break
        
        U = rng.randn(n*m, rank) * 0.3
        V = rng.randn(m*p, rank) * 0.3
        W = rng.randn(p*n, rank) * 0.3
        
        for inner in range(200):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i,j,r] of (sum_r' U[i,j,r']*V[j,k,r']*W[k,i,r'] - 1)^2
                        # = 2*(val-1)*V[j,k,r]*W[k,i,r]
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err_val = 2.0 * (val - 1.0)
                        for r in range(rank):
                            grad_U[i*m+j, r] += err_val * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += err_val * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += err_val * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if best_err < 1e-14:
                break
            if best_err < 1e-10:
                break
        
        if best_err < 1e-14:
            break
    
    if best_err < 1e-10:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    """
    n, m, p = 3, 3, 7
    deadline = time.time() + 110  # Safety margin (150s limit)
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try progressively lower ranks using random search
    for rank in range(best_rank - 1, max(20, best_rank - 20), -1):
        if time.time() > deadline - 30:
            break
        
        found = _try_rank_random(n, m, p, rank, n_iter=100, seed=rank * 1000 + 7, deadline=deadline)
        if found is not None:
            result = found
            best_rank = rank
            continue
    
    # Final verification
    U_list = [[float(x) for x in row] for row in result["U"]]
    V_list = [[float(x) for x in row] for row in result["V"]]
    W_list = [[float(x) for x in row] for row in result["W"]]
    if not _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
        result = _schoolbook(n, m, p)
    
    return result
# EVOLVE-BLOCK-END
