import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _get_rank_34_baseline():
    """Return the known rank-34 decomposition."""
    n, m, p = DIMENSIONS
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # Strassen Left (for columns 0,1 of B)
    U[0, col]=1; U[4, col]=1; V[0, col]=1; V[5, col]=1; W[0, col]=1; W[4, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[0, col]=1; W[1, col]=1; W[4, col]=-1; col+=1
    U[0, col]=1; V[1, col]=1; V[5, col]=-1; W[3, col]=1; W[4, col]=1; col+=1
    U[4, col]=1; V[4, col]=1; V[0, col]=-1; W[0, col]=1; W[1, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[5, col]=1; W[0, col]=-1; W[3, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[0, col]=1; V[1, col]=1; W[4, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[4, col]=1; V[5, col]=1; W[0, col]=1; col+=1
    # A12*B2L
    U[2, col]=1; V[8, col]=1; W[0, col]=1; col+=1
    U[2, col]=1; V[9, col]=1; W[3, col]=1; col+=1
    U[5, col]=1; V[8, col]=1; W[1, col]=1; col+=1
    U[5, col]=1; V[9, col]=1; W[4, col]=1; col+=1
    # A21*B1L
    U[6, col]=1; V[0, col]=1; W[2, col]=1; col+=1
    U[6, col]=1; V[1, col]=1; W[5, col]=1; col+=1
    U[7, col]=1; V[4, col]=1; W[2, col]=1; col+=1
    U[7, col]=1; V[5, col]=1; W[5, col]=1; col+=1
    # A22*B2L
    U[8, col]=1; V[8, col]=1; W[2, col]=1; col+=1
    U[8, col]=1; V[9, col]=1; W[5, col]=1; col+=1
    # Strassen Right (for columns 2,3 of B)
    U[0, col]=1; U[4, col]=1; V[2, col]=1; V[7, col]=1; W[6, col]=1; W[10, col]=1; col+=1
    U[3, col]=1; U[4, col]=1; V[2, col]=1; W[7, col]=1; W[10, col]=-1; col+=1
    U[0, col]=1; V[3, col]=1; V[7, col]=-1; W[9, col]=1; W[10, col]=1; col+=1
    U[4, col]=1; V[6, col]=1; V[2, col]=-1; W[6, col]=1; W[7, col]=1; col+=1
    U[0, col]=1; U[1, col]=1; V[7, col]=1; W[6, col]=-1; W[9, col]=1; col+=1
    U[3, col]=1; U[0, col]=-1; V[2, col]=1; V[3, col]=1; W[10, col]=1; col+=1
    U[1, col]=1; U[4, col]=-1; V[6, col]=1; V[7, col]=1; W[6, col]=1; col+=1
    # A12*B2R
    U[2, col]=1; V[10, col]=1; W[6, col]=1; col+=1
    U[2, col]=1; V[11, col]=1; W[9, col]=1; col+=1
    U[5, col]=1; V[10, col]=1; W[7, col]=1; col+=1
    U[5, col]=1; V[11, col]=1; W[10, col]=1; col+=1
    # A21*B1R
    U[6, col]=1; V[2, col]=1; W[8, col]=1; col+=1
    U[6, col]=1; V[3, col]=1; W[11, col]=1; col+=1
    U[7, col]=1; V[6, col]=1; W[8, col]=1; col+=1
    U[7, col]=1; V[7, col]=1; W[11, col]=1; col+=1
    # A22*B2R
    U[8, col]=1; V[10, col]=1; W[8, col]=1; col+=1
    U[8, col]=1; V[11, col]=1; W[11, col]=1; col+=1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_merge_terms(U, V, W):
    """Try to merge similar terms to reduce rank."""
    rank = U.shape[1]
    # Check for duplicate or linearly dependent columns
    # This is a simple check - could be more sophisticated
    return None

def _random_search_for_rank(target_rank, max_time=10.0, seed=42):
    """Search for a decomposition of given rank using random initialization."""
    if max_time <= 0:
        return None
    
    T = _build_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    np.random.seed(seed)
    start_time = time.time()
    
    for attempt in range(100):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with random vectors
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Simple gradient descent
        lr = 0.01
        for it in range(200):
            if time.time() - start_time > max_time:
                break
            
            T_recon = np.zeros_like(T)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = T_recon - T
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(target_rank):
                dU[:, r] = 2 * np.einsum('abc,b,c->a', err, V[:, r], W[:, r])
                dV[:, r] = 2 * np.einsum('abc,a,c->b', err, U[:, r], W[:, r])
                dW[:, r] = 2 * np.einsum('abc,a,b->c', err, U[:, r], V[:, r])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a valid decomposition
        if _verify_decomposition(U, V, W):
            # Try to round to integers/half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round):
                return (U_round.tolist(), V_round.tolist(), W_round.tolist())
    
    return None

def construct():
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Start with the known rank-34 solution
    baseline = _get_rank_34_baseline()
    best_result = baseline
    best_rank = 34
    
    # Try to find better decompositions
    for rank in range(30, best_rank):
        if time.time() > deadline - 20:
            break
        
        max_time = min(5.0, max(1.0, deadline - time.time() - 15))
        result = _random_search_for_rank(rank, max_time=max_time, seed=42+rank)
        if result is not None:
            U_list, V_list, W_list = result
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            # Don't break; keep searching for even better
            continue
    
    return best_result
