import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor for n x m * m x p."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _build_tensor_332():
    """Build the matrix multiplication tensor for 3x3 * 3x2."""
    return _build_tensor(3, 3, 2)

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor T."""
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

def _search_rank_15_332(max_time=10.0):
    """Search for a rank-15 decomposition of 3x3 * 3x2."""
    T = _build_tensor_332()
    n, m, p = 3, 3, 2
    nm, mp, pn = 9, 6, 6
    rank = 15
    
    np.random.seed(42)
    start_time = time.time()
    
    # Try a few random initializations with gradient descent
    for attempt in range(20):
        if time.time() - start_time > max_time:
            break
        
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        lr = 0.05
        for it in range(500):
            if time.time() - start_time > max_time:
                break
            
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = T_recon - T
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                dU[:, r] = 2 * np.einsum('abc,b,c->a', err, V[:, r], W[:, r])
                dV[:, r] = 2 * np.einsum('abc,a,c->b', err, U[:, r], W[:, r])
                dW[:, r] = 2 * np.einsum('abc,a,b->c', err, U[:, r], V[:, r])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
        # Check convergence
        if _verify_decomposition(U, V, W, T):
            # Round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round, T):
                return U_round, V_round, W_round
                
    return None

def _build_rank_30_from_332(U332, V332, W332):
    """Build a rank-30 decomposition of 3x3 * 3x4 from a rank-15 decomposition of 3x3 * 3x2."""
    n, m, p = 3, 3, 4
    nm, mp, pn = 9, 12, 12
    rank = 30
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # First 15 terms for left half (cols 0,1 of B)
    U[:, 0:15] = U332
    V[0:6, 0:15] = V332
    W[0:6, 0:15] = W332
    
    # Next 15 terms for right half (cols 2,3 of B)
    U[:, 15:30] = U332
    V[6:12, 15:30] = V332
    W[6:12, 15:30] = W332
    
    return U, V, W

def construct():
    deadline = time.time() + 140 # Safety margin
    
    # Start with baseline
    baseline = _get_rank_34_baseline()
    best_result = baseline
    best_rank = 34
    
    # Try to find rank-30 solution via 3x3x2 decomposition
    if time.time() < deadline - 20:
        res_332 = _search_rank_15_332(max_time=min(10.0, deadline - time.time() - 20))
        if res_332 is not None:
            U332, V332, W332 = res_332
            U30, V30, W30 = _build_rank_30_from_332(U332, V332, W332)
            
            # Verify the full 3x3x4 decomposition
            T_full = _build_tensor(3, 3, 4)
            if _verify_decomposition(U30, V30, W30, T_full):
                best_result = {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
                best_rank = 30
                
    return best_result
