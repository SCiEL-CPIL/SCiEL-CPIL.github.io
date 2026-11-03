import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Check if they are close
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W, rank

def _find_rank11_2x3x3():
    """
    Find a rank-11 decomposition for 2x3x3 matrix multiplication.
    U: 6x11, V: 9x11, W: 6x11
    """
    n, m, p = 2, 3, 3
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    R = 11
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for seed in range(100):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        lr = 0.05
        for it in range(1000):
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            diff = T_dec - T
            loss = np.sum(diff ** 2)
            
            if loss < 1e-12:
                break
                
            dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
            dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
            dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Round to nearest 0.5 to maintain half-integer property
            # This is a heuristic to snap to valid solutions
            if it % 10 == 0:
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
        T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
        final_loss = np.sum((T_dec - T) ** 2)
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_loss < 1e-12:
            break
            
    if best_loss < 1e-9:
        return best_U, best_V, best_W
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Try to find rank-11 for 2x3x3 and build rank-22 for 2x3x6
    if time.time() < deadline - 10:
        res = _find_rank11_2x3x3()
        if res is not None:
            U2, V2, W2 = res
            R2 = U2.shape[1] # 11
            
            U_full = np.zeros((n * m, 2 * R2))
            V_full = np.zeros((m * p, 2 * R2))
            W_full = np.zeros((p * n, 2 * R2))
            
            # Block 1: k in 0..2
            # U is same for both blocks
            U_full[:, 0:R2] = U2
            U_full[:, R2:2*R2] = U2
            
            for j in range(m):
                for k in range(3):
                    # V index in full: j * 6 + k
                    # V index in 2x3x3: j * 3 + k
                    V_full[j * p + k, 0:R2] = V2[j * 3 + k]
                    # V index in full for block 2: j * 6 + (3 + k)
                    # V index in 2x3x3: j * 3 + k
                    V_full[j * p + (3 + k), R2:2*R2] = V2[j * 3 + k]
                    
            for k in range(3):
                for i in range(n):
                    # W index in full: k * 2 + i
                    # W index in 2x3x3: k * 2 + i
                    W_full[k * n + i, 0:R2] = W2[k * 2 + i]
                    # W index in full for block 2: (3 + k) * 2 + i
                    # W index in 2x3x3: k * 2 + i
                    W_full[(3 + k) * n + i, R2:2*R2] = W2[k * 2 + i]
                    
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return {
                    "U": U_full.tolist(),
                    "V": V_full.tolist(),
                    "W": W_full.tolist()
                }
                
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
