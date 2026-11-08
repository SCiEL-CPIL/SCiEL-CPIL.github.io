"""Matrix multiplication 2x3 x 3x3 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose the 2x3 x 3x3 multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    U_reshaped = Ua.reshape(n, m, rank)
    V_reshaped = Va.reshape(m, p, rank)
    W_reshaped = Wa.reshape(p, n, rank)
    
    result = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if result.shape != (n, m, p):
        return False
    return np.allclose(result, 1.0, atol=tol)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # 20s safety margin before 150s limit
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    n, m, p = DIMENSIONS
    nU = n * m  # 6
    nV = m * p  # 9
    nW = p * n  # 6
    
    # Build target tensor T of shape (nU, nV, nW)
    T = np.zeros((nU, nV, nW))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    def check_half_integer(U, V, W, T, tol=1e-6):
        """Check if U, V, W can be rounded to half-integers and still decompose T."""
        # Try rounding to nearest half-integer
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        
        # Check if rounding was close
        if (np.max(np.abs(U - U_rounded)) > tol or 
            np.max(np.abs(V - V_rounded)) > tol or 
            np.max(np.abs(W - W_rounded)) > tol):
            return False, None, None, None
            
        # Verify the rounded matrices still decompose T
        T_recon = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
        if np.allclose(T_recon, T, atol=1e-8):
            return True, U_rounded, V_rounded, W_rounded
        return False, None, None, None
    
    def optimize_rank(R, max_steps=3000, lr_init=0.05):
        """Try to find a rank-R decomposition using gradient descent."""
        # Initialize with random values
        np.random.seed(42)  # Fixed seed for reproducibility within this call
        U = np.random.randn(nU, R) * 0.3
        V = np.random.randn(nV, R) * 0.3
        W = np.random.randn(nW, R) * 0.3
        
        lr = lr_init
        prev_err = np.inf
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Compute reconstruction error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err_tensor = T_recon - T
            err = np.sum(err_tensor**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if err < 1e-8:
                break
                
            # Compute gradients
            dU = np.einsum('ijk,kr,jr->ir', err_tensor, W, V)
            dV = np.einsum('ijk,ir,kr->jr', err_tensor, U, W)
            dW = np.einsum('ijk,ir,jr->kr', err_tensor, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Adaptive learning rate
            if err > prev_err * 1.2:
                lr *= 0.5
            else:
                if err < prev_err * 0.8:
                    lr *= 1.1
                prev_err = err
                
            # Clamp learning rate
            lr = np.clip(lr, 1e-5, 0.5)
            
        # Check if we found a good solution
        T_recon = np.einsum('ir,jr,kr->ijk', best_U, best_V, best_W)
        if np.allclose(T_recon, T, atol=1e-6):
            return True, best_U, best_V, best_W
        return False, best_U, best_V, best_W
    
    # Try to find lower rank decompositions
    # Start from rank 17 down to 10
    for R in range(best_rank - 1, 9, -1):
        if time.time() > deadline - 20:
            break
            
        # Try multiple random restarts for this rank
        found = False
        for restart in range(10):
            if time.time() > deadline - 15:
                break
                
            # Vary the random seed for each restart
            np.random.seed(42 + restart * 1000)
            
            success, U, V, W = optimize_rank(R)
            if success:
                # Check if it can be made exact with half-integers
                exact, U_exact, V_exact, W_exact = check_half_integer(U, V, W, T)
                if exact:
                    # Convert to lists
                    best_rank = R
                    best = {
                        "U": U_exact.tolist(),
                        "V": V_exact.tolist(),
                        "W": W_exact.tolist()
                    }
                    found = True
                    break
                    
        if found:
            # If we found a lower rank, stop searching for even lower ranks
            # (or continue if we want to try to go even lower, but for now stop)
            break
            
    # Final verification
    if not _verify(best['U'], best['V'], best['W']):
        # Fallback to schoolbook if verification fails
        best = _schoolbook()
        
    return best
# EVOLVE-BLOCK-END
