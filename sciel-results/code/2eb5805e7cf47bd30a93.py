import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[a,b,c] where:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 84)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact."""
        # S[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient check: for each (a,b), sum_r U[a,r]*V[b,r]*W[c,r] should match T[a,b,:]
        # Use einsum for efficiency
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def try_optimize(r, U_init, V_init, W_init, max_iter=200, lr=0.01):
        """Try to optimize decomposition to rank r."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Compute gradient of loss = ||S - T||^2 where S = sum_r U[:,r]⊗V[:,r]⊗W[:,r]
            # dL/dU[a,r] = 2 * sum_{b,c} (S[a,b,c]-T[a,b,c]) * V[b,r] * W[c,r]
            # Similarly for V and W
            
            S = np.einsum('ar,br,cr->abc', U, V, W)
            err = S - T
            
            # Gradients
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adaptive learning rate
            loss = np.sum(err**2)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Normalize gradients
            gnorm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if gnorm < 1e-12:
                break
            
            # Simple gradient descent with momentum-like behavior
            scale = lr / (gnorm + 1e-8)
            U -= scale * dU
            V -= scale * dV
            W -= scale * dW
            
            # Check for convergence
            if loss < 1e-10:
                break
        
        # Verify
        if verify(best_U, best_V, best_W):
            return best_U, best_V, best_W, True
        return None, None, None, False
    
    # Try various target ranks from high to low
    # Start from schoolbook and try to reduce
    for target_rank in range(best_rank - 1, 27, -1):
        if time.time() > deadline:
            break
        
        # Initialize by taking first target_rank columns of schoolbook
        # But better: random subset or structured initialization
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                # Take first target_rank columns
                cols = np.arange(target_rank)
            elif attempt == 1:
                # Random subset
                np.random.seed(target_rank + attempt)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            else:
                # Structured: try to keep diversity
                np.random.seed(target_rank * 7 + attempt)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            
            U_init = U_sb[:, cols].copy()
            V_init = V_sb[:, cols].copy()
            W_init = W_sb[:, cols].copy()
            
            Ur, Vr, Wr, success = try_optimize(target_rank, U_init, V_init, W_init, max_iter=100)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = Ur, Vr, Wr
                break
    
    # Also try direct random initialization for lower ranks
    for target_rank in range(max(28, best_rank - 10), min(best_rank, 50)):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            np.random.seed(target_rank * 13 + attempt)
            U_init = np.random.randn(nm, target_rank) * 0.1
            V_init = np.random.randn(mp, target_rank) * 0.1
            W_init = np.random.randn(pn, target_rank) * 0.1
            
            Ur, Vr, Wr, success = try_optimize(target_rank, U_init, V_init, W_init, max_iter=150)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = Ur, Vr, Wr
                break
    
    # Final verification and cleanup
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = U_sb.shape[1]
    
    # Convert to lists, rounding to nearest half-integer if close
    def to_clean_list(M):
        L = []
        for row in M:
            row_list = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            L.append(row_list)
        return L
    
    return {
        "U": to_clean_list(best_U),
        "V": to_clean_list(best_V),
        "W": to_clean_list(best_W)
    }
