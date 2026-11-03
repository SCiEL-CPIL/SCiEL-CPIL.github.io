import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((U_dim, rank_sb))
    V_sb = np.zeros((V_dim, rank_sb))
    W_sb = np.zeros((W_dim, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T with half-integer entries."""
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        if not (np.allclose(U, U2/2, atol=tol) and 
                np.allclose(V, V2/2, atol=tol) and 
                np.allclose(W, W2/2, atol=tol)):
            return False
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=tol)
    
    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
    def structured_init(rank, seed=42):
        """Initialize using structured patterns based on the algebraic structure."""
        np.random.seed(seed)
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        # Strategy: For each rank position, assign contributions from multiple (i,j,k) terms
        # Use a pattern that respects the structure
        for r in range(rank):
            # Each rank term can contribute to multiple (i,j,k) combinations
            # Use a cyclic pattern
            base_i = r % n
            base_j = (r // n) % m
            base_k = (r // (n*m)) % p
            
            # Primary contribution
            U[base_i*m + base_j, r] = 1.0
            V[base_j*p + base_k, r] = 1.0
            W[base_k*n + base_i, r] = 1.0
            
            # Add secondary contributions with small coefficients to allow cancellation
            if r < rank - 1:
                next_i = (base_i + 1) % n
                next_j = (base_j + 1) % m
                next_k = (base_k + 1) % p
                
                coeff = 0.5
                U[next_i*m + next_j, r] += coeff
                V[next_j*p + next_k, r] += coeff
                W[next_k*n + next_i, r] += coeff
        
        return U, V, W
    
    def try_gradient_descent(U, V, W, rank, max_iter=3000, lr=0.01, momentum=0.9):
        """Run gradient descent with momentum."""
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        no_improve = 0
        patience = 300
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            loss = compute_loss(U, V, W)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            
            if loss < 1e-10:
                break
            
            # Adaptive learning rate
            if it > 100 and loss > best_loss * 1.5:
                lr *= 0.5
        
        return best_U, best_V, best_W, best_loss
    
    def try_find_lower_rank(rank, num_restarts=3):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 10:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = 42 + rank * 1000 + restart * 7919
            U, V, W = structured_init(rank, seed)
            
            U_opt, V_opt, W_opt, loss = try_gradient_descent(U, V, W, rank)
            
            # Try to snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            if verify_decomposition(U_snap, V_snap, W_snap, tol=1e-8):
                return U_snap, V_snap, W_snap
            
            # Also try the unsnapped version if it's very close
            if loss < 1e-12:
                # Try a final polish with tighter tolerance
                U_final, V_final, W_final, _ = try_gradient_descent(
                    U_opt, V_opt, W_opt, rank, max_iter=500, lr=0.001, momentum=0.8
                )
                U_snap2 = np.round(U_final * 2) / 2
                V_snap2 = np.round(V_final * 2) / 2
                W_snap2 = np.round(W_final * 2) / 2
                if verify_decomposition(U_snap2, V_snap2, W_snap2, tol=1e-8):
                    return U_snap2, V_snap2, W_snap2
        
        return None
    
    # Try ranks from 49 down to 40
    for rank in range(rank_sb - 1, 39, -1):
        if time.time() > deadline:
            break
        
        result = try_find_lower_rank(rank, num_restarts=3)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
