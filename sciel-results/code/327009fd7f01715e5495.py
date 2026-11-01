"""Matrix multiplication 2,5,5 - rank minimization via structured search."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook baseline
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s kill
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact: M[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We need to contract over c and sum over j for the matrix product,
        # but actually we just need T[i,j,k] = 1 for all i,j,k
        
        # Compute T as a 3D array
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.zeros((n, m, p))
        for c in range(rank):
            # U[:, :, c] shape (n, m), V[:, :, c] shape (m, p), W[:, :, c] shape (p, n)
            Uc = U_reshaped[:, :, c]  # (n, m)
            Vc = V_reshaped[:, :, c]  # (m, p)
            Wc = W_reshaped[:, :, c]  # (p, n)
            
            # For each (i,j,k): Uc[i,j] * Vc[j,k] * Wc[k,i]
            # This is a 3D outer product contracted with specific indices
            # Build it as: for each j, Uc[:, j] (n,) * Vc[j, :] (p,) * Wc[:, j]... no
            # Wc[k, i] where i is from Uc and k is from Vc
            # So for fixed j: Uc[:, j] (n,) outer Vc[j, :] (p,) gives (n, p)
            # Then multiply element-wise by Wc.T (n, p) since Wc[k,i] = Wc.T[i,k]
            for j in range(m):
                # Uc[:, j] shape (n,), Vc[j, :] shape (p,)
                outer = np.outer(Uc[:, j], Vc[j, :])  # (n, p)
                # Wc[k, i] -> Wc.T[i, k] shape (n, p)
                T[:, j, :] += outer * Wc.T
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    def als_step(U, V, W, r, n, m, p):
        """One ALS iteration: update each factor by solving least squares."""
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Update U: for each column c, solve for U[:, c]
        # The contribution of column c to T[i,j,k] is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For fixed c, we want to find U[:, c] such that the total T matches 1.
        # This is complex. Instead, use a gradient-based approach.
        
        # Simpler: use coordinate descent on the tensor error
        # For each factor entry, compute gradient and update
        
        return U_new, V_new, W_new
    
    def gradient_descent(U, V, W, r, n_iters, lr, deadline):
        """Gradient descent on the tensor error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # dT/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c] (for the specific i,j,k)
            # But T is defined for all (i,j,k), and we want T[i,j,k] = 1
            # Loss = sum_{i,j,k} (T[i,j,k] - 1)^2
            # dLoss/dU[i*m+j, c] = 2 * sum_{i',j',k': i'=i, j'=j} (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
            # Wait, U[i*m+j, c] only appears in T[i,j,k] for all k (since j is fixed, i is fixed)
            # Actually U[i*m+j, c] appears in T[i,j,k] for all k.
            # dLoss/dU[i*m+j, c] = 2 * sum_k (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
            
            # Compute current T
            U_r = U.reshape(n, m, r)
            V_r = V.reshape(m, p, r)
            W_r = W.reshape(p, n, r)
            
            T = np.zeros((n, m, p))
            for c in range(r):
                for j in range(m):
                    outer = np.outer(U_r[:, j, c], V_r[j, :, c])  # (n, p)
                    T[:, j, :] += outer * W_r[:, j, c].T  # W_r[k, i, c] -> W_r.T[i, k, c]
            
            # Compute gradients for U
            U_grad = np.zeros_like(U)
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        # dLoss/dU[i*m+j, c] = 2 * sum_k (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
                        grad_val = 0.0
                        for k in range(p):
                            grad_val += (T[i, j, k] - 1.0) * V_r[j, k, c] * W_r[k, i, c]
                        U_grad[i*m+j, c] = 2.0 * grad_val
            
            # Compute gradients for V
            V_grad = np.zeros_like(V)
            for c in range(r):
                for j in range(m):
                    for k in range(p):
                        grad_val = 0.0
                        for i in range(n):
                            grad_val += (T[i, j, k] - 1.0) * U_r[i, j, c] * W_r[k, i, c]
                        V_grad[j*p+k, c] = 2.0 * grad_val
            
            # Compute gradients for W
            W_grad = np.zeros_like(W)
            for c in range(r):
                for k in range(p):
                    for i in range(n):
                        grad_val = 0.0
                        for j in range(m):
                            grad_val += (T[i, j, k] - 1.0) * U_r[i, j, c] * V_r[j, k, c]
                        W_grad[k*n+i, c] = 2.0 * grad_val
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
        
        return U, V, W
    
    # Try to find lower rank decompositions
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    rng = np.random.default_rng(42)
    
    # Try ranks from 49 down to 25
    for r in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = rng.standard_normal((n*m, r)) * 0.3
            V = rng.standard_normal((m*p, r)) * 0.3
            W = rng.standard_normal((p*n, r)) * 0.3
            
            # Run gradient descent
            U, V, W = gradient_descent(U, V, W, r, 200, 0.01, deadline)
            
            # Check if we found a valid decomposition
            if verify_fast(U, V, W, r):
                # Round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round, r):
                    best_rank = r
                    best_U, best_V, best_W = U_round.copy(), V_round.copy(), W_round.copy()
                    break
                # Also try without rounding
                if verify(U, V, W, r):
                    best_rank = r
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            # Also check the unrounded version
            if verify_fast(U, V, W, r):
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
