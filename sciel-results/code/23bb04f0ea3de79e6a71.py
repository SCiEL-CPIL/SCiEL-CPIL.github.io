"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the multiplication tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For random A (n x m) and B (m x p), the product C = A @ B should equal
    # sum_c (U_c^T A) * (V_c^T B) * W_c (broadcast appropriately)
    
    # Actually, the standard formulation:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c U[i*m+j,c]... 
    
    # Let me use the standard verification:
    # C = A @ B where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * ... 
    # Wait, that's not quite right. Let me think more carefully.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 (it's the identity in the appropriate sense).
    # A decomposition of rank r means: T[i,j,k] = sum_{c=1}^r U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for the verification, we just need to check that for all i,j,k:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 if this corresponds to a valid multiplication term
    # Actually, the multiplication tensor is: for A (n x m), B (m x p), C = A*B (n x p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = delta (this is 1 for all i in [0,n), j in [0,m), k in [0,p))
    # Wait no. The tensor for matrix multiplication is T[i,j,k] where the contraction is:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # A rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify by checking random instances.
    rng = np.random.RandomState(42)
    for _ in range(5):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition:
        # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]?
        # No wait. Let me re-derive.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k] / (A[i,j]*B[j,k])?
        # That doesn't make sense. Let me look at the seed code.
        
        # In the seed: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) triple.
        # So sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 only when c corresponds to (i,j,k), else 0.
        # So T[i,j,k] = sum_c ... = 1. Correct.
        
        # The actual matrix multiplication using the decomposition:
        # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j,c]) * (sum_j B[j,k] * V[j*p+k,c]) * W[k*n+i,c]?
        # No, that's not right either. Let me think again.
        
        # Standard: If T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i],
        # then C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_c w_c[k*n+i] (sum_j u_c[i*m+j] A[i,j]) (sum_j v_c[j*p+k] B[j,k])
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_vec = U[:, c]  # shape (n*m,)
            v_vec = V[:, c]  # shape (m*p,)
            w_val = W[:, c]  # shape (p*n,)
            
            # Reshape u_vec to (n, m), v_vec to (m, p)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            
            # sum_j A[i,j] * u_mat[i,j] for each i
            a_u = A * u_mat  # element-wise, shape (n, m)
            a_u_sum = a_u.sum(axis=1)  # shape (n,)
            
            # sum_j B[j,k] * v_mat[j,k] for each k
            b_v = B * v_mat  # element-wise, shape (m, p)
            b_v_sum = b_v.sum(axis=0)  # shape (p,)
            
            # w_vec shape (p*n,), reshape to (p, n)
            w_mat = w_val.reshape(p, n)
            
            # C[i,k] += w_mat[k,i] * a_u_sum[i] * b_v_sum[k]
            C_computed += np.outer(a_u_sum, b_v_sum) * w_mat.T
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True


def _try_random_search(n, m, p, target_rank, time_limit=10, seed=0):
    """Try to find a decomposition of the given rank using random initialization + optimization."""
    rng = np.random.RandomState(seed)
    start_time = time.time()
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of equations. Let's set up the linear system in the variables.
    # Actually, this is nonlinear. Let's use a different approach.
    
    # Fix one matrix and solve for the other two? Still nonlinear.
    
    # Alternative: use the fact that this is a bilinear/trilinear form.
    # Let's try a direct approach: for a fixed rank r, we have:
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    # This is equivalent to: the tensor T (n x m x p) has a decomposition of rank r.
    # We can use iterative methods.
    
    # Let's try: fix W, then for each c, U[:,c] and V[:,c] are related.
    # Actually, let's use a simpler approach: enumerate small ranks and try random init + least squares.
    
    # For rank r, we have r columns in each of U, V, W.
    # The constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's vectorize: create a matrix M of size (n*m*p) x (r) where
    # M[idx, c] = ... hmm, this isn't linear in the variables.
    
    # Let me try a different strategy: since n=3, m=4, p=5 are small,
    # let's try rank from 15 (n*p + m*p + ... no) up to 60.
    # Known lower bounds: rank >= max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # Upper bound (schoolbook): 60
    
    # Let's try to use the "laser method" or just brute force with small ranks.
    
    # Actually, for 3x4x5, the best known is 40 (from Strassen-like methods) or maybe lower.
    # Let's just try random search for various ranks.
    
    for r in range(20, 61):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # Optimize using gradient descent on the loss: sum_{i,j,k} (sum_c U[i*m+j,c]V[j*p+k,c]W[k*n+i,c] - 1)^2
        # This is a non-convex optimization. Let's try a few iterations.
        
        lr = 0.01
        for it in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current T_hat
            T_hat = np.zeros((n, m, p))
            for c in range(r):
                u = U[:, c].reshape(n, m)
                v = V[:, c].reshape(m, p)
                w = W[:, c].reshape(p, n)
                # T_hat[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_hat[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            
            error = T_hat - 1.0
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                # Found a valid decomposition!
                return U.tolist(), V.tolist(), W.tolist()
            
            # Compute gradients (this is expensive, let's approximate)
            # dLoss/dU[i*m+j,c] = 2 * sum_{i',j',k'} error[i',j',k'] * (delta_{i',i} delta_{j',j}) * V[j'*p+k',c] * W[k'*n+i',c]
            # = 2 * sum_k error[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        s = 0.0
                        for k in range(p):
                            s += error[i,j,k] * V[j*p+k, c] * W[k*n+i, c]
                        dU[i*m+j, c] = 2 * s
                
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for i in range(n):
                            s += error[i,j,k] * U[i*m+j, c] * W[k*n+i, c]
                        dV[j*p+k, c] = 2 * s
                
                for k in range(p):
                    for i in range(n):
                        s = 0.0
                        for j in range(m):
                            s += error[i,j,k] * U[i*m+j, c] * V[j*p+k, c]
                        dW[k*n+i, c] = 2 * s
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    return None


def _construct_known_decompositions():
    """Try known lower-rank decompositions for 3x4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank 60
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s margin before 150s limit
    
    # Start with schoolbook (rank 60)
    best_result = _construct_known_decompositions()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known results for small matrix multiplication:
    # 2x2x2: rank 7 (Strassen)
    # 3x3x3: rank 23 (best known)
    # For 3x4x5, we can try various approaches
    
    # Strategy 1: Try random search for ranks from 20 to 59
    for target_rank in range(20, 60):
        if time.time() > deadline:
            break
        
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline:
                break
            
            result = _try_random_search(n, m, p, target_rank, time_limit=5, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break  # Found valid decomposition at this rank, try lower
    
    return best_result
# EVOLVE-BLOCK-END
