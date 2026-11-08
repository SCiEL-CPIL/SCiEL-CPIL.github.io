"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 9
    pn = p * n  # 9

    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication: T[i,j,k] = 1 if (i,j,k) corresponds to (A[i,j]*B[j,k]) -> C[i,k]
        # Actually the tensor for matrix multiplication: T_{i,j,k} = delta_{j,j} ... no.
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard decomposition.
        # Wait, let me re-read: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i in [n], j in [m], k in [p].
        # A decomposition of rank r: M[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        r = U.shape[1]
        # Build the tensor via outer product sum
        # Efficient: for each c, add outer product
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # This is tricky - it's not a simple outer product
            # T[i,j,k] += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Target: all ones
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol, rtol=0)

    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using vectorized operations."""
        r = U.shape[1]
        # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We can do this more efficiently by iterating over c and using broadcasting
        T = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # Use einsum: sum over i,j,k of uc[i,j]*vc[j,k]*wc[k,i]
            # But we need element-wise, so:
            # For fixed c: T += uc[:, :, None] * vc[None, :, :] * wc[None, None, :] ... no
            # uc[i,j], vc[j,k], wc[k,i]
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # This is: for each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
            # We can compute this as:
            # temp[i,j,k] = uc[i,j] * vc[j,k] * wc[k,i]
            # = (uc[i,j] * vc[j,k]) * wc[k,i]
            # Let's use broadcasting:
            # uc[i,j,1] * vc[1,j,k] * wc[k,1,i] -> but wc is (p,n) = (k,i)
            # So: uc[:, :, None] * vc[None, :, :] * wc[None, None, :] ... 
            # wc[k,i] needs to be indexed as [k,i] where k is axis 2 and i is axis 0
            # So wc should be transposed to (i,k) or we need to handle indexing
            # wc[k*n+i] = wc[k,i] in 2D
            # We want T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # = uc[i,j,1] * vc[1,j,k] * wc[2,k,0]... 
            # Let me just do it with explicit loops for correctness, but vectorize the inner
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol, rtol=0)

    def verify_vectorized(U, V, W, tol=1e-9):
        """Fully vectorized verification."""
        r = U.shape[1]
        # Reshape
        U3 = U.reshape(n, m, r)  # U[i,j,c]
        V3 = V.reshape(m, p, r)  # V[j,k,c]
        W3 = W.reshape(p, n, r)  # W[k,i,c]
        
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol, rtol=0)

    def to_rational_list(mat):
        """Convert numpy array to list of lists with rational values."""
        r, c = mat.shape
        result = []
        for i in range(r):
            row = []
            for j in range(c):
                val = mat[i, j]
                # Check if it's close to an integer or half-integer
                rounded_int = round(val * 2) / 2
                if abs(val - rounded_int) < 1e-8:
                    row.append(rounded_int)
                else:
                    row.append(val)
            result.append(row)
        return result

    # Start with schoolbook (rank 27)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known: 3x3 matrix multiplication has rank 23 (Strassen-like improvements)
    # Actually, the optimal rank for 3x3 is 23 (Landsberg's bound shows it's not 20, 
    # and there are constructions with rank 23)
    
    # Strategy: Use randomized search / optimization to find lower rank
    # We'll try ranks from 23 down to 20
    
    for target_rank in range(23, 19, -1):
        if time.time() > DEADLINE:
            break
        
        found = False
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > DEADLINE:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Use gradient-based optimization to minimize ||T - 1||^2
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            
            def loss_and_grad(U, V, W):
                r = U.shape[1]
                U3 = U.reshape(n, m, r)
                V3 = V.reshape(m, p, r)
                W3 = W.reshape(p, n, r)
                
                T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                error = T - 1.0
                loss = np.sum(error ** 2)
                
                # Gradients
                # dL/dU[i,j,c] = 2 * error[i,j,k] * V[j,k,c] * W[k,i,c] summed over k
                # dU[i,j,c] = 2 * sum_k error[i,j,k] * V[j,k,c] * W[k,i,c]
                # dV[j,k,c] = 2 * sum_i error[i,j,k] * U[i,j,c] * W[k,i,c]
                # dW[k,i,c] = 2 * sum_j error[i,j,k] * U[i,j,c] * V[j,k,c]
                
                # dU[i,j,c] = 2 * sum_k error[i,j,k] * V3[j,k,c] * W3[k,i,c]
                dU = np.einsum('ijk,jkc,kic->ijc', error, V3, W3) * 2
                dV = np.einsum('ijk,ijc,kic->jkc', error, U3, W3) * 2
                dW = np.einsum('ijk,ijc,jkc->kic', error, U3, V3) * 2
                
                return loss, dU.reshape(nm, r), dV.reshape(mp, r), dW.reshape(pn, r)
            
            # Adam-like optimizer
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            t = 0
            
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            
            for step in range(2000):
                if time.time() > DEADLINE:
                    break
                
                t += 1
                loss, gU, gV, gW = loss_and_grad(U, V, W)
                
                if loss < 1e-10:
                    # Check if it's actually valid
                    if verify_vectorized(U, V, W, tol=1e-8):
                        found = True
                        break
                
                # Adam update
                m_U = beta1 * m_U + (1 - beta1) * gU
                v_U = beta2 * v_U + (1 - beta2) * gU ** 2
                m_U_hat = m_U / (1 - beta1 ** t)
                v_U_hat = v_U / (1 - beta2 ** t)
                
                m_V = beta1 * m_V + (1 - beta1) * gV
                v_V = beta2 * v_V + (1 - beta2) * gV ** 2
                m_V_hat = m_V / (1 - beta1 ** t)
                v_V_hat = v_V / (1 - beta2 ** t)
                
                m_W = beta1 * m_W + (1 - beta1) * gW
                v_W = beta2 * v_W + (1 - beta2) * gW ** 2
                m_W_hat = m_W / (1 - beta1 ** t)
                v_W_hat = v_W / (1 - beta2 ** t)
                
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)
            
            if found:
                # Try to clean up to rational values
                # Round to nearest half-integer
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                if verify_vectorized(U_clean, V_clean, W_clean, tol=1e-8):
                    best_U, best_V, best_W = U_clean, V_clean, W_clean
                    best_rank = target_rank
                    break
                else:
                    # Use the floating point version
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if found:
            break
    
    # Convert to lists
    U_list = to_rational_list(best_U)
    V_list = to_rational_list(best_V)
    W_list = to_rational_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
