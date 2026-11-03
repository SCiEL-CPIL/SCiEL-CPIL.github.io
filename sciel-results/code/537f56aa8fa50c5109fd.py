import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses schoolbook decomposition (rank 100) as guaranteed fallback,
    then attempts rank reduction via gradient descent on a smooth surrogate objective.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate(U, V, W, n, m, p, rank):
        """
        Validate that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all i in [0,n), j in [0,m), k in [0,p).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        return False
        return True

    def try_gradient_descent(rank, time_budget):
        """
        Attempt to find a decomposition of given rank using gradient descent.
        """
        nonlocal best_rank, best_U, best_V, best_W
        
        # Reshape for easier index handling in gradient calculation
        # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
        
        # Precompute indices for target tensor T (all 1s)
        # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        
        # Random initialization
        # Scale matters. If entries are 0 or 1, products are 0 or 1.
        # Let's start with random values in [-1, 1]
        
        max_iterations = 500
        lr = 0.01
        
        for attempt in range(10): # 10 restarts
            if time.time() > deadline:
                break
                
            U = np.random.uniform(-1, 1, (n, m, rank))
            V = np.random.uniform(-1, 1, (m, p, rank))
            W = np.random.uniform(-1, 1, (p, n, rank))
            
            # Compute gradient of Loss = sum (S_ijk - 1)^2
            # S_ijk = sum_r U_ijr V_jkr W_kir
            
            for it in range(max_iterations):
                if time.time() > deadline:
                    break
                
                # Compute S
                # Use einsum for speed
                # U is (n, m, r), V is (m, p, r), W is (p, n, r)
                # S[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
                # This is tricky with einsum because of the index wrapping.
                # Let's loop over r to build S efficiently?
                # S = 0
                # for r in range(rank):
                #    U_r = U[:,:,r] (n,m)
                #    V_r = V[:,:,r] (m,p)
                #    W_r = W[:,:,r] (p,n)
                #    # We need to combine these to get (n,m,p) tensor where S_ijk = U_ij V_jk W_ki
                #    # This is not a standard matmul.
                #    # S_ijk = U[i,j] * V[j,k] * W[k,i]
                #    # We can do this by broadcasting?
                #    # U[i,j,0] * V[0,j,k] * W[k,0,i] -> shape (n, m, p)
                #    # U_exp = U_r[:, :, np.newaxis] # (n, m, 1)
                #    # V_exp = V_r[np.newaxis, :, :] # (1, m, p)
                #    # W_exp = W_r.T[np.newaxis, np.newaxis, :] # (1, 1, n) -> wait, W[k,i] so we need W.T[i,k]? No, W is (p,n). W[k,i] is element.
                #    # We want S[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
                #    # Let's use np.einsum('ijr,jkr,kir->ijk', U, V, W)
                
                S = np.einsum('ijr,jkr,kir->ijk', U, V, W, optimize=True)
                
                # Loss
                loss = np.sum((S - 1)**2)
                
                # Gradient
                # dL/dS = 2(S - 1)
                # dS_ijk / dU_ijr = V_jkr * W_kir
                # dS_ijk / dV_jkr = U_ijr * W_kir
                # dS_ijk / dW_kir = U_ijr * V_jkr
                
                # grad_U[i,j,r] = sum_{k} dL/dS_ijk * V[j,k,r] * W[k,i,r]
                # grad_U = np.einsum('ijk,jkr,kir->ijr', (S-1), V, W)
                # Note: dL/dS is 2*(S-1). We can absorb 2 into lr.
                
                dS = S - 1
                grad_U = np.einsum('ijk,jkr,kir->ijr', dS, V, W, optimize=True)
                grad_V = np.einsum('ijk,ijr,kir->jkr', dS, U, W, optimize=True)
                grad_W = np.einsum('ijk,ijr,jkr->kir', dS, U, V, optimize=True)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check for convergence
                if np.abs(loss) < 1e-10:
                    break
            
            # After GD, check if we can round to integers/half-integers
            # Try rounding to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check exactness
            # Convert to lists for validation? Or check with numpy
            S_check = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
            if np.allclose(S_check, 1, atol=1e-9):
                # Success!
                if rank < best_rank:
                    best_rank = rank
                    # Convert to lists of floats/ints
                    best_U = U_r.reshape(n*m, rank).tolist()
                    best_V = V_r.reshape(m*p, rank).tolist()
                    best_W = W_r.reshape(p*n, rank).tolist()
                    return True
        
        return False

    # Try ranks from 99 down to 50
    for rank in range(base_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        if try_gradient_descent(rank, time_budget=5.0):
            break

    return {"U": best_U, "V": best_V, "W": best_W}
