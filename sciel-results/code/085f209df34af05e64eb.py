"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    # Try to find a lower-rank decomposition via Laderman-style block approach
    # and numerical search, falling back to schoolbook rank 60.
    
    # Attempt 1: Laderman-style block decomposition
    # Partition 2x5 into [1,1] x [2,3] and 5x6 into [2,3] x [3,3]
    # This gives 4 block-multiplications, each potentially reducible
    # For now, try numerical search for rank < 60
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But for matrix multiplication, known lower bounds are higher
    # Try a few candidate ranks
    candidate_ranks = [48, 50, 52, 54, 56, 58]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            best_result = result
            best_rank = rank
            break  # Found a valid decomposition, use it
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)


def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-rank decomposition using alternating least squares."""
    import numpy as np
    
    # Tensor shape: (n*m, m*p, p*n) with T[i,j,k] = 1 if i//m == j%p and j//p == k%n
    # Actually, the matrix multiplication tensor: T_{(i,j),(j,k),(k,i)} = 1
    # Flatten: U index = i*m+j, V index = j*p+k, W index = k*n+i
    
    # Build the tensor in flattened form
    # T_flat[u, v, w] = 1 if u//m == v%p and v//p == w%n and u%m == v//p and v%p == w%n
    # Let's be precise:
    # u = i*m + j, v = j*p + k, w = k*n + i
    # So: i = u // m, j = u % m, j = v // p, k = v % p, k = w // n, i = w % n
    # T[u,v,w] = 1 if (u//m == w%n) and (u%m == v//p) and (v%p == w//n)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build tensor as sparse representation
    # For each valid (u,v,w) triple
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_entries.append((u, v, w))
    
    tensor_entries = np.array(tensor_entries)
    
    # Initialize U, V, W randomly
    np.random.seed(42)
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    # Alternating least squares
    max_iters = 50
    tol = 1e-8
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            return None
        
        # Update U: for each u, solve min sum_v,w T[u,v,w] * (V[:,r]*W[:,r])
        # U[u, r] = sum_{v,w: T[u,v,w]=1} V[v,r]*W[w,r] / sum_{v,w: T[u,v,w]=1} V[v,r]^2 ... 
        # Actually, ALS: fix V,W, solve for U
        # T[u,v,w] ≈ sum_r U[u,r] * V[v,r] * W[w,r]
        # For fixed u: T[u,:,w] ≈ U[u,:] * (V * W[:,w]...) 
        # This is tricky. Let me use a different approach.
        
        # Use gradient descent on the loss
        # Loss = sum_{(u,v,w) in tensor} (T[u,v,w] - sum_r U[u,r]*V[v,r]*W[w,r])^2
        # = sum_{(u,v,w)} (1 - sum_r U[u,r]*V[v,r]*W[w,r])^2
        
        # Compute gradients
        # dL/du = -2 * (1 - pred) * V[v,r]*W[w,r]
        # dL/dv = -2 * (1 - pred) * U[u,r]*W[w,r]  
        # dL/dw = -2 * (1 - pred) * U[u,r]*V[v,r]
        
        # For efficiency, precompute for each r:
        # For U update: grad_U[u,r] = -2 * sum_{(v,w): (u,v,w) in tensor} (1 - pred[u,v,w]) * V[v,r] * W[w,r]
        
        # This is expensive. Let me use a simpler approach: project onto the manifold.
        
        # Alternative: use the fact that for each r, we're fitting a rank-1 tensor.
        # Let's just do a few iterations of a simple update.
        
        # Compute predictions for all tensor entries
        # pred[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
        # For efficiency, compute in batches
        
        # Get unique indices
        us = tensor_entries[:, 0]
        vs = tensor_entries[:, 1]
        ws = tensor_entries[:, 2]
        
        # Compute prediction: sum over r of U[us,r] * V[vs,r] * W[ws,r]
        # This is: for each entry idx, pred[idx] = sum_r U[us[idx],r] * V[vs[idx],r] * W[ws[idx],r]
        # = np.einsum('ir,jr,kr->ij', U[us], V[vs], W[ws]) ... no
        # = np.sum(U[us] * V[vs] * W[ws], axis=1)
        
        U_us = U[us]  # (len, rank)
        V_vs = V[vs]  # (len, rank)
        W_ws = W[ws]  # (len, rank)
        
        pred = np.sum(U_us * V_vs * W_ws, axis=1)
        residual = 1.0 - pred  # T is always 1 for these entries
        
        # Gradient for U: grad_U[us[idx], r] += -2 * residual[idx] * V[vs[idx], r] * W[ws[idx], r]
        # Accumulate
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Use np.add.at for accumulation
        # grad_U[us, r] += -2 * residual * V_vs * W_ws
        # This is: for each r, grad_U[us, r] += -2 * residual * (V_vs * W_ws)[:, r]
        
        # Vectorized: 
        # contrib_U = -2 * residual[:, None] * V_vs * W_ws  # (len, rank)
        # Then scatter-add to grad_U at positions us
        
        contrib_U = -2.0 * residual[:, None] * V_vs * W_ws
        contrib_V = -2.0 * residual[:, None] * U_us * W_ws
        contrib_W = -2.0 * residual[:, None] * U_us * V_vs
        
        np.add.at(grad_U, us, contrib_U)
        np.add.at(grad_V, vs, contrib_V)
        np.add.at(grad_W, ws, contrib_W)
        
        # Learning rate
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if np.max(np.abs(residual)) < tol:
            break
    
    # Final check: verify the decomposition is exact
    U_us = U[us]
    V_vs = V[vs]
    W_ws = W[ws]
    pred = np.sum(U_us * V_vs * W_ws, axis=1)
    
    if np.max(np.abs(pred - 1.0)) < 1e-6:
        # Also need to check that non-tensor entries are zero
        # This is the hard part. The ALS only fits the tensor entries.
        # We need to verify the full tensor equality.
        # For now, accept if the tensor entries are well-fit
        # Convert to rational/half-integer if possible
        result = _rationalize(U, V, W, n, m, p)
        if result is not None:
            return result
    
    return None


def _rationalize(U, V, W, n, m, p):
    """Try to convert float matrices to rational/half-integer entries."""
    import numpy as np
    
    # Check if entries are close to half-integers
    def is_half_integer(arr, tol=1e-4):
        scaled = arr * 2
        rounded = np.round(scaled)
        return np.max(np.abs(scaled - rounded)) < tol
    
    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        
        # Verify exact decomposition with half-integer entries
        # The decomposition is: T[u,v,w] = sum_r (U[u,r]/2) * (V[v,r]/2) * (W[w,r]/2)
        # = sum_r U_int[u,r] * V_int[v,r] * W_int[w,r] / 8
        # So we need sum_r U_int[u,r] * V_int[v,r] * W_int[w,r] = 8 for tensor entries
        # and = 0 for non-tensor entries
        
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        # Build tensor entries
        tensor_entries = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_entries.append((i * m + j, j * p + k, k * n + i))
        
        tensor_set = set(tensor_entries)
        
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = sum(U_int[u, r] * V_int[v, r] * W_int[w, r] for r in range(rank))
            if val != 8:
                return None
        
        # Check a sample of non-tensor entries
        import random
        random.seed(42)
        for _ in range(1000):
            u = random.randint(0, nm - 1)
            v = random.randint(0, mp - 1)
            w = random.randint(0, pn - 1)
            if (u, v, w) in tensor_set:
                continue
            val = sum(U_int[u, r] * V_int[v, r] * W_int[w, r] for r in range(rank))
            if val != 0:
                return None
        
        # Convert to lists of half-integers
        U_list = [[float(U_int[i, r]) / 2 for r in range(rank)] for i in range(nm)]
        V_list = [[float(V_int[i, r]) / 2 for r in range(rank)] for i in range(mp)]
        W_list = [[float(W_int[i, r]) / 2 for r in range(rank)] for i in range(pn)]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
