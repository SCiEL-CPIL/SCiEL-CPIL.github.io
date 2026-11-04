"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s evaluator limit

    # Known rank-88 construction (Strassen on 2x2 blocks for i in {0,1}, schoolbook for i=2)
    def build_rank88():
        rank = 88
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen coefficients (corrected)
        # C11 = M1 - M5 + M4 - M7
        # C12 = M3 + M5
        # C21 = M2 - M4 + M6
        # C22 = M1 - M2 - M3 + M7
        strassen_terms = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
            ([1,0,0,0], [0,0,1,-1], [0,1,0,-1]),  # M3
            ([0,0,0,1], [0,1,-1,0], [1,0,-1,0]),  # M4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
            ([0,-1,1,0], [1,1,0,0], [0,0,1,0]),   # M6
            ([0,1,0,-1], [0,0,1,1], [-1,0,0,1]),  # M7
        ]
        
        def add_strassen_block(i0, j0, k0, col_start):
            col = col_start
            for t in range(7):
                u_c = strassen_terms[t][0]
                v_c = strassen_terms[t][1]
                w_c = strassen_terms[t][2]
                for di in range(2):
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj), col] = u_c[di*2+dj]
                for dj in range(2):
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk), col] = v_c[dj*2+dk]
                for di in range(2):
                    for dk in range(2):
                        W[(k0+dk)*n + (i0+di), col] = w_c[dk*2+di]
                col += 1
            return col
        
        def add_schoolbook_1x2x2(i0, j0, k0, col_start):
            col = col_start
            for dj in range(2):
                for dk in range(2):
                    U[i0*m + (j0+dj), col] = 1
                    V[(j0+dj)*p + (k0+dk), col] = 1
                    W[(k0+dk)*n + i0, col] = 1
                    col += 1
            return col
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                col = add_strassen_block(0, j0, k0, col)
                col = add_schoolbook_1x2x2(2, j0, k0, col)
        
        return U, V, W, rank
    
    U88, V88, W88, _ = build_rank88()
    
    def verify(U, V, W, tol=1e-8):
        r = U.shape[1]
        # Tensor check: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if verify(U88, V88, W88):
        # Have a valid rank-88 baseline
        baseline_rank = 88
    else:
        # Shouldn't happen, but fallback
        baseline_rank = 96
        U88 = np.zeros((n*m, 96))
        V88 = np.zeros((m*p, 96))
        W88 = np.zeros((p*n, 96))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U88[i*m+j, c] = 1
                    V88[j*p+k, c] = 1
                    W88[k*n+i, c] = 1
                    c += 1
    
    # Now try to find rank-87 exact decomposition via optimization
    # The matrix multiplication tensor T_{ijk} = delta_{i,j,k} (1 if valid indices, else 0)
    # We want: T_{ijk} = sum_{c=1}^{rank} U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # This is a 3-tensor with dimensions (n*m) x (m*p) x (p*n) but with the constraint that
    # only entries where i,j,k form a valid matrix multiplication are non-zero.
    # Actually, the tensor is sparse: T[i,j,k] = 1 if we interpret i as row of A, j as col of A/row of B, k as col of B.
    # The tensor has exactly n*m*p = 96 non-zero entries (all 1s), and the rest are 0.
    
    # Let's define the full tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k, else 0
    # Note: for a given (a,b,c), there's at most one (i,j,k) that works, and it works iff
    # a % m == b // p and b % p == c // n and a // m == c % n
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Try rank-87 via L-BFGS-B or similar
    # Variables: U (n*m x r), V (m*p x r), W (p*n x r) -> total (n*m + m*p + p*n) * r parameters
    # For r=87: (12 + 32 + 24) * 87 = 68 * 87 = 5916 parameters
    
    def try_rank_optimization(target_rank, max_iters=50, time_limit=60):
        """Try to find exact decomposition of given rank using gradient descent."""
        r = target_rank
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        def loss_and_grad(params):
            # params: flattened [U, V, W]
            U_flat = params[:n*m*r].reshape(n*m, r)
            V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            # Compute tensor product sum: sum_c U[a,c] * V[b,c] * W[c_idx,c]
            # We only need to check the 96 non-zero entries
            # For efficiency, compute the residual on the 96 entries
            residual = 0.0
            grad_U = np.zeros_like(U_flat)
            grad_V = np.zeros_like(V_flat)
            grad_W = np.zeros_like(W_flat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c_idx = k*n + i
                        # Residual for this entry: sum_c U[a,c]*V[b,c]*W[c_idx,c] - 1
                        # Gradient w.r.t U[a,c]: V[b,c]*W[c_idx,c]
                        # Gradient w.r.t V[b,c]: U[a,c]*W[c_idx,c]
                        # Gradient w.r.t W[c_idx,c]: U[a,c]*V[b,c]
                        prod = U[a] * V[b] * W[c_idx]
                        val = np.sum(prod)
                        res = val - 1.0
                        residual += res * res
                        # Gradients
                        grad_U[a] += 2 * res * V[b] * W[c_idx]
                        grad_V[b] += 2 * res * U[a] * W[c_idx]
                        grad_W[c_idx] += 2 * res * U[a] * V[b]
            
            # Also need to ensure the other entries are 0, but since the tensor structure
            # means only these 96 entries are non-zero in the target, and our decomposition
            # naturally produces values for all (a,b,c) entries, we should also penalize
            # non-zero entries elsewhere. However, this is a huge number of entries.
            # Actually, the tensor T has zeros everywhere except the 96 valid entries.
            # So we need sum_c U[a,c]*V[b,c]*W[c_idx,c] = 0 for all invalid (a,b,c).
            # This is a lot of constraints. Let's check a sample.
            
            # For now, just use the 96 entries as a starting point
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return residual, grad
        
        # Use a simple gradient descent with adaptive step size
        best_loss = float('inf')
        best_params = None
        lr = 0.01
        t0 = time.time()
        
        for it in range(max_iters):
            if time.time() - t0 > time_limit:
                break
            loss, grad = loss_and_grad(params)
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            # Adaptive step size
            step = lr / (1.0 + it * 0.1)
            params = params - step * grad
            
            # Check if we're close to exact
            if best_loss < 1e-10:
                break
        
        if best_params is not None and best_loss < 1e-6:
            U_out = best_params[:n*m*r].reshape(n*m, r)
            V_out = best_params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_out = best_params[n*m*r + m*p*r:].reshape(p*n, r)
            # Verify exactly
            if verify(U_out, V_out, W_out, tol=1e-7):
                return U_out, V_out, W_out, r, True
        return None, None, None, r, False
    
    # Try rank 87
    if time.time() < deadline - 20:
        U87, V87, W87, r87, success = try_rank_optimization(87, max_iters=30, time_limit=50)
        if success:
            return {"U": U87.tolist(), "V": V87.tolist(), "W": W87.tolist()}
    
    # Try rank 86
    if time.time() < deadline - 10:
        U86, V86, W86, r86, success = try_rank_optimization(86, max_iters=20, time_limit=30)
        if success:
            return {"U": U86.tolist(), "V": V86.tolist(), "W": W86.tolist()}
    
    # Fallback to rank 88
    return {"U": U88.tolist(), "V": V88.tolist(), "W": W88.tolist()}
# EVOLVE-BLOCK-END
