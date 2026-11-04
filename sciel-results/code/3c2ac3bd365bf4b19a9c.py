"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 145  # leave 5s margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    # Precompute target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j
                b = j*p+k
                c = k*n+i
                T[a, b, c] = 1.0

    def verify_decomposition_full(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # Compute tensor product
        # T_comp[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        # We can do this efficiently
        # For each d, add U[:,d] outer V[:,d] outer W[:,d]
        T_comp = np.zeros((dim_u, dim_v, dim_w))
        for d in range(U.shape[1]):
            T_comp += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
        
        # Check non-zero entries
        if not np.allclose(T_comp, T, atol=tol, rtol=0):
            return False
        return True

    def compute_loss(U, V, W):
        """Compute the total squared error over all tensor entries."""
        T_comp = np.zeros((dim_u, dim_v, dim_w))
        for d in range(U.shape[1]):
            T_comp += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
        return np.sum((T_comp - T)**2)

    def fast_als(U, V, W, max_iters=500, tol=1e-12):
        """Perform fast alternating least squares."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        # Precompute T slices for efficiency in g calculation
        # T is (dim_u, dim_v, dim_w)
        
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U
            # H = (V^T V) o (W^T W)
            SV = V.T @ V
            SW = W.T @ W
            H = SV * SW # Hadamard product
            
            # Add small regularization for stability
            H += 1e-8 * np.eye(rank)
            
            # Compute g_a for all a
            # g_a[d] = sum_{b,c} V[b,d] W[c,d] T[a,b,c]
            # We can compute M_bc_d = V[b,d] W[c,d] for all b,c,d
            # M shape: (dim_v, dim_w, rank)
            M = np.einsum('bd,cd->bcd', V, W)
            # g[a,d] = sum_{b,c} M[b,c,d] T[a,b,c]
            # g shape: (dim_u, rank)
            g = np.einsum('bcd,abc->ad', M, T)
            
            # Solve H U_a = g_a for all a
            # U shape: (dim_u, rank)
            # H shape: (rank, rank)
            # We need to solve H X = g.T for each column? No.
            # For each a, U[a,:] is the solution to H u = g[a,:]
            # So U = (H^{-1} g.T).T = (g H^{-1}).T?
            # Let's use np.linalg.solve
            try:
                # g is (dim_u, rank). We want U (dim_u, rank).
                # H u_a = g_a => u_a = H^{-1} g_a
                # U = (H^{-1} @ g.T).T
                U = (np.linalg.solve(H, g.T)).T
            except:
                pass

            # Update V
            SU = U.T @ U
            SW = W.T @ W
            H = SU * SW
            H += 1e-8 * np.eye(rank)
            
            # g_b[d] = sum_{a,c} U[a,d] W[c,d] T[a,b,c]
            # M_ac_d = U[a,d] W[c,d]
            M = np.einsum('ad,cd->acd', U, W)
            g = np.einsum('acd,abc->bd', M, T)
            
            try:
                V = (np.linalg.solve(H, g.T)).T
            except:
                pass

            # Update W
            SU = U.T @ U
            SV = V.T @ V
            H = SU * SV
            H += 1e-8 * np.eye(rank)
            
            # g_c[d] = sum_{a,b} U[a,d] V[b,d] T[a,b,c]
            M = np.einsum('ad,bd->abd', U, V)
            g = np.einsum('abd,abc->cd', M, T)
            
            try:
                W = (np.linalg.solve(H, g.T)).T
            except:
                pass
            
            loss = compute_loss(U, V, W)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        
        return U, V, W, prev_loss

    # Try ranks from schoolbook down to 30
    # Start with schoolbook, then try lower
    for rank in range(schoolbook_rank, 29, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Number of trials depends on rank
        # Lower ranks are harder, so maybe fewer trials? Or more?
        # Let's do 20 trials for each rank
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialization
            # Start with schoolbook structure if rank == schoolbook_rank
            if rank == schoolbook_rank:
                U = U_sb.copy()
                V = V_sb.copy()
                W = W_sb.copy()
                # Add small noise to escape local minima?
                # Or just use it directly.
                # Actually, if we start with schoolbook, we are already at a local minimum (global for rank 54?).
                # We want to find rank < 54.
                # So for rank < 54, we need a different init.
            else:
                # Random init
                rng = np.random.RandomState(trial * 1000 + rank)
                U = rng.randn(dim_u, rank) * 0.1
                V = rng.randn(dim_v, rank) * 0.1
                W = rng.randn(dim_w, rank) * 0.1
                
                # Alternative init: Project schoolbook onto lower rank?
                # Truncate SVD?
                # Let's stick to random for now, maybe add structured init later if needed.
            
            U, V, W, loss = fast_als(U, V, W, max_iters=100, tol=1e-14)
            
            if loss < 1e-8:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if verify_decomposition_full(U_r, V_r, W_r, tol=1e-10):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    best_rank = rank
                    found = True
                    break
                else:
                    # Check if unrounded is "close enough" for a different check?
                    # No, must be exact half-integers.
                    # Try a few more iterations or different init?
                    pass
        
        if found:
            # Found a valid decomposition at this rank.
            # Since we are iterating downwards, this is the best so far.
            # We can continue to try lower ranks if time permits.
            continue

    return best_result
# EVOLVE-BLOCK-END
