import numpy as np
import time

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    # Ensure symmetry
    adj = (adj + adj.T) // 2
    # Zero diagonal
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_C = -1
    best_x = None
    
    S_total = np.sum(adj)
    
    def get_cut(x):
        # x is -1/1
        xAx = x @ (adj @ x)
        return 0.25 * (S_total - xAx)
    
    # Spectral Initialization using Laplacian Fiedler vector
    D = np.diag(adj.sum(axis=1))
    L = D - adj
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        x_spec = np.sign(fiedler)
        x_spec[x_spec == 0] = 1
    except:
        rng = np.random.default_rng(42)
        x_spec = rng.choice([-1, 1], size=n)
    
    rng = np.random.default_rng(42)
    
    def local_search_1flip(x, S=None):
        # x is -1/1 array
        if S is None:
            S = adj @ x
        C = get_cut(x)
        
        improved = True
        while improved:
            improved = False
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            
            if max_gain > 0:
                v = x[i]
                x[i] = -v
                C += max_gain
                S -= 2 * v * adj[:, i]
                improved = True
        
        return x, C, S
    
    def local_search_2swap(x, S=None):
        if S is None:
            S = adj @ x
        C = get_cut(x)
        
        improved = True
        while improved:
            improved = False
            gains_1d = x * S
            # G[i,j] = gain of flipping i and j
            # Gain(i) = x_i S_i
            # Gain(j) = x_j S_j
            # Interaction term: -2 A_ij x_i x_j
            G = gains_1d[:, None] + gains_1d[None, :] - 2 * adj * (x[:, None] * x[None, :])
            np.fill_diagonal(G, -1)
            
            max_idx = np.unravel_index(np.argmax(G), G.shape)
            i, j = max_idx
            max_pair_gain = G[i, j]
            
            if max_pair_gain > 0:
                vi = x[i]
                vj = x[j]
                x[i] = -vi
                x[j] = -vj
                C += max_pair_gain
                S -= 2 * vi * adj[:, i]
                S -= 2 * vj * adj[:, j]
                improved = True
        
        return x, C, S
    
    def full_local_search(x):
        x, C, S = local_search_1flip(x)
        x, C, S = local_search_2swap(x, S)
        x, C, S = local_search_1flip(x, S)
        return x, C
    
    # 1. Spectral Initialization
    x_init = x_spec.copy()
    x_ls, C_ls = full_local_search(x_init)
    if C_ls > best_C:
        best_C = C_ls
        best_x = x_ls.copy()
    
    # 2. Goemans-Williamson Style SDP Relaxation Initialization
    # We approximate the SDP solution using a spectral embedding.
    # The SDP relaxation of MaxCut involves finding a matrix Y with Y_ii = 1 and Y_ij >= -1.
    # The objective is 0.25 * (S_total - Tr(A Y)).
    # A common heuristic is to use the eigenvectors of A or L.
    # Here we use the top few eigenvectors of the Laplacian to create a low-rank embedding,
    # then project onto random hyperplanes. This mimics the randomized rounding of the SDP solution.
    
    # Get top k eigenvectors of L (excluding the trivial one)
    k = min(10, n - 1)
    try:
        # We already have vals, vecs from above.
        # Use the next k-1 eigenvectors after the first one? Or just top k largest?
        # Fiedler vector is index 1. Let's use indices 1 to k.
        V = vecs[:, 1:k+1]
        # Normalize rows to unit length to mimic Y_ii = 1?
        # In SDP, Y_ij = v_i . v_j. If we use eigenvectors of L, they are orthogonal.
        # Let's try using the eigenvectors of A instead, as they relate to the correlation.
        # Actually, the standard GW uses the solution Y* to the SDP.
        # Since we can't solve the SDP exactly, we use a spectral approximation.
        # A known approximation is to use the eigenvectors of the Laplacian.
        
        # Let's compute the embedding matrix E where E_ij = sum_{l=1}^k V_il V_jl
        # This is V V^T.
        # But we want vectors v_i such that v_i . v_j approximates Y_ij.
        # If we take v_i as the row of V, then v_i . v_j = (V V^T)_ij.
        # However, the norm of v_i is ||V_i||^2 = sum_l V_il^2.
        # For the SDP constraint Y_ii=1, we should normalize the rows.
        
        norms = np.linalg.norm(V, axis=1)
        norms[norms == 0] = 1
        V_norm = V / norms[:, None]
        
        # Now v_i = V_norm[i]
        # The "SDP" value would be Tr(A V_norm V_norm^T) = sum_ij A_ij (V_norm V_norm^T)_ij
        # This is equivalent to x^T A x if we were in 1D.
        
        # Randomized Hyperplane Projection:
        # Generate random vectors u in R^k.
        # Assign x_i = sign(u . v_i).
        
        num_hyperplanes = 20
        for _ in range(num_hyperplanes):
            if time.time() > deadline - 5:
                break
            
            u = rng.standard_normal(k)
            # Projection
            x_proj = np.sign(V_norm @ u)
            x_proj[x_proj == 0] = 1
            
            x_ls, C_ls = full_local_search(x_proj)
            if C_ls > best_C:
                best_C = C_ls
                best_x = x_ls.copy()
                
    except:
        pass
    
    # 3. Multi-start with random and perturbed initializations
    # Continue with the original strategy, which is robust.
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if i % 3 == 0:
            x_curr = rng.choice([-1, 1], size=n)
        elif i % 3 == 1:
            x_curr = best_x.copy()
            num_flips = max(1, n // 10)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        else:
            x_curr = best_x.copy()
            num_flips = max(1, n // 20)
            flip_idx = rng.choice(n, size=num_flips, replace=False)
            x_curr[flip_idx] *= -1
        
        x_ls, C_ls = full_local_search(x_curr)
        
        if C_ls > best_C:
            best_C = C_ls
            best_x = x_ls.copy()
    
    if best_x is None:
        best_x = rng.choice([-1, 1], size=n)
        
    labels = (best_x + 1) // 2
    return labels.astype(int)
