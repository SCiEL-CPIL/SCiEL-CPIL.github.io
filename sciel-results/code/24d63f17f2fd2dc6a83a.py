import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T as a sparse representation
    # T is zero everywhere except at (i*m+j, j*p+k, k*n+i) which is 1
    target_indices_u = []
    target_indices_v = []
    target_indices_w = []
    target_values = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices_u.append(idx_u)
                target_indices_v.append(idx_v)
                target_indices_w.append(idx_w)
                target_values.append(1.0)
                
    target_indices_u = np.array(target_indices_u)
    target_indices_v = np.array(target_indices_v)
    target_indices_w = np.array(target_indices_w)
    target_values = np.array(target_values)
    
    def check_exact(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        # Compute reconstruction for target indices
        # T_recon[idx_u, idx_v, idx_w] = sum_r U[idx_u, r] V[idx_v, r] W[idx_w, r]
        # We can do this efficiently by looping over r or using einsum if dense, but U,V,W are small enough.
        # Let's just compute the full tensor or just the target parts?
        # Full tensor size is 16*20*20 = 6400. This is tiny.
        # We can compute the full reconstructed tensor and compare.
        
        # Reshape U, V, W to (dim, rank)
        # T_recon[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Using einsum: 'ir,jr,kr->ijk'
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Create target tensor
        T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
        T_target[target_indices_u, target_indices_v, target_indices_w] = 1.0
        
        # Check for exact match
        # Due to floating point, we check closeness. But entries are half-integers, so they are exact in float64.
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return False
            
        # Check if entries are integer or half-integer
        # Multiply by 2 and check if integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-9): return False
        if not np.allclose(V2, np.round(V2), atol=1e-9): return False
        if not np.allclose(W2, np.round(W2), atol=1e-9): return False
        
        return True

    def run_nmtf(rank, iterations=100, seed=42):
        np.random.seed(seed)
        # Initialize with small positive random numbers
        U = np.random.rand(dim_u, rank) + 0.1
        V = np.random.rand(dim_v, rank) + 0.1
        W = np.random.rand(dim_w, rank) + 0.1
        
        # Precompute target tensor for faster access if needed, but einsum is fast
        T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
        T_target[target_indices_u, target_indices_v, target_indices_w] = 1.0
        
        epsilon = 1e-10
        
        for it in range(iterations):
            # Update U
            # Numerator: sum_{beta, gamma} T[alpha, beta, gamma] V[beta, r] W[gamma, r]
            # Denominator: sum_{beta, gamma} (sum_{r'} U[alpha, r'] V[beta, r'] W[gamma, r']) V[beta, r] W[gamma, r]
            
            # T_recon = einsum('ir,jr,kr->ijk', U, V, W)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # Num_U[alpha, r] = sum_{beta, gamma} T[alpha, beta, gamma] V[beta, r] W[gamma, r]
            # This is effectively (T * (V r W)) summed over beta, gamma?
            # Let's compute Num_U = einsum('ijk,jr,kr->ir', T_target, V, W)
            Num_U = np.einsum('ijk,jr,kr->ir', T_target, V, W)
            
            # Den_U[alpha, r] = sum_{beta, gamma} T_recon[alpha, beta, gamma] V[beta, r] W[gamma, r]
            Den_U = np.einsum('ijk,jr,kr->ir', T_recon, V, W)
            
            U = U * (Num_U + epsilon) / (Den_U + epsilon)
            
            # Update V
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            Num_V = np.einsum('ijk,ir,kr->jr', T_target, U, W)
            Den_V = np.einsum('ijk,ir,kr->jr', T_recon, U, W)
            V = V * (Num_V + epsilon) / (Den_V + epsilon)
            
            # Update W
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            Num_W = np.einsum('ijk,ir,jr->kr', T_target, U, V)
            Den_W = np.einsum('ijk,ir,jr->kr', T_recon, U, V)
            W = W * (Num_W + epsilon) / (Den_W + epsilon)
            
        # Round to half-integers
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        
        return U, V, W

    # Try to find a solution with rank < 80
    # We start from 80 and go down.
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school))
    V_school = np.zeros((dim_v, rank_school))
    W_school = np.zeros((dim_w, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try ranks from 80 down to 70
    # Note: Running optimization for multiple ranks might be slow.
    # Let's try a few promising ranks.
    # Known: R(4,4,4) is 49? If so, R(4,4,5) might be around 60-70?
    # Let's try 79, 78, 77...
    
    for rank in range(80, 69, -1):
        for seed in [1, 2, 3]:
            U, V, W = run_nmtf(rank, iterations=50, seed=seed)
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no improvement found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
