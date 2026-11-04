import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor
    target_indices_u = []
    target_indices_v = []
    target_indices_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                target_indices_u.append(idx_u)
                target_indices_v.append(idx_v)
                target_indices_w.append(idx_w)
                
    target_indices_u = np.array(target_indices_u)
    target_indices_v = np.array(target_indices_v)
    target_indices_w = np.array(target_indices_w)
    
    # Dense target tensor for fast einsum
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    T_target[target_indices_u, target_indices_v, target_indices_w] = 1.0

    def check_exact(U, V, W):
        """Check if U, V, W exactly decompose the target tensor."""
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return False
            
        # Check if entries are integer or half-integer
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        if not np.allclose(U2, np.round(U2), atol=1e-9): return False
        if not np.allclose(V2, np.round(V2), atol=1e-9): return False
        if not np.allclose(W2, np.round(W2), atol=1e-9): return False
        
        return True

    def run_nmtf(rank, iterations=200, seed=42):
        np.random.seed(seed)
        # Initialize with small positive random numbers
        # Scaling matters. If entries are too large, updates can be unstable.
        # If too small, convergence is slow.
        U = np.random.rand(dim_u, rank) * 0.1 + 0.01
        V = np.random.rand(dim_v, rank) * 0.1 + 0.01
        W = np.random.rand(dim_w, rank) * 0.1 + 0.01
        
        epsilon = 1e-12
        
        for it in range(iterations):
            # Update U
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            
            # Num_U[alpha, r] = sum_{beta, gamma} T[alpha, beta, gamma] V[beta, r] W[gamma, r]
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
            
            # Optional: Normalize to prevent overflow/underflow
            # U /= np.linalg.norm(U)
            # V /= np.linalg.norm(V)
            # W /= np.linalg.norm(W)
            
        # Round to half-integers
        U = np.round(U * 2) / 2.0
        V = np.round(V * 2) / 2.0
        W = np.round(W * 2) / 2.0
        
        return U, V, W

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

    # Try ranks from 65 to 79
    # We prioritize lower ranks.
    # We try multiple seeds for each rank.
    # We limit the total number of NMTF runs to stay within time limits.
    # Each run takes ~ (iterations * cost_per_iter).
    # Cost per iter is ~ 3 * (16*20*20) * 2 (mults) ~ 40k ops. 200 iters ~ 8M ops. Very fast.
    # We can try many seeds.
    
    candidate_ranks = list(range(65, 80))
    
    for rank in candidate_ranks:
        for seed in range(10):
            U, V, W = run_nmtf(rank, iterations=200, seed=seed)
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no improvement found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-END
