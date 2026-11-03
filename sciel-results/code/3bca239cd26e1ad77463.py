import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # Start with schoolbook (rank 72)
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Build the target tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # Shape: (n*m, m*p, p*n) where index mapping:
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    # T[u_idx, v_idx, w_idx] = 1 if they share consistent (i,j,k)
    
    # Actually, the tensor entry T[a, b, c] where:
    #   a = i*m + j  (U row)
    #   b = j*p + k  (V row)
    #   c = k*n + i  (W row)
    # T[a,b,c] = 1 if there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise T[a,b,c] = 0
    
    # We want: sum_c U[a,c]*V[b,c]*W[c_idx,c] = T_target[a,b,c_idx]
    # where T_target[a,b,c_idx] = 1 if a,b,c_idx form a valid (i,j,k) triple, else 0
    
    # Let's build the target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def hals_decompose(target_tensor, rank, n_iter=200, seed=0):
        """
        HALS (Hierarchical Alternating Least Squares) for tensor decomposition.
        Minimizes ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2
        Returns (U, V, W) arrays of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        """
        rng = np.random.RandomState(seed)
        nU, nV, nW = target_tensor.shape
        
        # Initialize with random small values
        U = rng.randn(nU, rank) * 0.1
        V = rng.randn(nV, rank) * 0.1
        W = rng.randn(nW, rank) * 0.1
        
        for it in range(n_iter):
            # Update U: for each r, U[:,r] = argmin ||T - U⊗V_r⊗W_r - sum_{r'≠r} U_r'⊗V_r'⊗W_r'||
            # Equivalent: U[:,r] = (T * (V_r ⊙ W_r)) / (V_r^T V_r * W_r^T W_r) approximately
            # More precisely, using mode-n matricizations:
            # T_(1) = reshape T to (nU, nV*nW)
            # For rank-r term: T_(1)[:, r] = U[:,r] * (V_r ⊗ W_r)^T
            # So U[:,r] = T_(1)[:,r] / (V_r ⊗ W_r)  (element-wise division, with guard)
            
            # Mode-1 matricization
            T1 = target_tensor.reshape(nU, nV * nW)
            # Mode-2 matricization  
            T2 = target_tensor.transpose(1, 0, 2).reshape(nV, nU * nW)
            # Mode-3 matricization
            T3 = target_tensor.transpose(2, 0, 1).reshape(nW, nU * nV)
            
            # Update U
            for r in range(rank):
                # V_r ⊗ W_r has shape (nV*nW,)
                vw = np.kron(V[:, r], W[:, r])
                # U[:,r] should satisfy: T1[:,r] ≈ U[:,r] * vw
                # Least squares: U[:,r] = T1[:,r] / vw (where vw != 0)
                mask = np.abs(vw) > 1e-12
                U_new = np.zeros(nU)
                if np.any(mask):
                    U_new[mask] = T1[mask, r] / vw[mask]
                # Normalize to prevent divergence
                norm = np.linalg.norm(U_new)
                if norm > 1e-12:
                    U_new = U_new / norm
                U[:, r] = U_new
            
            # Update V
            for r in range(rank):
                uw = np.kron(U[:, r], W[:, r])
                mask = np.abs(uw) > 1e-12
                V_new = np.zeros(nV)
                if np.any(mask):
                    V_new[mask] = T2[mask, r] / uw[mask]
                norm = np.linalg.norm(V_new)
                if norm > 1e-12:
                    V_new = V_new / norm
                V[:, r] = V_new
            
            # Update W
            for r in range(rank):
                uv = np.kron(U[:, r], V[:, r])
                mask = np.abs(uv) > 1e-12
                W_new = np.zeros(nW)
                if np.any(mask):
                    W_new[mask] = T3[mask, r] / uv[mask]
                norm = np.linalg.norm(W_new)
                if norm > 1e-12:
                    W_new = W_new / norm
                W[:, r] = W_new
            
            # Check convergence
            if it % 20 == 0:
                residual = np.linalg.norm(target_tensor - np.einsum('ir,jr,kr->ijk', U, V, W))
                if residual < 1e-10:
                    break
        
        return U, V, W
    
    def hals_decompose_v2(target_tensor, rank, n_iter=300, seed=0):
        """
        Improved HALS using proper matricization approach.
        For each factor, solve a linear system.
        """
        rng = np.random.RandomState(seed)
        nU, nV, nW = target_tensor.shape
        
        U = rng.randn(nU, rank) * 0.1
        V = rng.randn(nV, rank) * 0.1
        W = rng.randn(nW, rank) * 0.1
        
        for it in range(n_iter):
            # Update U: For each r, minimize ||T - sum_r U_r⊗V_r⊗W_r||
            # Fix V and W, update U column r:
            # T_(1)[:,r] = U[:,r] * (V_r ⊙ W_r)
            # where ⊙ is Hadamard product, but in matricized form:
            # T_(1) = (U ⊙ V ⊙ W) where we reshape appropriately
            
            # Actually, let's use the standard HALS formulation:
            # T_(1) = U (W⊙V)^T  where (W⊙V) is the Khatri-Rao product
            # Khatri-Rao of W and V: matrix of shape (nW*nV, rank) where col r = W[:,r] ⊗ V[:,r]
            
            # Update U:
            # U = T_(1) @ (kr(W,V))^+  (pseudoinverse)
            kr_WV = np.zeros((nW * nV, rank))
            for r in range(rank):
                kr_WV[:, r] = np.kron(W[:, r], V[:, r])
            T1 = target_tensor.reshape(nU, nV * nW)
            # Note: T1 column indexing: T1[i, j*nW+k] = T[i,j,k]
            # kr_WV column r: [W[0,r]*V[0,r], W[0,r]*V[1,r], ..., W[nW-1,r]*V[nV-1,r]]
            # This matches: (W⊗V)[j*nW+k] = W[j,r]*V[k,r] for the r-th column
            # But T1[i, j*nW+k] should equal sum_r U[i,r]*W[j,r]*V[k,r]
            # So T1 = U @ (W⊗V)^T  where (W⊗V) has rows indexed by (j,k)
            # kr_WV as defined has kr_WV[j*nW+k, r] = W[j,r]*V[k,r]
            # So T1 = U @ kr_WV.T
            # U = T1 @ pinv(kr_WV.T) = T1 @ pinv(kr_WV).T
            
            try:
                U_new = T1 @ np.linalg.pinv(kr_WV.T)
            except:
                U_new = T1 @ np.linalg.pinv(kr_WV).T
            U = U_new
            
            # Update V:
            # T_(2) = V @ (U⊗W)^T  where T_(2)[j, i*nW+k] = T[i,j,k]
            kr_UW = np.zeros((nU * nW, rank))
            for r in range(rank):
                kr_UW[:, r] = np.kron(U[:, r], W[:, r])
            # T_(2) has shape (nV, nU*nW): T2[j, i*nW+k] = T[i,j,k]
            T2 = target_tensor.transpose(1, 0, 2).reshape(nV, nU * nW)
            # T2 = V @ kr_UW.T
            try:
                V_new = T2 @ np.linalg.pinv(kr_UW.T)
            except:
                V_new = T2 @ np.linalg.pinv(kr_UW).T
            V = V_new
            
            # Update W:
            # T_(3) = W @ (U⊗V)^T  where T_(3)[k, i*nV+j] = T[i,j,k]
            kr_UV = np.zeros((nU * nV, rank))
            for r in range(rank):
                kr_UV[:, r] = np.kron(U[:, r], V[:, r])
            T3 = target_tensor.transpose(2, 0, 1).reshape(nW, nU * nV)
            # T3 = W @ kr_UV.T
            try:
                W_new = T3 @ np.linalg.pinv(kr_UV.T)
            except:
                W_new = T3 @ np.linalg.pinv(kr_UV).T
            W = W_new
            
            # Check convergence
            if it % 50 == 0:
                recon = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = np.linalg.norm(target_tensor - recon)
                if residual < 1e-10:
                    break
        
        return U, V, W
    
    def try_rank_with_hals(target, rank, n_seeds=5, n_iter=200):
        """Try to find exact decomposition at given rank using HALS from multiple seeds."""
        for seed in range(n_seeds):
            if time.time() > deadline:
                return None
            U, V, W = hals_decompose_v2(target, rank, n_iter=n_iter, seed=seed)
            
            # Check if we got an exact (or near-exact) decomposition
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.max(np.abs(recon - target))
            
            if residual < 1e-8:
                # Snap to nearest integers or half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify
                recon2 = np.einsum('ir,jr,kr->ijk', U_snap, V_snap, W_snap)
                if np.max(np.abs(recon2 - target)) < 1e-9:
                    # Convert to lists
                    U_list = [list(U_snap[i]) for i in range(len(U_snap))]
                    V_list = [list(V_snap[i]) for i in range(len(V_snap))]
                    W_list = [list(W_snap[i]) for i in range(len(W_snap))]
                    if verify(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try HALS with decreasing ranks
    # Start from the Strassen-based bound of 66 and go down
    start_rank = 66
    for rank in range(start_rank, 40, -1):
        if time.time() > deadline - 30:
            break
        
        result = try_rank_with_hals(target, rank, n_seeds=3, n_iter=150)
        if result is not None:
            best = result
            best_rank = rank
            print(f"HALS found rank {rank}")
            break
    
    # Also try a few specific ranks with more seeds
    for rank in [50, 48, 45, 42, 40]:
        if time.time() > deadline - 15:
            break
        result = try_rank_with_hals(target, rank, n_seeds=5, n_iter=200)
        if result is not None and rank < best_rank:
            best = result
            best_rank = rank
            print(f"HALS found rank {rank}")
    
    # Fallback: try the block-structured Strassen approach as backup
    if best_rank > 60:
        def strassen_2x2x2():
            terms = []
            terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
            terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
            terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
            terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
            terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
            terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
            terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
            return terms
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        n_blocks = [[0, 1], [2]]
        m_blocks = [[0, 1], [2, 3]]
        p_blocks = [[0, 1], [2, 3], [4, 5]]
        
        strassen_terms = strassen_2x2x2()
        
        for ni, n_blk in enumerate(n_blocks):
            for mi, m_blk in enumerate(m_blocks):
                for pi, p_blk in enumerate(p_blocks):
                    if len(n_blk) == 2 and len(m_blk) == 2 and len(p_blk) == 2:
                        for u_vals, v_vals, w_vals in strassen_terms:
                            U_cols.append(u_vals)
                            V_cols.append(v_vals)
                            W_cols.append(w_vals)
                    else:
                        for k_idx in range(2):
                            for j_idx in range(2):
                                u_vals = [0, 0]
                                u_vals[k_idx] = 1
                                v_vals = [0, 0, 0, 0]
                                v_vals[k_idx * 2 + j_idx] = 1
                                w_vals = [0, 0]
                                w_vals[j_idx] = 1
                                U_cols.append(u_vals)
                                V_cols.append(v_vals)
                                W_cols.append(w_vals)
        
        rank_found = len(U_cols)
        
        U_mat = np.zeros((n * m, rank_found))
        V_mat = np.zeros((m * p, rank_found))
        W_mat = np.zeros((p * n, rank_found))
        
        for c in range(rank_found):
            for idx, val in enumerate(U_cols[c]):
                if val != 0:
                    U_mat[idx, c] = val
            for idx, val in enumerate(V_cols[c]):
                if val != 0:
                    V_mat[idx, c] = val
            for idx, val in enumerate(W_cols[c]):
                if val != 0:
                    W_mat[idx, c] = val
        
        U_res = [list(map(int, U_mat[i])) for i in range(n * m)]
        V_res = [list(map(int, V_mat[i])) for i in range(m * p)]
        W_res = [list(map(int, W_mat[i])) for i in range(p * n)]
        
        if verify(U_res, V_res, W_res, n, m, p) and rank_found < best_rank:
            best = {"U": U_res, "V": V_res, "W": W_res}
            best_rank = rank_found
    
    return best
