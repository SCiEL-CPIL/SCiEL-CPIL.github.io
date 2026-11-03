import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    def schoolbook(n_, m_, p_):
        """Trivial decomposition for n_ x m_ x p_"""
        rank = n_ * m_ * p_
        U = np.zeros((n_ * m_, rank))
        V = np.zeros((m_ * p_, rank))
        W = np.zeros((p_ * n_, rank))
        col = 0
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    U[i * m_ + j, col] = 1.0
                    V[j * p_ + k, col] = 1.0
                    W[k * n_ + i, col] = 1.0
                    col += 1
        return U, V, W
    
    def verify_decomposition(U, V, W, T_target, tol=1e-8):
        """Check if U, V, W decompose T_target"""
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)
    
    def try_block_decomposition(p_split):
        """
        Try splitting the p dimension at p_split.
        Block 1: n x m x p_split
        Block 2: n x m x (p - p_split)
        """
        p1 = p_split
        p2 = p - p_split
        
        # Target tensors for each block
        T1 = np.zeros((nm, m * p1, p1 * n))
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    u = i * m + j
                    v = j * p1 + k
                    w = k * n + i
                    T1[u, v, w] = 1.0
        
        T2 = np.zeros((nm, m * p2, p2 * n))
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    u = i * m + j
                    v = j * p2 + k
                    w = k * n + i
                    T2[u, v, w] = 1.0
        
        # Try to find minimal rank for each block
        best_r1 = n * m * p1
        best_r2 = n * m * p2
        best_U1, best_V1, best_W1 = schoolbook(n, m, p1)
        best_U2, best_V2, best_W2 = schoolbook(n, m, p2)
        
        # Try rank reduction for block 1
        for r1 in range(best_r1 - 1, max(8, best_r1 - 5)):
            if time.time() > deadline - 20:
                break
            # Try to find rank r1 decomposition
            found = _find_rank_decomposition(T1, r1, nm, m * p1, p1 * n, deadline)
            if found is not None:
                U1, V1, W1 = found
                best_r1 = r1
                best_U1, best_V1, best_W1 = U1, V1, W1
                break
        
        # Try rank reduction for block 2
        for r2 in range(best_r2 - 1, max(10, best_r2 - 5)):
            if time.time() > deadline - 10:
                break
            found = _find_rank_decomposition(T2, r2, nm, m * p2, p2 * n, deadline)
            if found is not None:
                U2, V2, W2 = found
                best_r2 = r2
                best_U2, best_V2, best_W2 = U2, V2, W2
                break
        
        total_rank = best_r1 + best_r2
        return best_U1, best_V1, best_W1, best_U2, best_V2, best_W2, total_rank
    
    def _find_rank_decomposition(T, rank, dim_u, dim_v, dim_w, deadline):
        """
        Try to find a rank decomposition of T with given rank.
        Uses alternating minimization with multiple restarts.
        """
        best_error = np.inf
        best_UVW = None
        
        n_restarts = 20
        max_iter = 100
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize randomly
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            # Normalize
            U /= (np.linalg.norm(U, axis=0, keepdims=True) + 1e-12)
            V /= (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
            
            prev_error = np.inf
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column r, U[:,r] = argmin ||T - sum_r' U[:,r'] ⊗ V[:,r'] ⊗ W[:,r']||
                # Using ALS: U[:,r] = (T contracted with V[:,r] and W[:,r]) / (V[:,r]^T V[:,r] * W[:,r]^T W[:,r])
                # More precisely: U[:,r] = einsum('uvw,v,w->u', T, V[:,r], W[:,r]) / (np.dot(V[:,r],V[:,r]) * np.dot(W[:,r],W[:,r]))
                
                for r in range(rank):
                    # Compute the effective target for U[:,r]
                    # T_effective_u = sum_{v,w} T[u,v,w] * V[v,r] * W[w,r]
                    T_eff_u = np.einsum('uvw,v,w->u', T, V[:, r], W[:, r])
                    denom = np.dot(V[:, r], V[:, r]) * np.dot(W[:, r], W[:, r])
                    if denom > 1e-12:
                        U[:, r] = T_eff_u / denom
                    
                    # Similarly for V
                    T_eff_v = np.einsum('uvw,u,w->v', T, U[:, r], W[:, r])
                    denom = np.dot(U[:, r], U[:, r]) * np.dot(W[:, r], W[:, r])
                    if denom > 1e-12:
                        V[:, r] = T_eff_v / denom
                    
                    # And W
                    T_eff_w = np.einsum('uvw,u,v->w', T, U[:, r], V[:, r])
                    denom = np.dot(U[:, r], U[:, r]) * np.dot(V[:, r], V[:, r])
                    if denom > 1e-12:
                        W[:, r] = T_eff_w / denom
                
                # Compute error
                T_approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                error = np.linalg.norm(T_approx - T)
                
                if error < best_error:
                    best_error = error
                    best_UVW = (U.copy(), V.copy(), W.copy())
                
                if error < 1e-10:
                    break
                
                if abs(prev_error - error) < 1e-14:
                    break
                prev_error = error
        
        if best_error < 1e-8:
            return best_UVW
        return None
    
    # Try different block splits
    best_total_rank = n * m * p  # 42
    best_result = None
    
    for p_split in [3, 4]:
        if time.time() > deadline - 30:
            break
        
        U1, V1, W1, U2, V2, W2, total_rank = try_block_decomposition(p_split)
        
        if total_rank < best_total_rank:
            # Combine the two block decompositions into one
            R1 = U1.shape[1]
            R2 = U2.shape[1]
            R_total = R1 + R2
            
            # For block 1 (k=0..p_split-1):
            # U1 has shape (nm, R1), V1 has shape (m*p1, R1), W1 has shape (p1*n, R1)
            # For block 2 (k=p_split..p-1):
            # U2 has shape (nm, R2), V2 has shape (m*p2, R2), W2 has shape (p2*n, R2)
            
            # Full U: (nm, R_total)
            U_full = np.zeros((nm, R_total))
            V_full = np.zeros((mp, R_total))
            W_full = np.zeros((pn, R_total))
            
            # Block 1 occupies columns 0..R1-1
            U_full[:, :R1] = U1
            # V1 has shape (m*p1, R1), need to place in V_full rows 0..m*p1-1
            V_full[:m * p_split, :R1] = V1
            # W1 has shape (p1*n, R1), need to place in W_full rows 0..p1*n-1
            W_full[:p_split * n, :R1] = W1
            
            # Block 2 occupies columns R1..R_total-1
            U_full[:, R1:] = U2
            # V2 has shape (m*p2, R2), need to place in V_full rows m*p1..m*p-1
            V_full[m * p_split:, R1:] = V2
            # W2 has shape (p2*n, R2), need to place in W_full rows p1*n..p*n-1
            W_full[p_split * n:, R1:] = W2
            
            # Verify
            if verify_decomposition(U_full, V_full, W_full, T_target, tol=1e-8):
                best_total_rank = total_rank
                best_result = (U_full, V_full, W_full)
        
        if time.time() > deadline - 20:
            break
    
    if best_result is not None:
        U_full, V_full, W_full = best_result
        # Try to clean up: round to nearest half-integer if close
        U_clean = np.round(U_full * 2) / 2
        V_clean = np.round(V_full * 2) / 2
        W_clean = np.round(W_full * 2) / 2
        
        if verify_decomposition(U_clean, V_clean, W_clean, T_target, tol=1e-8):
            return {"U": U_clean.tolist(), "V": V_clean.tolist(), "W": W_clean.tolist()}
        else:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
