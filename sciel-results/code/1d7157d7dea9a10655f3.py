import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute tensor slices for faster verification
    T_flat = T.ravel()
    
    # Initialize best with schoolbook
    rank_school = n * m * p
    U_school = np.zeros((U_dim, rank_school), dtype=np.float64)
    V_school = np.zeros((V_dim, rank_school), dtype=np.float64)
    W_school = np.zeros((W_dim, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T"""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        for a in range(U_dim):
            for b in range(V_dim):
                for c in range(W_dim):
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - T[a, b, c]) > tol:
                        return False
        return True
    
    def round_and_verify(U, V, W):
        """Round to half-integers and verify"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify_decomposition(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None
    
    # Try ranks from 20 to 47 (below schoolbook)
    for target_rank in range(20, rank_school):
        if time.time() > deadline - 15:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 15:
                break
            
            seed = attempt * 1000 + target_rank
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(U_dim, target_rank) * 0.5
            V = np.random.randn(V_dim, target_rank) * 0.5
            W = np.random.randn(W_dim, target_rank) * 0.5
            
            success = False
            for it in range(200):
                if time.time() > deadline - 15:
                    break
                    
                # Fix U and V, solve for W
                # For each c: T[:, :, c] = sum_r (U[:,r] * V[:,r]) * W[r,c]
                # Vectorize: create matrix M of shape (U_dim*V_dim, target_rank)
                # M[(a,b), r] = U[a,r] * V[b,r]
                # Then T_flat[:, c] = M @ W[:, c] for each c
                
                # Build M efficiently
                M = np.zeros((U_dim * V_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(U[:, r], V[:, r])  # (U_dim, V_dim)
                    M[:, r] = outer.ravel()
                
                # Solve for W: for each c, W[:, c] = lstsq(M, T[:, :, c].ravel())
                # Stack all c's: T_2d has shape (U_dim*V_dim, W_dim)
                T_2d = T.reshape(U_dim * V_dim, W_dim)
                
                # Solve M @ W = T_2d where W has shape (target_rank, W_dim)
                # This is: W^T = lstsq(M^T, T_2d^T)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M.T, T_2d.T, rcond=None)
                    W = W_new.T
                except:
                    pass
                
                # Fix V and W, solve for U
                # For each a: T[a, :, :] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # Vectorize: create matrix N of shape (V_dim*W_dim, target_rank)
                # N[(b,c), r] = V[b,r] * W[c,r]
                # Then T[a].ravel() = N @ U[a, :]
                
                N = np.zeros((V_dim * W_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(V[:, r], W[:, r])  # (V_dim, W_dim)
                    N[:, r] = outer.ravel()
                
                # T_3d[a] has shape (V_dim, W_dim), flatten to (V_dim*W_dim,)
                T_3d = T.reshape(U_dim, V_dim * W_dim)
                
                # Solve N @ U^T = T_3d^T where U^T has shape (target_rank, U_dim)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(N.T, T_3d.T, rcond=None)
                    U = U_new.T
                except:
                    pass
                
                # Fix U and W, solve for V
                # For each b: T[:, b, :] = sum_r V[b,r] * (U[:,r] * W[:,r])
                # Vectorize: create matrix P of shape (U_dim*W_dim, target_rank)
                # P[(a,c), r] = U[a,r] * W[c,r]
                # Then T[:, b].ravel() = P @ V[b, :]
                
                P = np.zeros((U_dim * W_dim, target_rank))
                for r in range(target_rank):
                    outer = np.outer(U[:, r], W[:, r])  # (U_dim, W_dim)
                    P[:, r] = outer.ravel()
                
                # T[:, b] has shape (U_dim, W_dim), flatten to (U_dim*W_dim,)
                T_3d_v = T.reshape(V_dim, U_dim * W_dim)
                
                # Solve P @ V^T = T_3d_v^T where V^T has shape (target_rank, V_dim)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(P.T, T_3d_v.T, rcond=None)
                    V = V_new.T
                except:
                    pass
                
                # Check error periodically
                if it % 20 == 0:
                    # Compute residual
                    error = 0.0
                    for a in range(U_dim):
                        for b in range(V_dim):
                            for c in range(W_dim):
                                val = np.dot(U[a, :], V[b, :] * W[c, :])
                                error += (val - T[a, b, c]) ** 2
                    
                    if error < 1e-8:
                        result = round_and_verify(U, V, W)
                        if result is not None:
                            U_r, V_r, W_r = result
                            found = True
                            best_rank = target_rank
                            best_U, best_V, best_W = U_r, V_r, W_r
                            break
                
                if found:
                    break
            
            if found:
                break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
