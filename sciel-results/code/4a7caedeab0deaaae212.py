import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115  # 115s out of 120s
    
    # Construct the target tensor T
    # T has shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Fallback to schoolbook
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_schoolbook
    
    def compute_T_V(T, n, m, p):
        """Vectorized computation of T_V[j, i*p*n + k] = T[i, j, k]"""
        # T has shape (n*m, m*p, p*n)
        # We need T_V with shape (m*p, n*m*p*n)
        # T_V[j, idx] where idx = i * p * n + k, and T[i, j, k]
        # Reshape T to (n, m, m, p, p, n) then transpose
        T_reshaped = T.reshape(n, m, m, p, p, n)
        # We want output indexed by [j, i, k] -> [m, n, p]
        # T_reshaped[i, j, j2, k, k2, i2] where j=j2, i2=i, k=k2
        # So we need T_reshaped[i, j, j, k, k, i] - but that's a diagonal
        # Actually T[i*m+j, j*p+k, k*n+i] = 1 only when the indices match
        # T_V[j, i*p*n+k] = T[i, j, k]
        # Let's just do it with einsum-style approach
        # T has shape (n*m, m*p, p*n)
        # Flatten i = a*n+b (a in [0,n), b in [0,m)), j = c*p+d (c in [0,m), d in [0,p)), k = e*n+f (e in [0,p), f in [0,n))
        # T_V[j, i*p*n+k] = T[i, j, k]
        # So T_V is T with axes permuted: T[i, j, k] -> T_V[j, i, k]
        # Reshape T to (n, m, m, p, p, n), transpose to (m, n, p, m, p, n), then we need to select where first m = second m, and last n = first n
        # This is getting complex. Let's use a direct approach with numpy.
        
        T_V = np.zeros((m * p, n * m * p * n))
        for j in range(m * p):
            for i in range(n * m):
                for k in range(p * n):
                    T_V[j, i * p * n + k] = T[i, j, k]
        return T_V
    
    def compute_T_W(T, n, m, p):
        """Vectorized computation of T_W[k, i*m*p + j] = T[i, j, k]"""
        T_W = np.zeros((p * n, n * m * m * p))
        for k in range(p * n):
            for i in range(n * m):
                for j in range(m * p):
                    T_W[k, i * m * p + j] = T[i, j, k]
        return T_W
    
    def try_rank(rank, U_init, V_init, W_init, max_iters=200, tol=1e-10):
        """Try to refine a decomposition using ALS. Returns (success, U, V, W)."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Precompute T_V and T_W once
        T_V = np.zeros((m * p, n * m * p * n))
        for j in range(m * p):
            for i in range(n * m):
                for k in range(p * n):
                    T_V[j, i * p * n + k] = T[i, j, k]
        
        T_W = np.zeros((p * n, n * m * m * p))
        for k in range(p * n):
            for i in range(n * m):
                for j in range(m * p):
                    T_W[k, i * m * p + j] = T[i, j, k]
        
        T_U = T.reshape(n * m, m * p * p * n)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return False, U, V, W
            
            # Update U: fix V, W
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V: fix U, W
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Update W: fix U, V
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)
                W = W_new.T
            except:
                break
            
            # Check convergence
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.linalg.norm(T - S) < tol:
                return True, U, V, W
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.linalg.norm(T - S) < 1e-8:
            return True, U, V, W
        return False, U, V, W
    
    def polish(U, V, W, max_iters=100, tol=1e-12):
        """High-precision polish pass"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        T_U = T.reshape(n * m, m * p * p * n)
        T_V = np.zeros((m * p, n * m * p * n))
        for j in range(m * p):
            for i in range(n * m):
                for k in range(p * n):
                    T_V[j, i * p * n + k] = T[i, j, k]
        T_W = np.zeros((p * n, n * m * m * p))
        for k in range(p * n):
            for i in range(n * m):
                for j in range(m * p):
                    T_W[k, i * m * p + j] = T[i, j, k]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)
                U = U_new.T
            except:
                break
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)
                V = V_new.T
            except:
                break
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)
                W = W_new.T
            except:
                break
            S = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.linalg.norm(T - S) < tol:
                break
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.linalg.norm(T - S) < 1e-10:
            return True, U, V, W
        return False, U, V, W
    
    # Try various ranks and initializations
    target_ranks = [48, 40, 36, 32, 28, 24]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        num_restarts = max(5, int(60 / rank))  # More restarts for lower ranks
        for init_seed in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + init_seed * 7)
            
            U_init = np.random.randn(n * m, rank) * 0.3
            V_init = np.random.randn(m * p, rank) * 0.3
            W_init = np.random.randn(p * n, rank) * 0.3
            
            success, U, V, W = try_rank(rank, U_init, V_init, W_init, max_iters=150)
            
            if success:
                # Polish
                polished, U_p, V_p, W_p = polish(U, V, W)
                if polished:
                    U_rounded = np.round(U_p * 2) / 2
                    V_rounded = np.round(V_p * 2) / 2
                    W_rounded = np.round(W_p * 2) / 2
                    S = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(S, T, atol=1e-10):
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
                    if np.allclose(S, T, atol=1e-6):
                        return {
                            "U": U_p.tolist(),
                            "V": V_p.tolist(),
                            "W": W_p.tolist()
                        }
        
        # Try structured initialization: use schoolbook columns
        if rank < rank_schoolbook:
            for trial in range(5):
                if time.time() > deadline:
                    break
                
                np.random.seed(2000 + trial * 13)
                cols = np.random.choice(rank_schoolbook, rank, replace=False)
                U_init = U_school[:, cols]
                V_init = V_school[:, cols]
                W_init = W_school[:, cols]
                
                success, U, V, W = try_rank(rank, U_init, V_init, W_init, max_iters=100)
                
                if success:
                    polished, U_p, V_p, W_p = polish(U, V, W)
                    if polished:
                        U_rounded = np.round(U_p * 2) / 2
                        V_rounded = np.round(V_p * 2) / 2
                        W_rounded = np.round(W_p * 2) / 2
                        S = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                        if np.allclose(S, T, atol=1e-10):
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
    
    return best_result
# EVOLVE-BLOCK-END
