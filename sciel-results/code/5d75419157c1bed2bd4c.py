import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 115
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(n * 0 + m):  # j in 0..m-1
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    def verify(U, V, W, atol=1e-9):
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=atol)
    
    def try_als(U, V, W, max_iters=200, tol=1e-10):
        """Alternating Least Squares for tensor decomposition."""
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        rank = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: fix V, W
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed V,W: T_reshaped = U @ M_U.T
            # M_U[r, (j,k)] = V[j,r] * W[k,r]
            M_U = np.einsum('jr,kr->rjk', V, W).reshape(rank, m * p * p * n)
            T_U = T.reshape(n * m, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(M_U.T, T_U.T, rcond=None)[0].T
                U = U_new
            except:
                break
            
            # Update V: fix U, W
            M_V = np.einsum('ir,kr->rik', U, W).reshape(rank, n * m * p * n)
            T_V = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(M_V.T, T_V.T, rcond=None)[0].T
                V = V_new
            except:
                break
            
            # Update W: fix U, V
            M_W = np.einsum('ir,jr->rij', U, V).reshape(rank, n * m * m * p)
            T_W = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(M_W.T, T_W.T, rcond=None)[0].T
                W = W_new
            except:
                break
            
            # Check convergence
            if it % 10 == 0:
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                residual = np.linalg.norm(T - S)
                if residual < tol:
                    break
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(S, T, atol=1e-8):
            return U, V, W
        return None
    
    # Strategy: Try structured initializations that exploit the tensor's product structure
    # The 3x4x8 multiplication can be seen as: for each i in 0..2, compute row i of A times B
    # This gives 3 independent 4x8 multiplications, but they share the same B
    
    # Key insight: use the fact that m=4=2*2, p=8=2*4
    # We can try to decompose using Kronecker products of smaller tensors
    
    # Try various ranks with structured initialization
    target_ranks = [48, 40, 36, 32, 28]
    
    for rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Multiple structured initialization strategies
        init_strategies = [
            'random',
            'structured_kronecker', 
            'sparse_pattern',
            'fourier_based'
        ]
        
        for strategy in init_strategies:
            if time.time() > deadline:
                break
            
            for seed in range(30):
                if time.time() > deadline:
                    break
                
                np.random.seed(seed)
                
                if strategy == 'random':
                    U = np.random.randn(n * m, rank) * 0.1
                    V = np.random.randn(m * p, rank) * 0.1
                    W = np.random.randn(p * n, rank) * 0.1
                    
                elif strategy == 'structured_kronecker':
                    # Exploit that m=4=2*2, p=8=2*4
                    # Initialize with Kronecker product structure
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    
                    # Split into blocks
                    for r in range(rank):
                        # Random signs to break symmetry
                        sign_u = np.random.choice([-1, 1], size=n * m)
                        sign_v = np.random.choice([-1, 1], size=m * p)
                        sign_w = np.random.choice([-1, 1], size=p * n)
                        
                        # Sparse pattern: each factor has ~sqrt(rank) nonzeros
                        nnz_u = max(1, int(np.sqrt(rank) * np.random.uniform(0.5, 1.5)))
                        nnz_v = max(1, int(np.sqrt(rank) * np.random.uniform(0.5, 1.5)))
                        nnz_w = max(1, int(np.sqrt(rank) * np.random.uniform(0.5, 1.5)))
                        
                        idx_u = np.random.choice(n * m, nnz_u, replace=False)
                        idx_v = np.random.choice(m * p, nnz_v, replace=False)
                        idx_w = np.random.choice(p * n, nnz_w, replace=False)
                        
                        U[idx_u, r] = sign_u[idx_u] * np.random.uniform(0.5, 1.5)
                        V[idx_v, r] = sign_v[idx_v] * np.random.uniform(0.5, 1.5)
                        W[idx_w, r] = sign_w[idx_w] * np.random.uniform(0.5, 1.5)
                        
                elif strategy == 'sparse_pattern':
                    # Very sparse initialization
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    
                    for r in range(rank):
                        # Each rank-1 term touches a small subset
                        nnz = max(2, rank // 8)
                        idx_u = np.random.choice(n * m, min(nnz, n * m), replace=False)
                        idx_v = np.random.choice(m * p, min(nnz, m * p), replace=False)
                        idx_w = np.random.choice(p * n, min(nnz, p * n), replace=False)
                        
                        U[idx_u, r] = np.random.choice([-1, 1], size=len(idx_u))
                        V[idx_v, r] = np.random.choice([-1, 1], size=len(idx_v))
                        W[idx_w, r] = np.random.choice([-1, 1], size=len(idx_w))
                        
                elif strategy == 'fourier_based':
                    # Use DFT-like structure
                    U = np.zeros((n * m, rank))
                    V = np.zeros((m * p, rank))
                    W = np.zeros((p * n, rank))
                    
                    for r in range(rank):
                        phase = 2 * np.pi * r / rank
                        U[:, r] = np.cos(phase * np.arange(n * m) + np.random.randn(n * m) * 0.1)
                        V[:, r] = np.cos(phase * np.arange(m * p) + np.random.randn(m * p) * 0.1)
                        W[:, r] = np.cos(phase * np.arange(p * n) + np.random.randn(p * n) * 0.1)
                    
                    # Scale
                    U *= 0.5
                    V *= 0.5
                    W *= 0.5
                
                result = try_als(U, V, W, max_iters=100)
                if result is not None:
                    U_f, V_f, W_f = result
                    
                    # Try to round to half-integers
                    U_r = np.round(U_f * 2) / 2
                    V_r = np.round(V_f * 2) / 2
                    W_r = np.round(W_f * 2) / 2
                    
                    if verify(U_r, V_r, W_r, atol=1e-8):
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    else:
                        # Return without rounding if it's close enough
                        if verify(U_f, V_f, W_f, atol=1e-6):
                            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
