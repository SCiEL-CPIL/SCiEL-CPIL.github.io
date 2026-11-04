"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find rank-minimal decomposition
    # Lower bound for 2x5x6 is likely around 30.
    # Schoolbook is 60.
    for rank in range(30, 61):
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)

def _try_rank(n, m, p, rank, deadline):
    max_restarts = 50
    max_iter = 50
    
    # Pre-allocate indices for vectorization
    # Indices for U: (i, j) -> flat index in U (n*m)
    # Indices for V: (j, k) -> flat index in V (m*p)
    # Indices for W: (k, i) -> flat index in W (p*n)
    
    # Build index arrays for the big linear systems
    # For U: We solve for U[i, j, r] for all i, j.
    # The system is: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all k.
    # This is a system of p equations for each (i, j).
    # Total equations: n * m * p.
    # Unknowns: n * m * rank.
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    # For V: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all i.
    # System of n equations for each (j, k).
    # Total equations: m * p * n.
    # Unknowns: m * p * rank.
    
    i_idx_v = np.tile(np.arange(n), m * p)
    j_idx_v = np.repeat(np.arange(m), p * n)
    k_idx_v = np.tile(np.repeat(np.arange(p), n), m)
    
    # For W: sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] = 1 for all j.
    # System of m equations for each (k, i).
    # Total equations: p * n * m.
    # Unknowns: p * n * rank.
    
    i_idx_w = np.tile(np.arange(n), m * p)
    j_idx_w = np.tile(np.repeat(np.arange(m), p), n)
    k_idx_w = np.repeat(np.arange(p), n * m)

    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = np.random.RandomState(12345 + restart)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for U
            # A_U[row, r] where row corresponds to (i, j, k)
            # A_U[row, r] = V[j, k, r] * W[k, i, r]
            # We need to construct this matrix efficiently.
            # V[j_idx, k_idx, r] * W[k_idx, i_idx, r]
            # Note: V is (m*p, rank), so V[j*p+k, r]
            # W is (p*n, rank), so W[k*n+i, r]
            
            V_flat = V[j_idx * p + k_idx, :]
            W_flat = W[k_idx * n + i_idx, :]
            A_U = V_flat * W_flat
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(n * m * p), rcond=None)
                U = U_flat.reshape(n * m, rank)
            except:
                pass
            
            # Solve for V
            # A_V[row, r] = U[i, j, r] * W[k, i, r]
            # row corresponds to (j, k, i)
            U_flat_v = U[i_idx_v * m + j_idx_v, :]
            W_flat_v = W[k_idx_v * n + i_idx_v, :]
            A_V = U_flat_v * W_flat_v
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(m * p * n), rcond=None)
                V = V_flat.reshape(m * p, rank)
            except:
                pass
            
            # Solve for W
            # A_W[row, r] = U[i, j, r] * V[j, k, r]
            # row corresponds to (k, i, j)
            U_flat_w = U[i_idx_w * m + j_idx_w, :]
            V_flat_w = V[j_idx_w * p + k_idx_w, :]
            A_W = U_flat_w * V_flat_w
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(p * n * m), rcond=None)
                W = W_flat.reshape(p * n, rank)
            except:
                pass
            
            # Check error every few iterations
            if it % 5 == 4:
                # Calculate error efficiently
                # Err[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r] - 1
                # We can compute the tensor contraction.
                # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
                
                # Reshape U to (n, m, rank)
                U_3d = U.reshape(n, m, rank)
                V_3d = V.reshape(m, p, rank)
                W_3d = W.reshape(p, n, rank)
                
                # Contract over r
                # T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                
                # Use einsum for speed
                try:
                    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                    max_err = np.max(np.abs(T - 1.0))
                except:
                    max_err = 1.0
                    
                if max_err < 1e-7:
                    result = _rationalize(U, V, W, n, m, p)
                    if result:
                        return result
                        
    return None

def _rationalize(U, V, W, n, m, p):
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    U_3d = U_r.reshape(n, m, -1)
    V_3d = V_r.reshape(m, p, -1)
    W_3d = W_r.reshape(p, n, -1)
    
    try:
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if np.max(np.abs(T - 1.0)) > 1e-9:
            return None
    except:
        return None
        
    rank = U.shape[1]
    U_list = [[float(U_r[i,j]) for j in range(rank)] for i in range(U_r.shape[0])]
    V_list = [[float(V_r[i,j]) for j in range(rank)] for i in range(V_r.shape[0])]
    W_list = [[float(W_r[i,j]) for j in range(rank)] for i in range(W_r.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
