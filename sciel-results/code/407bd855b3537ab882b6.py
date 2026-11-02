import numpy as np
import time
import random
from scipy.optimize import least_squares

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # Vectorized check for all entries
    # T_{a,b,c,d,i,k} = delta_{a,i} delta_{c,k} delta_{b,d}
    
    # We check all 2*3*3*6*2*6 = 1296 entries
    # To be fast, we can vectorize the dot products.
    # U[u_row, :], V[v_row, :], W[w_row, :]
    
    # Create index arrays
    a_idx, b_idx, c_idx, d_idx, i_idx, k_idx = np.meshgrid(
        range(n), range(m), range(m), range(p), range(n), range(p),
        indexing='ij'
    )
    
    u_rows = a_idx * m + b_idx
    v_rows = c_idx * p + d_idx
    w_rows = k_idx * n + i_idx
    
    targets = np.where((a_idx == i_idx) & (c_idx == k_idx) & (b_idx == d_idx), 1.0, 0.0)
    
    # Reshape to 1D for vectorized dot
    u_flat = U[u_rows.flatten(), :]
    v_flat = V[v_rows.flatten(), :]
    w_flat = W[w_rows.flatten(), :]
    targets_flat = targets.flatten()
    
    # Compute sum_l U[u,l] * V[v,l] * W[w,l]
    # This is element-wise product of v and w, then dot with u
    # (N, r) * (N, r) -> (N, r) -> sum over r -> (N,)
    computed = np.sum(u_flat * (v_flat * w_flat), axis=1)
    
    if not np.allclose(computed, targets_flat, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Strategy:
    # M_{2,3,6} is a direct sum of two M_{2,3,3} tensors.
    # Rank(M_{2,3,6}) = 2 * Rank(M_{2,3,3}).
    # We need to find the minimal rank for M_{2,3,3}.
    # Schoolbook rank is 18.
    # Lower bound is likely 16 or 17.
    # Let's try to find a rank 16 solution for M_{2,3,3}.
    
    def try_find_m233_decomposition(rank, max_restarts=10, timeout_per_restart=5.0):
        """
        Attempts to find a rank 'rank' decomposition for M_{2,3,3}.
        Returns (U, V, W) if found, else None.
        """
        n_sub, m_sub, p_sub = 2, 3, 3
        
        # Sizes
        u_size = n_sub * m_sub * rank
        v_size = m_sub * p_sub * rank
        w_size = p_sub * n_sub * rank
        
        def residuals(params):
            U_flat = params[:u_size]
            V_flat = params[u_size:u_size+v_size]
            W_flat = params[u_size+v_size:]
            
            U = U_flat.reshape(n_sub * m_sub, rank)
            V = V_flat.reshape(m_sub * p_sub, rank)
            W = W_flat.reshape(p_sub * n_sub, rank)
            
            # Vectorized residual calculation
            a_idx, b_idx, c_idx, d_idx, i_idx, k_idx = np.meshgrid(
                range(n_sub), range(m_sub), range(m_sub), range(p_sub), range(n_sub), range(p_sub),
                indexing='ij'
            )
            u_rows = a_idx * m_sub + b_idx
            v_rows = c_idx * p_sub + d_idx
            w_rows = k_idx * n_sub + i_idx
            targets = np.where((a_idx == i_idx) & (c_idx == k_idx) & (b_idx == d_idx), 1.0, 0.0)
            
            u_flat = U[u_rows.flatten(), :]
            v_flat = V[v_rows.flatten(), :]
            w_flat = W[w_rows.flatten(), :]
            targets_flat = targets.flatten()
            
            computed = np.sum(u_flat * (v_flat * w_flat), axis=1)
            return computed - targets_flat
        
        best_solution = None
        best_cost = np.inf
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize with structured noise
            # Use a mix of random and small perturbations of schoolbook-like structures
            # Schoolbook for M_{2,3,3} has rank 18.
            # If rank < 18, we need to merge terms.
            
            # Random init
            U_init = np.random.randn(n_sub * m_sub, rank) * 0.5
            V_init = np.random.randn(m_sub * p_sub, rank) * 0.5
            W_init = np.random.randn(p_sub * n_sub, rank) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, params_init, method='lm', max_nfev=500)
                if result.cost < best_cost:
                    best_cost = result.cost
                    if result.cost < 1e-12:
                        U_sol = result.x[:u_size].reshape(n_sub * m_sub, rank)
                        V_sol = result.x[u_size:u_size+v_size].reshape(m_sub * p_sub, rank)
                        W_sol = result.x[u_size+v_size:].reshape(p_sub * n_sub, rank)
                        
                        # Snap to half-integers
                        U_snap = np.round(U_sol * 2) / 2
                        V_snap = np.round(V_sol * 2) / 2
                        W_snap = np.round(W_sol * 2) / 2
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, n_sub, m_sub, p_sub):
                            return U_snap, V_snap, W_snap
            except Exception:
                pass
        
        return None

    # Attempt rank 16 for M_{2,3,3}
    m233_rank16 = try_find_m233_decomposition(16, max_restarts=20, timeout_per_restart=5.0)
    
    if m233_rank16 is not None:
        U233, V233, W233 = m233_rank16
        # Construct M_{2,3,6} from two M_{2,3,3} blocks
        # Total rank 32
        U_comb = np.zeros((6, 32))
        V_comb = np.zeros((18, 32))
        W_comb = np.zeros((12, 32))
        
        # Block 1: d in {0,1,2}, k in {0,1,2}
        U_comb[:, :16] = U233
        for c in range(3):
            for d in range(3):
                V_comb[c * 6 + d, :16] = V233[c * 3 + d]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :16] = W233[k * 2 + i]
                
        # Block 2: d in {3,4,5}, k in {3,4,5}
        U_comb[:, 16:] = U233
        for c in range(3):
            for d in range(3, 6):
                d_local = d - 3
                V_comb[c * 6 + d, 16:] = V233[c * 3 + d_local]
        for k in range(3, 6):
            k_local = k - 3
            for i in range(2):
                W_comb[k * 2 + i, 16:] = W233[k_local * 2 + i]
                
        if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }

    # Attempt rank 17 for M_{2,3,3} if 16 failed
    m233_rank17 = try_find_m233_decomposition(17, max_restarts=20, timeout_per_restart=5.0)
    
    if m233_rank17 is not None:
        U233, V233, W233 = m233_rank17
        # Construct M_{2,3,6} from two M_{2,3,3} blocks
        # Total rank 34
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        # Block 1
        U_comb[:, :17] = U233
        for c in range(3):
            for d in range(3):
                V_comb[c * 6 + d, :17] = V233[c * 3 + d]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233[k * 2 + i]
                
        # Block 2
        U_comb[:, 17:] = U233
        for c in range(3):
            for d in range(3, 6):
                d_local = d - 3
                V_comb[c * 6 + d, 17:] = V233[c * 3 + d_local]
        for k in range(3, 6):
            k_local = k - 3
            for i in range(2):
                W_comb[k * 2 + i, 17:] = W233[k_local * 2 + i]
                
        if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }

    # Fallback: Hardcoded Rank 34 solution from seed
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
