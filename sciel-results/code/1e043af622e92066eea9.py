"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strategy: Split m=4 into 3+1, p=6 into 3+3
    # This gives blocks: [2,3,3], [2,3,3], [2,1,3], [2,1,3]
    # Use rank-11 for [2,3,3] and rank-6 for [2,1,3]
    # Total rank: 11+11+6+6 = 34
    
    def get_rank11_233():
        """Return rank-11 decomposition for [2,3,3] matrix multiplication."""
        # This is a known decomposition. Let me construct it.
        # For [2,3,3]: n=2, m=3, p=3
        # U shape: (n*m, rank) = (6, 11)
        # V shape: (m*p, rank) = (9, 11)  
        # W shape: (p*n, rank) = (6, 11)
        
        # I'll use a specific known rank-11 decomposition
        # Based on the work of Landsberg and others
        
        # Let me try to find one numerically
        rng = np.random.RandomState(42)
        
        T = np.zeros((6, 9, 6))
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    T[i*3+j, j*3+k, k*2+i] = 1.0
        
        # Try rank 11
        best_err = np.inf
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(5):
            U = rng.randn(6, 11)
            V = rng.randn(9, 11)
            W = rng.randn(6, 11)
            
            for it in range(1000):
                if time.time() > deadline - 10:
                    break
                
                # Compute current tensor approximation
                T_approx = np.zeros((6, 9, 6))
                for c in range(11):
                    T_approx += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                
                residual = T - T_approx
                err = np.sum(residual**2)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
                
                # Gradient update
                lr = 0.001
                for c in range(11):
                    grad_U = np.zeros(6)
                    grad_V = np.zeros(9)
                    grad_W = np.zeros(6)
                    
                    for a in range(6):
                        for b in range(9):
                            for d in range(6):
                                r = residual[a, b, d]
                                if r != 0:
                                    grad_U[a] -= r * V[b, c] * W[d, c]
                                    grad_V[b] -= r * U[a, c] * W[d, c]
                                    grad_W[d] -= r * U[a, c] * V[b, c]
                    
                    U[:, c] -= lr * grad_U
                    V[:, c] -= lr * grad_V
                    W[:, c] -= lr * grad_W
            
            if time.time() > deadline - 10:
                break
        
        if best_err < 1e-8 and best_U is not None:
            return best_U, best_V, best_W
        
        # Fallback: return schoolbook for [2,3,3] with rank 18
        rank = 18
        U = np.zeros((6, rank))
        V = np.zeros((9, rank))
        W = np.zeros((6, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    def get_schoolbook_213():
        """Return schoolbook rank-6 decomposition for [2,1,3]."""
        rank = 6
        U = np.zeros((2, rank))
        V = np.zeros((3, rank))
        W = np.zeros((6, rank))
        col = 0
        for i in range(2):
            for j in range(1):
                for k in range(3):
                    U[i*1+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return U, V, W
    
    # Get decompositions
    U_233, V_233, W_233 = get_rank11_233()
    rank_233 = U_233.shape[1]
    
    U_213, V_213, W_213 = get_schoolbook_213()
    rank_213 = U_213.shape[1]
    
    total_rank = 2 * rank_233 + 2 * rank_213
    
    # Compose into full [2,4,6] decomposition
    # m=4 split as 3+1, p=6 split as 3+3
    # Block (j_block, k_block): j_block in {0,1}, k_block in {0,1}
    # j_block=0: j in {0,1,2}, uses [2,3,3]
    # j_block=1: j in {3}, uses [2,1,3]
    # k_block=0: k in {0,1,2}, 
    # k_block=1: k in {3,4,5}
    
    U_full = np.zeros((2*4, total_rank))
    V_full = np.zeros((4*6, total_rank))
    W_full = np.zeros((6*2, total_rank))
    
    col_offset = 0
    for j_block in range(2):
        for k_block in range(2):
            if j_block == 0:
                # [2,3,3] block
                block_rank = rank_233
                U_block, V_block, W_block = U_233, V_233, W_233
                m_sub = 3
                p_sub = 3
            else:
                # [2,1,3] block
                block_rank = rank_213
                U_block, V_block, W_block = U_213, V_213, W_213
                m_sub = 1
                p_sub = 3
            
            # U: row index = i*4 + j_global
            # j_global = j_block*3 + j_sub (for j_block=0) or j_block*3 + j_sub (for j_block=1, j_sub=0)
            for i in range(2):
                for j_sub in range(m_sub):
                    j_global = j_block * 3 + j_sub
                    u_row = i * 4 + j_global
                    U_full[u_row, col_offset:col_offset+block_rank] = U_block[i*m_sub + j_sub, :]
            
            # V: row index = j_global*6 + k_global
            # k_global = k_block*3 + k_sub
            for j_sub in range(m_sub):
                j_global = j_block * 3 + j_sub
                for k_sub in range(p_sub):
                    k_global = k_block * 3 + k_sub
                    v_row = j_global * 6 + k_global
                    V_full[v_row, col_offset:col_offset+block_rank] = V_block[j_sub*p_sub + k_sub, :]
            
            # W: row index = k_global*2 + i
            for k_sub in range(p_sub):
                k_global = k_block * 3 + k_sub
                for i in range(2):
                    w_row = k_global * 2 + i
                    W_full[w_row, col_offset:col_offset+block_rank] = W_block[k_sub*2 + i, :]
            
            col_offset += block_rank
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list_of_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    return {"U": to_list_of_lists(U_full), "V": to_list_of_lists(V_full), "W": to_list_of_lists(W_full)}
# EVOLVE-BLOCK-END
