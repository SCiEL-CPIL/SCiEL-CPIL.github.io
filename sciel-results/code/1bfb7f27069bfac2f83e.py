import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W
    
    def get_hybrid():
        # Schoolbook for 3x3x3 (rank 27) + Simple for 3x3x2 (rank 2)
        # This gives rank 29
        rank = 29
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part 1: Schoolbook for k=0,1,2 (3x3x3)
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        
        # Part 2: Simple for k=3,4 (3x3x2)
        # For k=3: U[i,j] = 1, V[j,3] = 1, W[3,i] = 1
        # For k=4: U[i,j] = 1, V[j,4] = 1, W[4,i] = 1
        # Wait, this doesn't work because we need to separate the columns for k=3 and k=4
        # Let me reconsider.
        
        # Actually, the "simple" approach for 3x3x2 with rank 2 is:
        # r=27: U[i,j]=1, V[j,3]=1, W[3,i]=1  (this handles the (i,j,3) entry)
        # r=28: U[i,j]=1, V[j,4]=1, W[4,i]=1  (this handles the (i,j,4) entry)
        
        # So:
        for k_extra in [3, 4]:
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    v_idx = j * p + k_extra
                    w_idx = k_extra * n + i
                    
                    # This would require 9 more terms, not 2
                    # The "rank 2" claim for 3x3x2 is incorrect in this context
                    # because we need to cover all (i,j) pairs for a fixed k.
                    
                    # Let me stick with pure schoolbook for safety.
                    pass
        
        # It seems the hybrid approach is more complex than I thought.
        # Let's just use schoolbook for now, but I'll try to find a better decomposition.
        
        return U, V, W
    
    def get_strassen_3x3():
        # Strassen-like decomposition for 3x3x3 with rank 23
        # This is complex; let's use a known decomposition if possible
        # For now, return None to indicate not available
        return None
    
    # Try to find a better decomposition
    U_sb, V_sb, W_sb = get_schoolbook()
    
    # Verify schoolbook decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_sb[i*m+j, :]
                v_val = V_sb[j*p+k, :]
                w_val = W_sb[k*n+i, :]
                s = np.sum(u_vec * v_val * w_val)
                if not np.isclose(s, 1.0):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # For now, return schoolbook as the best we can reliably produce
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
