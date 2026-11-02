import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    # Strategy:
    # Decompose 2x5x6 into blocks based on partitioning indices.
    # I (rows of A): 0..1 (size 2)
    # J (cols of A / rows of B): 0..4 (size 5)
    # K (cols of B / cols of C): 0..5 (size 6)
    
    # We partition J into J1={0,1}, J2={2,3}, J3={4}
    # We partition K into K1={0,1}, K2={2,3}, K3={4,5}
    
    # Blocks:
    # 1. I x J1 x K1: 2x2x2 -> Strassen (Rank 7)
    # 2. I x J1 x K2: 2x2x2 -> Strassen (Rank 7)
    # 3. I x J1 x K3: 2x2x2 -> Strassen (Rank 7)
    # 4. I x J2 x K1: 2x2x2 -> Strassen (Rank 7)
    # 5. I x J2 x K2: 2x2x2 -> Strassen (Rank 7)
    # 6. I x J2 x K3: 2x2x2 -> Strassen (Rank 7)
    # 7. I x J3 x K1: 2x1x2 -> Standard (Rank 4)
    # 8. I x J3 x K2: 2x1x2 -> Standard (Rank 4)
    # 9. I x J3 x K3: 2x1x2 -> Standard (Rank 4)
    
    # Total Rank = 6 * 7 + 3 * 4 = 42 + 12 = 54

    total_rank = 54
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    current_rank = 0

    def add_strassen_block(j_offset, k_offset):
        """
        Adds a 2x2x2 Strassen block.
        j_offset: start index of the 2-element slice in J
        k_offset: start index of the 2-element slice in K
        I is always 0..1
        """
        nonlocal current_rank
        
        # Indices in U (n*m): i*m + j
        # i=0, j=j_offset -> 0*m + j_offset
        # i=0, j=j_offset+1 -> 0*m + j_offset + 1
        # i=1, j=j_offset -> 1*m + j_offset
        # i=1, j=j_offset+1 -> 1*m + j_offset + 1
        
        u_00 = 0 * m + j_offset
        u_01 = 0 * m + (j_offset + 1)
        u_10 = 1 * m + j_offset
        u_11 = 1 * m + (j_offset + 1)
        
        # Indices in V (m*p): j*p + k
        # j=j_offset, k=k_offset -> j_offset*p + k_offset
        # j=j_offset, k=k_offset+1 -> j_offset*p + k_offset + 1
        # j=j_offset+1, k=k_offset -> (j_offset+1)*p + k_offset
        # j=j_offset+1, k=k_offset+1 -> (j_offset+1)*p + k_offset + 1
        
        v_00 = j_offset * p + k_offset
        v_01 = j_offset * p + (k_offset + 1)
        v_10 = (j_offset + 1) * p + k_offset
        v_11 = (j_offset + 1) * p + (k_offset + 1)
        
        # Indices in W (p*n): k*n + i
        # k=k_offset, i=0 -> k_offset*n + 0
        # k=k_offset, i=1 -> k_offset*n + 1
        # k=k_offset+1, i=0 -> (k_offset+1)*n + 0
        # k=k_offset+1, i=1 -> (k_offset+1)*n + 1
        
        w_00 = k_offset * n + 0
        w_01 = k_offset * n + 1
        w_10 = (k_offset + 1) * n + 0
        w_11 = (k_offset + 1) * n + 1

        # Strassen 7 steps
        # M1 = (A00 + A11) * (B00 + B11)
        U[u_00, current_rank] += 1
        U[u_11, current_rank] += 1
        V[v_00, current_rank] += 1
        V[v_11, current_rank] += 1
        W[w_00, current_rank] += 1
        W[w_11, current_rank] += 1
        current_rank += 1

        # M2 = (A10 + A11) * B00
        U[u_10, current_rank] += 1
        U[u_11, current_rank] += 1
        V[v_00, current_rank] += 1
        W[w_01, current_rank] += 1
        W[w_11, current_rank] -= 1
        current_rank += 1

        # M3 = A00 * (B01 - B11)
        U[u_00, current_rank] += 1
        V[v_01, current_rank] += 1
        V[v_11, current_rank] -= 1
        W[w_10, current_rank] += 1
        W[w_11, current_rank] += 1
        current_rank += 1

        # M4 = A11 * (B10 - B00)
        U[u_11, current_rank] += 1
        V[v_10, current_rank] += 1
        V[v_00, current_rank] -= 1
        W[w_00, current_rank] += 1
        W[w_01, current_rank] += 1
        current_rank += 1

        # M5 = (A00 + A01) * B11
        U[u_00, current_rank] += 1
        U[u_01, current_rank] += 1
        V[v_11, current_rank] += 1
        W[w_00, current_rank] -= 1
        W[w_10, current_rank] += 1
        current_rank += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[u_10, current_rank] += 1
        U[u_00, current_rank] -= 1
        V[v_00, current_rank] += 1
        V[v_01, current_rank] += 1
        W[w_11, current_rank] += 1
        current_rank += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[u_01, current_rank] += 1
        U[u_11, current_rank] -= 1
        V[v_10, current_rank] += 1
        V[v_11, current_rank] += 1
        W[w_00, current_rank] += 1
        current_rank += 1

    def add_standard_block(j_idx, k_start, k_end):
        """
        Adds a 2x1x2 standard block.
        j_idx: specific index in J (size 1)
        k_start, k_end: range of indices in K (size 2)
        I is always 0..1
        """
        nonlocal current_rank
        
        for k in range(k_start, k_end):
            for i in range(n):
                # U index: i*m + j_idx
                u_idx = i * m + j_idx
                # V index: j_idx*p + k
                v_idx = j_idx * p + k
                # W index: k*n + i
                w_idx = k * n + i
                
                U[u_idx, current_rank] = 1
                V[v_idx, current_rank] = 1
                W[w_idx, current_rank] = 1
                current_rank += 1

    # Apply blocks
    # J1 = {0, 1}, J2 = {2, 3}, J3 = {4}
    # K1 = {0, 1}, K2 = {2, 3}, K3 = {4, 5}
    
    j_parts = [0, 2, 4] # Start indices for J1, J2, J3
    k_parts = [0, 2, 4] # Start indices for K1, K2, K3
    
    for j_start in j_parts:
        for k_start in k_parts:
            # Check if we have a 2x2 block or 2x1 block
            # J1, J2 have size 2. J3 has size 1.
            # K1, K2, K3 have size 2.
            
            if j_start in [0, 2]: # Size 2 J-part
                add_strassen_block(j_start, k_start)
            else: # Size 1 J-part (j_start=4)
                add_standard_block(j_start, k_start, k_start + 2)

    # Verify the decomposition
    # Tensor T_{i,j,k} should be 1.
    # T = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    
    U_final = U[:, :current_rank]
    V_final = V[:, :current_rank]
    W_final = W[:, :current_rank]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_final[i * m + j, :]
                v_vec = V_final[j * p + k, :]
                w_vec = W_final[k * n + i, :]
                
                # Product sum
                res = np.dot(np.dot(u_vec, v_vec), w_vec)
                if abs(res - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: Trivial rank 60
        U_f = np.zeros((n * m, n * m * p), dtype=float)
        V_f = np.zeros((m * p, n * m * p), dtype=float)
        W_f = np.zeros((p * n, n * m * p), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, idx] = 1
                    V_f[j * p + k, idx] = 1
                    W_f[k * n + i, idx] = 1
                    idx += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
