import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 blocks of 2x2x2 (Strassen, rank 7) and 1 block of 2x2x1 (Trivial, rank 4)
    num_blocks = (p + 1) // 2
    total_rank = 0
    block_ranks = []
    for b in range(num_blocks):
        if b < num_blocks - 1:
            block_ranks.append(7)
        else:
            block_ranks.append(4)
    total_rank = sum(block_ranks)
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    r_offset = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal r_offset
        U[u_idx, r_offset] += u_val
        V[v_idx, r_offset] += v_val
        W[w_idx, r_offset] += w_val
        r_offset += 1
        
    for b in range(num_blocks):
        k_start = 2 * b
        if b < num_blocks - 1:
            # Strassen for 2x2x2 block
            # Local k=0 -> global k_start, local k=1 -> global k_start+1
            k0 = k_start
            k1 = k_start + 1
            
            # M1 = (A00+A11)(B00+B11)
            U[0*m+0, r_offset] = 1; U[1*m+1, r_offset] = 1
            V[0*p+k0, r_offset] = 1; V[1*p+k1, r_offset] = 1
            W[k0*n+0, r_offset] = 1; W[k1*n+1, r_offset] = 1
            r_offset += 1
            
            # M2 = (A10+A11)B00
            U[1*m+0, r_offset] = 1; U[1*m+1, r_offset] = 1
            V[0*p+k0, r_offset] = 1
            W[k0*n+1, r_offset] = 1; W[k1*n+1, r_offset] = -1
            r_offset += 1
            
            # M3 = A00(B01-B11)
            U[0*m+0, r_offset] = 1
            V[0*p+k1, r_offset] = 1; V[1*p+k1, r_offset] = -1
            W[k1*n+0, r_offset] = 1; W[k1*n+1, r_offset] = 1
            r_offset += 1
            
            # M4 = A11(B10-B00)
            U[1*m+1, r_offset] = 1
            V[1*p+k0, r_offset] = 1; V[0*p+k0, r_offset] = -1
            W[k0*n+0, r_offset] = 1; W[k0*n+1, r_offset] = 1
            r_offset += 1
            
            # M5 = (A00+A01)B11
            U[0*m+0, r_offset] = 1; U[0*m+1, r_offset] = 1
            V[1*p+k1, r_offset] = 1
            W[k0*n+0, r_offset] = -1; W[k1*n+0, r_offset] = 1
            r_offset += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[1*m+0, r_offset] = 1; U[0*m+0, r_offset] = -1
            V[0*p+k0, r_offset] = 1; V[0*p+k1, r_offset] = 1
            W[k1*n+1, r_offset] = -1
            r_offset += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[0*m+1, r_offset] = 1; U[1*m+1, r_offset] = -1
            V[1*p+k0, r_offset] = 1; V[1*p+k1, r_offset] = 1
            W[k0*n+0, r_offset] = 1
            r_offset += 1
            
        else:
            # Trivial for 2x2x1 block
            k0 = k_start
            
            # A00 B00
            U[0*m+0, r_offset] = 1
            V[0*p+k0, r_offset] = 1
            W[k0*n+0, r_offset] = 1
            r_offset += 1
            
            # A01 B10
            U[0*m+1, r_offset] = 1
            V[1*p+k0, r_offset] = 1
            W[k0*n+0, r_offset] = 1
            r_offset += 1
            
            # A10 B00
            U[1*m+0, r_offset] = 1
            V[0*p+k0, r_offset] = 1
            W[k0*n+1, r_offset] = 1
            r_offset += 1
            
            # A11 B10
            U[1*m+1, r_offset] = 1
            V[1*p+k0, r_offset] = 1
            W[k0*n+1, r_offset] = 1
            r_offset += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
