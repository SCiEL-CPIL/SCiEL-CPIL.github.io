"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s margin
    
    n, m, p = DIMENSIONS
    
    # Split indices into 2, 2, 1
    # I1 = {0, 1}, I2 = {2, 3}, I3 = {4}
    # Sizes: s1=2, s2=2, s3=1
    
    # We will build the decomposition by summing contributions from each block combination.
    # Each block combination (a, b, c) corresponds to multiplying A[Ia, Ib] by B[Ib, Ic] to get C[Ia, Ic].
    
    # Helper to create a Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        # Returns U, V, W for 2x2x2 multiplication
        # U shape (4, 7), V shape (4, 7), W shape (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        # M1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        W[1][0] = -1; W[2][0] = -1
        
        # M2
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # M5
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[1][4] = 1
        
        # M6
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[2][5] = 1; W[0][5] = 1
        
        # M7
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[3][6] = 1; W[1][6] = 1
        
        return U, V, W

    # Helper to create a schoolbook decomposition for rxk x kxc -> rxc
    def schoolbook(r, k, c):
        U = np.zeros((r*k, r*k*c))
        V = np.zeros((k*c, r*k*c))
        W = np.zeros((c*r, r*k*c))
        col = 0
        for i in range(r):
            for j in range(k):
                for l in range(c):
                    U[i*k+j][col] = 1
                    V[j*c+l][col] = 1
                    W[l*r+i][col] = 1
                    col += 1
        return U, V, W

    # We need to assemble the full 5x5x5 decomposition.
    # Total rank is sum of ranks of sub-blocks.
    # We will store the contributions in large arrays.
    
    # Max rank estimate: 117
    max_rank = 120
    U_full = np.zeros((25, max_rank))
    V_full = np.zeros((25, max_rank))
    W_full = np.zeros((25, max_rank))
    current_rank = 0
    
    # Block indices
    # 0: size 2 (indices 0,1)
    # 1: size 2 (indices 2,3)
    # 2: size 1 (index 4)
    
    blocks = [
        (0, 2, 0, 2), # I0: 0-1
        (2, 4, 2, 2), # I1: 2-3
        (4, 5, 1, 1)  # I2: 4
    ]
    # Actually, let's just use offsets and sizes.
    # Block 0: rows 0,1; cols 0,1
    # Block 1: rows 2,3; cols 2,3
    # Block 2: row 4; col 4
    
    def add_block(a_idx, b_idx, c_idx, U_sub, V_sub, W_sub, offset):
        # a_idx: (start, size) for A rows
        # b_idx: (start, size) for A cols / B rows
        # c_idx: (start, size) for B cols / C cols
        # U_sub: (r*k, rank)
        # V_sub: (k*c, rank)
        # W_sub: (c*r, rank)
        
        r, k = a_idx[1], b_idx[1]
        kc = b_idx[1] * c_idx[1]
        cr = c_idx[1] * r
        
        # Map A indices: i in [0, r), j in [0, k) -> i*k + j
        # Global A index: (a_start + i)*5 + (b_start + j) ? No, A is 5x5.
        # A is flattened row-major?
        # The task says U shape (n*m, rank). n=5, m=5.
        # So U index is i*5 + j.
        
        a_start, a_size = a_idx
        b_start, b_size = b_idx
        c_start, c_size = c_idx
        
        for local_i in range(a_size):
            for local_j in range(b_size):
                global_a = (a_start + local_i) * 5 + (b_start + local_j)
                local_u = local_i * b_size + local_j
                U_full[global_a, offset:offset+U_sub.shape[1]] += U_sub[local_u]
                
        for local_j in range(b_size):
            for local_k in range(c_size):
                global_b = (b_start + local_j) * 5 + (c_start + local_k)
                local_v = local_j * c_size + local_k
                V_full[global_b, offset:offset+V_sub.shape[1]] += V_sub[local_v]
                
        for local_k in range(c_size):
            for local_i in range(a_size):
                global_c = (c_start + local_k) * 5 + (a_start + local_i)
                local_w = local_k * a_size + local_i
                W_full[global_c, offset:offset+W_sub.shape[1]] += W_sub[local_w]
                
        return U_sub.shape[1]

    # Precompute sub-decompositions
    U_222, V_222, W_222 = strassen_2x2x2()
    U_221, V_221, W_221 = schoolbook(2, 2, 1)
    U_212, V_212, W_212 = schoolbook(2, 1, 2)
    U_122, V_122, W_122 = schoolbook(1, 2, 2)
    U_211, V_211, W_211 = schoolbook(2, 1, 1)
    U_121, V_121, W_121 = schoolbook(1, 2, 1)
    U_112, V_112, W_112 = schoolbook(1, 1, 2)
    U_111, V_111, W_111 = schoolbook(1, 1, 1)
    
    # Block definitions: (start, size)
    B0 = (0, 2)
    B1 = (2, 2)
    B2 = (4, 1)
    
    # Iterate over all combinations
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if time.time() > deadline:
                    break
                
                a_block = [B0, B1, B2][a]
                b_block = [B0, B1, B2][b]
                c_block = [B0, B1, B2][c]
                
                r = a_block[1]
                k = b_block[1]
                cc = c_block[1]
                
                if r == 2 and k == 2 and cc == 2:
                    U_sub, V_sub, W_sub = U_222, V_222, W_222
                elif r == 2 and k == 2 and cc == 1:
                    U_sub, V_sub, W_sub = U_221, V_221, W_221
                elif r == 2 and k == 1 and cc == 2:
                    U_sub, V_sub, W_sub = U_212, V_212, W_212
                elif r == 1 and k == 2 and cc == 2:
                    U_sub, V_sub, W_sub = U_122, V_122, W_122
                elif r == 2 and k == 1 and cc == 1:
                    U_sub, V_sub, W_sub = U_211, V_211, W_211
                elif r == 1 and k == 2 and cc == 1:
                    U_sub, V_sub, W_sub = U_121, V_121, W_121
                elif r == 1 and k == 1 and cc == 2:
                    U_sub, V_sub, W_sub = U_112, V_112, W_112
                elif r == 1 and k == 1 and cc == 1:
                    U_sub, V_sub, W_sub = U_111, V_111, W_111
                else:
                    # Fallback to schoolbook
                    U_sub, V_sub, W_sub = schoolbook(r, k, cc)
                    
                current_rank += add_block(a_block, b_block, c_block, U_sub, V_sub, W_sub, current_rank)
                
    # Trim arrays to actual rank
    U_final = U_full[:, :current_rank]
    V_final = V_full[:, :current_rank]
    W_final = W_full[:, :current_rank]
    
    # Convert to lists of lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
