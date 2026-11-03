import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We will build U, V, W lists of vectors
    # U shape: (n*m, rank) -> (12, rank)
    # V shape: (m*p, rank) -> (32, rank)
    # W shape: (p*n, rank) -> (24, rank)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Part 1: i=2 (1x4x8), rank 32
    # i=2, j in 0..3, k in 0..7
    for j in range(4):
        for k in range(8):
            u = [0.0] * (n*m)
            v = [0.0] * (m*p)
            w = [0.0] * (p*n)
            
            u[2*m + j] = 1.0  # i=2, col j
            v[j*p + k] = 1.0  # row j, col k
            w[k*n + 2] = 1.0  # row k, col i=2
            
            add_term(u, v, w)
            
    # Part 2: i in {0,1} (2x4x8), rank 56
    # Split j into pairs: j0={0,1}, j1={2,3}
    # Split k into pairs: k0={0,1}, k1={2,3}, k2={4,5}, k3={6,7}
    
    # Strassen decomposition for 2x2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = AB
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Indices in our global system:
    # i_local 0 -> i_global 0
    # i_local 1 -> i_global 1
    # j_local 0 -> j_global j_base
    # j_local 1 -> j_global j_base + 1
    # k_local 0 -> k_global k_base
    # k_local 1 -> k_global k_base + 1
    
    def strassen_term(i_base, j_base, k_base, coeff_u, coeff_v, coeff_w, u_idx, v_idx, w_idx):
        # This is not quite right. Strassen combines terms.
        # We need to add 7 terms for each 2x2x2 block.
        pass
        
    # Let's manually expand the 7 Strassen terms for a specific block (i_base, j_base, k_base)
    # i_local = 0, 1
    # j_local = 0, 1
    # k_local = 0, 1
    
    # Term 1: M1 = (A_00 + A_11)(B_00 + B_11)
    # This expands to 4 scalar products.
    # But Strassen uses 7 multiplications.
    # We define the 7 multiplications and their contributions to C.
    
    # For block at (i_base, j_base, k_base):
    # A is 2x2: rows i_base, i_base+1; cols j_base, j_base+1
    # B is 2x2: rows j_base, j_base+1; cols k_base, k_base+1
    # C is 2x2: rows k_base, k_base+1; cols i_base, i_base+1
    
    j_pairs = [(0, 2), (2, 4)] # (start, end) for j
    k_pairs = [(0, 2), (2, 4), (4, 6), (6, 8)]
    
    for i_base in [0]: # Only one pair for i: 0,1
        for j_start in [0, 2]:
            j_end = j_start + 2
            for k_start in [0, 2, 4, 6]:
                k_end = k_start + 2
                
                # Define the 7 Strassen multiplications
                # Each multiplication is of form (sum of A terms) * (sum of B terms)
                # We need to distribute these into U, V, W vectors.
                
                # U vector: non-zero at (i, j)
                # V vector: non-zero at (j, k)
                # W vector: non-zero at (k, i)
                
                # M1 = (A_00 + A_11)(B_00 + B_11)
                # U_1: A_00 + A_11 -> u[i_base, j_start] = 1, u[i_base+1, j_start+1] = 1
                # V_1: B_00 + B_11 -> v[j_start, k_start] = 1, v[j_start+1, k_start+1] = 1
                # W_1: C_00 += 1, C_01 += 1, C_10 += 1, C_11 += 1
                # Wait, W is (k, i).
                # C_00 (k=0, i=0) -> w[k_start, i_base] += 1
                # C_01 (k=0, i=1) -> w[k_start, i_base+1] += 1
                # C_10 (k=1, i=0) -> w[k_start+1, i_base] += 1
                # C_11 (k=1, i=1) -> w[k_start+1, i_base+1] += 1
                
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                
                # M1
                u[i_base*m + j_start] = 1.0
                u[(i_base+1)*m + (j_start+1)] = 1.0
                v[j_start*p + k_start] = 1.0
                v[(j_start+1)*p + (k_start+1)] = 1.0
                w[k_start*n + i_base] = 1.0
                w[k_start*n + (i_base+1)] = 1.0
                w[(k_start+1)*n + i_base] = 1.0
                w[(k_start+1)*n + (i_base+1)] = 1.0
                add_term(u, v, w)
                
                # M2 = (A_10 + A_11)B_00
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[(i_base+1)*m + j_start] = 1.0
                u[(i_base+1)*m + (j_start+1)] = 1.0
                v[j_start*p + k_start] = 1.0
                # C_10 += 1, C_11 += 1
                w[(k_start+1)*n + i_base] = 1.0
                w[(k_start+1)*n + (i_base+1)] = 1.0
                add_term(u, v, w)
                
                # M3 = A_00(B_01 - B_11)
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[i_base*m + j_start] = 1.0
                v[j_start*p + (k_start+1)] = 1.0
                v[(j_start+1)*p + (k_start+1)] = -1.0
                # C_01 += 1, C_11 += 1
                w[k_start*n + (i_base+1)] = 1.0
                w[(k_start+1)*n + (i_base+1)] = 1.0
                add_term(u, v, w)
                
                # M4 = A_11(B_10 - B_00)
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[(i_base+1)*m + (j_start+1)] = 1.0
                v[(j_start+1)*p + k_start] = 1.0
                v[j_start*p + k_start] = -1.0
                # C_00 += 1, C_10 += 1
                w[k_start*n + i_base] = 1.0
                w[(k_start+1)*n + i_base] = 1.0
                add_term(u, v, w)
                
                # M5 = (A_00 + A_01)B_11
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[i_base*m + j_start] = 1.0
                u[i_base*m + (j_start+1)] = 1.0
                v[(j_start+1)*p + (k_start+1)] = 1.0
                # C_00 -= 1, C_10 -= 1
                w[k_start*n + i_base] = -1.0
                w[(k_start+1)*n + i_base] = -1.0
                add_term(u, v, w)
                
                # M6 = (A_10 - A_00)(B_00 + B_01)
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[(i_base+1)*m + j_start] = 1.0
                u[i_base*m + j_start] = -1.0
                v[j_start*p + k_start] = 1.0
                v[j_start*p + (k_start+1)] = 1.0
                # C_11 += 1
                w[(k_start+1)*n + (i_base+1)] = 1.0
                add_term(u, v, w)
                
                # M7 = (A_01 - A_11)(B_10 + B_11)
                u = [0.0] * (n*m)
                v = [0.0] * (m*p)
                w = [0.0] * (p*n)
                u[i_base*m + (j_start+1)] = 1.0
                u[(i_base+1)*m + (j_start+1)] = -1.0
                v[(j_start+1)*p + k_start] = 1.0
                v[(j_start+1)*p + (k_start+1)] = 1.0
                # C_00 += 1
                w[k_start*n + i_base] = 1.0
                add_term(u, v, w)
                
    rank = len(U_cols)
    
    U = [col for col in U_cols]
    V = [col for col in V_cols]
    W = [col for col in W_cols]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
