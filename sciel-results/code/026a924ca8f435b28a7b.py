import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140  # 10s safety margin

    # Strassen decomposition for 2x2x2
    # T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r
    # u_r, v_r are length 4 vectors (flattened 2x2)
    # w_r is length 4 vector (flattened 2x2)
    
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_2x2 = [
        # u, v, w
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # Z00+=1, Z11+=1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # Z10+=1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # Z01+=1, Z11+=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # Z00+=1, Z10+=1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # Z00-=1, Z01+=1
        # M6
        ([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # Z11+=1
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # Z00+=1
    ]
    
    # Verify Strassen 2x2
    # We will assume this is correct.
    
    # Construct 4x4x4 decomposition
    # A, B, C are 4x4 matrices.
    # Blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    # Same for B, C.
    
    # Block indices mapping:
    # Block (i,j) of 4x4 matrix corresponds to 2x2 block.
    # i=0, j=0 -> indices 0,1,4,5
    # i=0, j=1 -> indices 2,3,6,7
    # i=1, j=0 -> indices 8,9,12,13
    # i=1, j=1 -> indices 10,11,14,15
    
    def get_block_indices(i, j):
        # i, j in {0, 1}
        base = i * 8 + j * 2
        return [base, base + 1, base + 4, base + 5]
        
    # Block-level Strassen for 4x4x4
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Z00 (C00) = M1 + M4 - M5 + M7
    # Z01 (C01) = M3 + M5
    # Z10 (C10) = M2 + M4
    # Z11 (C11) = M1 - M2 + M3 + M6
    
    # For each block-level term k, we have:
    # X_k = sum_{i,j} coeff_k[i,j] * A_{ij}
    # Y_k = sum_{i,j} coeff_k[i,j] * B_{ij}
    # Z_k = X_k Y_k
    # C += sum_{i,j} out_coeff_k[i,j] * Z_k_{ij}
    
    # X_k is a 2x2 matrix of blocks? No, X_k is a 2x2 matrix of scalars?
    # No, X_k is a 2x2 matrix where each entry is a linear combo of A's entries.
    # Wait, X_k is a 2x2 matrix.
    # X_k = [ X_{00}, X_{01}; X_{10}, X_{11} ]
    # X_{ij} is a scalar (linear combo of A's 16 entries).
    
    # Let's define the block-level Strassen coefficients.
    # For each k in 1..7:
    # u_block_k: 4x4 matrix? No, 2x2 matrix of linear forms on A.
    # Actually, X_k is a 2x2 matrix.
    # X_{00} = linear form on A.
    # X_{01} = linear form on A.
    # X_{10} = linear form on A.
    # X_{11} = linear form on A.
    
    # We can represent X_k as a 4x4 matrix where the 2x2 blocks are the linear forms?
    # No, X_k is just a 2x2 matrix.
    # The entries of X_k are linear combinations of the entries of A.
    # So we can represent the linear form for each entry of X_k as a vector of length 16.
    
    # Let's define the 7 block-level terms.
    # Each term is defined by:
    # - A_coeffs: 2x2 matrix of vectors of length 16.
    # - B_coeffs: 2x2 matrix of vectors of length 16.
    # - C_coeffs: 2x2 matrix of scalars (coefficients for the resulting 2x2 block of C).
    
    # M1 = (A00+A11)(B00+B11)
    # A00: block 00, A11: block 11.
    # X_{00} = A00_{00} + A11_{00}? No.
    # X is a 2x2 matrix.
    # X = A00 + A11?
    # A00 is a 2x2 matrix. A11 is a 2x2 matrix.
    # X = A00 + A11 is a 2x2 matrix.
    # So X_{00} = A00_{00} + A11_{00}.
    # X_{01} = A00_{01} + A11_{01}.
    # X_{10} = A00_{10} + A11_{10}.
    # X_{11} = A00_{11} + A11_{11}.
    
    # So for M1:
    # X = A00 + A11.
    # Y = B00 + B11.
    # Z = XY.
    # C00 += Z. (So C_coeffs[0,0] = 1, others 0? No, Z is added to C00).
    # Wait, C00 is a 2x2 block.
    # So C00_{00} += Z_{00}, etc.
    # So the contribution to C is:
    # C_{0,0} block += Z.
    
    # Let's formalize.
    # For each k in 1..7:
    # X_k = sum_{i,j in {0,1}} alpha_{k,ij} A_{ij}
    # Y_k = sum_{i,j in {0,1}} beta_{k,ij} B_{ij}
    # Z_k = X_k Y_k
    # C_{ij} += gamma_{k,ij} Z_k
    
    # Where A_{ij} is the 2x2 block at (i,j) of A.
    
    # M1: X = A00 + A11, Y = B00 + B11, C00 += Z
    # M2: X = A10 + A11, Y = B00, C10 += Z
    # M3: X = A00, Y = B01 - B11, C01 += Z, C11 += Z
    # M4: X = A11, Y = B10 - B00, C00 += Z, C10 += Z
    # M5: X = A00 + A01, Y = B11, C00 -= Z, C01 += Z
    # M6: X = A10 - A00, Y = B00 + B01, C11 += Z
    # M7: X = A01 - A11, Y = B10 + B11, C00 += Z
    
    # We need to construct the vectors for U, V, W for the 4x4x4 tensor.
    # For each k, and for each inner Strassen term r (1..7):
    # u_r(X) is a linear form on X.
    # v_r(Y) is a linear form on Y.
    # w_r is a 2x2 matrix of coefficients for Z.
    
    # u_r(X) = sum_{p,q} u_{r,pq} X_{pq}
    # X_{pq} = sum_{i,j} alpha_{k,ij} A_{ij,pq}
    # So u_r(X) = sum_{i,j,p,q} u_{r,pq} alpha_{k,ij} A_{ij,pq}
    # This is a linear form on A.
    # We can compute the vector of coefficients for A.
    
    # Similarly for v_r(Y) on B.
    
    # The contribution to C is:
    # C_{ij} += gamma_{k,ij} Z_k
    # Z_k_{pq} = u_r(X) v_r(Y) w_{r,pq}
    # So C_{ij,pq} += gamma_{k,ij} w_{r,pq} u_r(X) v_r(Y)
    
    # So for each (k, r), we have a rank-1 term.
    # U_vec: coefficients for A.
    # V_vec: coefficients for B.
    # W_vec: coefficients for C.
    
    # Let's implement this.
    
    # Helper to get vector for block i,j
    def block_vec(i, j):
        # Returns a vector of length 16 with 1s at the positions of block (i,j)
        vec = [0.0] * 16
        idxs = get_block_indices(i, j)
        for idx in idxs:
            vec[idx] = 1.0
        return vec
    
    # Define block-level terms
    # alpha_k: 2x2 matrix of coefficients for A blocks
    # beta_k: 2x2 matrix of coefficients for B blocks
    # gamma_k: 2x2 matrix of coefficients for C blocks
    
    block_terms = [
        # k=1
        {
            'alpha': [[1, 0], [0, 1]], # A00 + A11
            'beta': [[1, 0], [0, 1]],  # B00 + B11
            'gamma': [[1, 0], [0, 0]]  # C00 += Z
        },
        # k=2
        {
            'alpha': [[0, 0], [1, 1]], # A10 + A11
            'beta': [[1, 0], [0, 0]],  # B00
            'gamma': [[0, 0], [1, 0]]  # C10 += Z
        },
        # k=3
        {
            'alpha': [[1, 0], [0, 0]], # A00
            'beta': [[0, 1], [0, -1]], # B01 - B11
            'gamma': [[0, 1], [0, 1]]  # C01 += Z, C11 += Z
        },
        # k=4
        {
            'alpha': [[0, 0], [0, 1]], # A11
            'beta': [[0, 0], [1, -1]], # B10 - B00
            'gamma': [[1, 0], [1, 0]]  # C00 += Z, C10 += Z
        },
        # k=5
        {
            'alpha': [[1, 1], [0, 0]], # A00 + A01
            'beta': [[0, 0], [0, 1]],  # B11
            'gamma': [[-1, 1], [0, 0]] # C00 -= Z, C01 += Z
        },
        # k=6
        {
            'alpha': [[-1, 0], [1, 0]], # A10 - A00
            'beta': [[1, 1], [0, 0]],   # B00 + B01
            'gamma': [[0, 0], [0, 1]]   # C11 += Z
        },
        # k=7
        {
            'alpha': [[0, 1], [0, -1]], # A01 - A11
            'beta': [[0, 0], [1, 1]],   # B10 + B11
            'gamma': [[1, 0], [0, 0]]   # C00 += Z
        }
    ]
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    for term in block_terms:
        alpha = term['alpha']
        beta = term['beta']
        gamma = term['gamma']
        
        # Construct X_k and Y_k as linear forms on A and B
        # X_k is a 2x2 matrix of linear forms.
        # X_{pq} = sum_{i,j} alpha_{ij} A_{ij,pq}
        
        for r, (u_r, v_r, w_r) in enumerate(strassen_2x2):
            # u_r is a vector of length 4 for X.
            # u_r = [u00, u01, u10, u11]
            # u_r(X) = u00 X00 + u01 X01 + u10 X10 + u11 X11
            
            # Compute U_vec for A
            U_vec = [0.0] * 16
            for p in range(2):
                for q in range(2):
                    coeff_u = u_r[p*2+q]
                    if coeff_u == 0: continue
                    # X_{pq} = sum_{i,j} alpha_{ij} A_{ij,pq}
                    for i in range(2):
                        for j in range(2):
                            coeff_a = alpha[i][j]
                            if coeff_a == 0: continue
                            # Add coeff_u * coeff_a to A_{ij,pq}
                            idxs = get_block_indices(i, j)
                            # A_{ij,pq} is at idxs[p*2+q]
                            idx = idxs[p*2+q]
                            U_vec[idx] += coeff_u * coeff_a
            
            # Compute V_vec for B
            V_vec = [0.0] * 16
            for p in range(2):
                for q in range(2):
                    coeff_v = v_r[p*2+q]
                    if coeff_v == 0: continue
                    for i in range(2):
                        for j in range(2):
                            coeff_b = beta[i][j]
                            if coeff_b == 0: continue
                            idxs = get_block_indices(i, j)
                            idx = idxs[p*2+q]
                            V_vec[idx] += coeff_v * coeff_b
            
            # Compute W_vec for C
            # C_{ij} += gamma_{ij} Z_k
            # Z_k_{pq} = u_r(X) v_r(Y) w_r_{pq}
            # So C_{ij,pq} += gamma_{ij} w_r_{pq} u_r(X) v_r(Y)
            
            W_vec = [0.0] * 16
            for i in range(2):
                for j in range(2):
                    coeff_g = gamma[i][j]
                    if coeff_g == 0: continue
                    for p in range(2):
                        for q in range(2):
                            coeff_w = w_r[p*2+q]
                            if coeff_w == 0: continue
                            idxs = get_block_indices(i, j)
                            idx = idxs[p*2+q]
                            W_vec[idx] += coeff_g * coeff_w
            
            U_4x4.append(U_vec)
            V_4x4.append(V_vec)
            W_4x4.append(W_vec)
            
    # Now we have a rank-49 decomposition for 4x4x4.
    # We need to extend this to 4x4x8.
    # B is 4x8. Split into B_left (4x4) and B_right (4x4).
    # C is 4x8. Split into C_left (4x4) and C_right (4x4).
    
    # For the left block:
    # U is the same.
    # V is V_4x4, but mapped to the first 16 entries of B (4x8 flattened to 32).
    # B_left indices: 0..15.
    # B_right indices: 16..31.
    
    # For the right block:
    # U is the same.
    # V is V_4x4, but mapped to the last 16 entries of B.
    # W is W_4x4, but mapped to the last 16 entries of C.
    # C_left indices: 0..15.
    # C_right indices: 16..31.
    
    U_final = []
    V_final = []
    W_final = []
    
    for u, v, w in zip(U_4x4, V_4x4, W_4x4):
        # Left block
        V_left = [0.0] * 32
        for i in range(16):
            V_left[i] = v[i]
        W_left = [0.0] * 32
        for i in range(16):
            W_left[i] = w[i]
        
        U_final.append(u)
        V_final.append(V_left)
        W_final.append(W_left)
        
        # Right block
        V_right = [0.0] * 32
        for i in range(16):
            V_right[16+i] = v[i]
        W_right = [0.0] * 32
        for i in range(16):
            W_right[16+i] = w[i]
        
        U_final.append(u)
        V_final.append(V_right)
        W_final.append(W_right)
        
    rank = len(U_final)
    
    result = {
        "U": U_final,
        "V": V_final,
        "W": W_final
    }
    
    return result
# EVOLVE-BLOCK-END
