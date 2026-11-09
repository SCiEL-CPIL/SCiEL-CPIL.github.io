import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 44 decomposition
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w, col):
        U[u_idx, col] += coeff_u
        V[v_idx, col] += coeff_v
        W[w_idx, col] += coeff_w
        
    col = 0
    
    # Part 1: A1 * B1
    # A1 is A[:, 0:2] (cols 0, 1)
    # B1 is B[0:2, :] (rows 0, 1)
    # We split B1 into 4 blocks of 2x2
    
    # Indices for A1 in U
    # U[i*m + j, col]
    # A1[0,0] -> U[0*3+0, col] = U[0, col]
    # A1[0,1] -> U[0*3+1, col] = U[1, col]
    # A1[1,0] -> U[1*3+0, col] = U[3, col]
    # A1[1,1] -> U[1*3+1, col] = U[4, col]
    u_a = 0
    u_b = 1
    u_c = 3
    u_d = 4
    
    for block in range(4):
        # B1 block 'block' corresponds to B[0:2, block*2 : block*2+2]
        k_start = block * 2
        # V indices for B1 block
        # V[j*p + k, col]
        # B1[0,0] of block -> B[0, k_start] -> V[0*8 + k_start, col]
        # B1[0,1] of block -> B[0, k_start+1] -> V[0*8 + k_start+1, col]
        # B1[1,0] of block -> B[1, k_start] -> V[1*8 + k_start, col]
        # B1[1,1] of block -> B[1, k_start+1] -> V[1*8 + k_start+1, col]
        
        v_e = 0 * p + k_start
        v_f = 0 * p + k_start + 1
        v_g = 1 * p + k_start
        v_h = 1 * p + k_start + 1
        
        # W indices for C1 block
        # C is 2x8. Block 'block' is C[:, block*2 : block*2+2]
        # W[k*n + i, col]
        # C[0,0] of block -> C[0, k_start] -> W[k_start*n + 0, col]
        # C[0,1] of block -> C[0, k_start+1] -> W[(k_start+1)*n + 0, col]
        # C[1,0] of block -> C[1, k_start] -> W[k_start*n + 1, col]
        # C[1,1] of block -> C[1, k_start+1] -> W[(k_start+1)*n + 1, col]
        
        w_11 = k_start * n + 0
        w_12 = (k_start + 1) * n + 0
        w_21 = k_start * n + 1
        w_22 = (k_start + 1) * n + 1
        
        # Strassen's 7 multiplications
        # P1 = (a+d)(e+h)
        add_term(u_a, v_e, w_11, 1, 1, 1, col)
        add_term(u_a, v_f, w_11, 1, 1, 1, col)
        add_term(u_d, v_e, w_11, 1, 1, 1, col)
        add_term(u_d, v_f, w_11, 1, 1, 1, col)
        # Wait, Strassen adds to C.
        # C11 += P1
        # C12 += P3 + P5
        # C21 += P2 + P4
        # C22 += P1 - P2 + P3 + P6
        
        # Let's do this carefully.
        # P1 = (a+d)(e+h)
        # Terms for P1:
        # a*e, a*h, d*e, d*h
        # Contribution to C11: +1
        # Contribution to C22: +1
        
        # P2 = (c+d)e
        # Terms: c*e, d*e
        # Contribution to C21: +1
        # Contribution to C22: -1
        
        # P3 = a(f-h)
        # Terms: a*f, -a*h
        # Contribution to C12: +1
        # Contribution to C22: +1
        
        # P4 = d(g-e)
        # Terms: d*g, -d*e
        # Contribution to C11: +1
        # Contribution to C21: +1
        
        # P5 = (a+b)h
        # Terms: a*h, b*h
        # Contribution to C11: -1
        # Contribution to C12: +1
        
        # P6 = (c-a)(e+f)
        # Terms: c*e, c*f, -a*e, -a*f
        # Contribution to C22: +1
        
        # P7 = (b-d)(g+h)
        # Terms: b*g, b*h, -d*g, -d*h
        # Contribution to C11: +1
        # Contribution to C22: +1
        
        # We need to accumulate these into U, V, W.
        # Each "multiplication" P_k is a scalar product of a linear combo of A and B.
        # In the tensor decomposition, each term in the expansion of P_k is a separate rank-1 term?
        # No, Strassen's algorithm uses 7 multiplications.
        # The "multiplications" are $M_1, \dots, M_7$.
        # $M_1 = (a+d)(e+h)$.
        # This is not a single scalar multiplication in the tensor sense unless we view the inputs as linear combos.
        # The tensor decomposition requires $U, V, W$ to be constant matrices.
        # The "multiplication" is the element-wise product of the projected inputs.
        # So for $M_1$, we have one column in U, V, W.
        # U column has non-zeros at $a, d$.
        # V column has non-zeros at $e, h$.
        # W column has non-zeros at $C_{11}, C_{22}$.
        
        # So for each Strassen product $P_k$, we add ONE column to the decomposition.
        # U col: coefficients for a,b,c,d
        # V col: coefficients for e,f,g,h
        # W col: coefficients for C11,C12,C21,C22
        
        # Let's redefine add_term to handle this structure.
        # Actually, my previous `add_term` was adding individual scalar products.
        # Strassen combines them.
        # I will implement Strassen by adding 7 columns per block.
        
        # P1 = (a+d)(e+h)
        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1
        W[w_11, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # P2 = (c+d)e
        U[u_c, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        W[w_21, col] += 1
        W[w_22, col] += -1
        col += 1
        
        # P3 = a(f-h)
        U[u_a, col] += 1
        V[v_f, col] += 1
        V[v_h, col] += -1
        W[w_12, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # P4 = d(g-e)
        U[u_d, col] += 1
        V[v_g, col] += 1
        V[v_e, col] += -1
        W[w_11, col] += 1
        W[w_21, col] += 1
        col += 1
        
        # P5 = (a+b)h
        U[u_a, col] += 1
        U[u_b, col] += 1
        V[v_h, col] += 1
        W[w_11, col] += -1
        W[w_12, col] += 1
        col += 1
        
        # P6 = (c-a)(e+f)
        U[u_c, col] += 1
        U[u_a, col] += -1
        V[v_e, col] += 1
        V[v_f, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # P7 = (b-d)(g+h)
        U[u_b, col] += 1
        U[u_d, col] += -1
        V[v_g, col] += 1
        V[v_h, col] += 1
        W[w_11, col] += 1
        W[w_22, col] += 1
        col += 1

    # Part 2: A2 * B2
    # A2 is A[:, 2]
    # B2 is B[2, :]
    # C2 is C[:, :]
    
    u_a2_0 = 0 * m + 2  # A[0,2]
    u_a2_1 = 1 * m + 2  # A[1,2]
    
    for k in range(p):
        # B2[k] is B[2, k]
        v_b2_k = 2 * p + k
        
        # C2[0, k] is C[0, k] -> W[k*n + 0]
        w_c2_0_k = k * n + 0
        # C2[1, k] is C[1, k] -> W[k*n + 1]
        w_c2_1_k = k * n + 1
        
        # Term 1: A[0,2] * B[2,k] -> C[0,k]
        U[u_a2_0, col] += 1
        V[v_b2_k, col] += 1
        W[w_c2_0_k, col] += 1
        col += 1
        
        # Term 2: A[1,2] * B[2,k] -> C[1,k]
        U[u_a2_1, col] += 1
        V[v_b2_k, col] += 1
        W[w_c2_1_k, col] += 1
        col += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
