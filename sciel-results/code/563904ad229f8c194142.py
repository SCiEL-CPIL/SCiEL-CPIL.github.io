import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using Strassen for 2x2x2 blocks
    rank = 28
    
    # U shape: (n*m, rank) = (8, 28)
    # V shape: (m*p, rank) = (16, 28)
    # W shape: (p*n, rank) = (8, 28)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We need to map the 2x2 Strassen terms to the 2x4x4 indices.
    # A is 2x4. Indices i (0..1), j (0..3).
    # B is 4x4. Indices j (0..3), k (0..3).
    # C is 2x4. Indices i (0..1), k (0..3).
    # W index is k*n + i.
    
    # Blocks:
    # A1: A[0:2, 0:2] -> i in 0..1, j in 0..1
    # A2: A[0:2, 2:4] -> i in 0..1, j in 2..3
    # B1: B[0:2, 0:4] -> j in 0..1, k in 0..3
    # B2: B[2:4, 0:4] -> j in 2..3, k in 0..3
    
    # C1 = A1 B11 + A2 B21
    # B11: B[0:2, 0:2] -> j in 0..1, k in 0..1
    # B21: B[2:4, 0:2] -> j in 2..3, k in 0..1
    # C1 is C[:, 0:2] -> i in 0..1, k in 0..1
    
    # C2 = A1 B12 + A2 B22
    # B12: B[0:2, 2:4] -> j in 0..1, k in 2..3
    # B22: B[2:4, 2:4] -> j in 2..3, k in 2..3
    # C2 is C[:, 2:4] -> i in 0..1, k in 2..3
    
    # Strassen 2x2x2 terms for A_block * B_block
    # A_block elements: a,b,c,d at local (0,0), (0,1), (1,0), (1,1)
    # B_block elements: e,f,g,h at local (0,0), (0,1), (1,0), (1,1)
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # We will define a helper to add a Strassen block to the global matrices.
    # offset_u, offset_v, offset_w are the global row indices for the 2x2 block.
    # block_col_offset is the column offset in U,V,W for this set of 7 terms.
    
    def add_strassen_block(A_i_start, A_j_start, B_j_start, B_k_start, C_i_start, C_k_start, col_offset):
        # A block is 2x2 starting at (A_i_start, A_j_start)
        # B block is 2x2 starting at (B_j_start, B_k_start)
        # C block is 2x2 starting at (C_i_start, C_k_start)
        
        # Local indices for A: (0,0)->a, (0,1)->b, (1,0)->c, (1,1)->d
        # Global U index: (A_i_start + di) * m + (A_j_start + dj)
        # Local indices for B: (0,0)->e, (0,1)->f, (1,0)->g, (1,1)->h
        # Global V index: (B_j_start + dj) * p + (B_k_start + dk)
        # Local indices for C: (0,0)->C11, (0,1)->C12, (1,0)->C21, (1,1)->C22
        # Global W index: (C_k_start + dk) * n + (C_i_start + di)
        
        # P1 = (a+d)(e+h)
        # U: a, d; V: e, h; W: C11, C22
        # U[a] += 1, U[d] += 1
        # V[e] += 1, V[h] += 1
        # W[C11] += 1, W[C22] += 1
        
        def add_term(u_idx, v_idx, w_idx, sign):
            U[u_idx, col_offset] += sign
            V[v_idx, col_offset] += sign
            W[w_idx, col_offset] += sign

        # a
        a_idx = (A_i_start + 0) * m + (A_j_start + 0)
        b_idx = (A_i_start + 0) * m + (A_j_start + 1)
        c_idx = (A_i_start + 1) * m + (A_j_start + 0)
        d_idx = (A_i_start + 1) * m + (A_j_start + 1)
        
        # e
        e_idx = (B_j_start + 0) * p + (B_k_start + 0)
        f_idx = (B_j_start + 0) * p + (B_k_start + 1)
        g_idx = (B_j_start + 1) * p + (B_k_start + 0)
        h_idx = (B_j_start + 1) * p + (B_k_start + 1)
        
        # C11
        c11_idx = (C_k_start + 0) * n + (C_i_start + 0)
        c12_idx = (C_k_start + 1) * n + (C_i_start + 0)
        c21_idx = (C_k_start + 0) * n + (C_i_start + 1)
        c22_idx = (C_k_start + 1) * n + (C_i_start + 1)
        
        # P1 = (a+d)(e+h)
        # Terms: ae, ah, de, dh
        # W contributions: C11 += 1, C22 += 1
        # Wait, Strassen sums are:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We need to add the products to the tensor.
        # The tensor decomposition is sum_k U_k V_k W_k.
        # Each P_i is a bilinear form.
        # P1 = (a+d)(e+h) = ae + ah + de + dh
        # This corresponds to 4 rank-1 terms? No, it's one rank-1 term in the bilinear sense if we consider the product of sums?
        # No, the tensor rank decomposition is $\sum_{r} u_r \otimes v_r \otimes w_r$.
        # Strassen's algorithm computes the product using 7 multiplications.
        # Each multiplication $P_i$ is a product of linear combinations of A and B entries.
        # $P_i = (\sum u_{ij} A_{ij}) (\sum v_{jk} B_{jk})$.
        # This is not a rank-1 term in the tensor $T_{ijk}$ directly.
        # The tensor is $T_{ijk} A_{ij} B_{jk} = C_{ik}$.
        # Strassen expresses $C_{ik}$ as linear combinations of $P_l$.
        # $C_{ik} = \sum_l w_{ikl} P_l$.
        # And $P_l = \sum_{ij} u_{ijl} A_{ij} \sum_{jk} v_{jkl} B_{jk}$.
        # So $C_{ik} = \sum_l w_{ikl} (\sum_{ij} u_{ijl} A_{ij}) (\sum_{jk} v_{jkl} B_{jk})$.
        # This is a sum of $L$ terms, each of which is a product of a linear form in A and a linear form in B.
        # This corresponds to a rank $L$ decomposition of the tensor.
        
        # So for each Strassen term $P_l$, we have:
        # $u_l$ is the vector of coefficients for A.
        # $v_l$ is the vector of coefficients for B.
        # $w_l$ is the vector of coefficients for C (how $P_l$ contributes to $C_{ik}$).
        
        # Let's define the coefficients for the 7 Strassen terms.
        # P1: (a+d)(e+h)
        # u: a:1, d:1
        # v: e:1, h:1
        # w: C11:1, C22:1
        
        # P2: (c+d)e
        # u: c:1, d:1
        # v: e:1
        # w: C21:1, C22:-1
        
        # P3: a(f-h)
        # u: a:1
        # v: f:1, h:-1
        # w: C12:1, C22:1
        
        # P4: d(g-e)
        # u: d:1
        # v: g:1, e:-1
        # w: C11:1, C21:1
        
        # P5: (a+b)h
        # u: a:1, b:1
        # v: h:1
        # w: C11:-1, C12:1
        
        # P6: (c-a)(e+f)
        # u: c:1, a:-1
        # v: e:1, f:1
        # w: C22:1
        
        # P7: (b-d)(g+h)
        # u: b:1, d:-1
        # v: g:1, h:1
        # w: C11:1
        
        # We will iterate through these 7 terms and add them to U, V, W.
        
        terms = [
            # (u_dict, v_dict, w_dict)
            ({a_idx: 1, d_idx: 1}, {e_idx: 1, h_idx: 1}, {c11_idx: 1, c22_idx: 1}),
            ({c_idx: 1, d_idx: 1}, {e_idx: 1}, {c21_idx: 1, c22_idx: -1}),
            ({a_idx: 1}, {f_idx: 1, h_idx: -1}, {c12_idx: 1, c22_idx: 1}),
            ({d_idx: 1}, {g_idx: 1, e_idx: -1}, {c11_idx: 1, c21_idx: 1}),
            ({a_idx: 1, b_idx: 1}, {h_idx: 1}, {c11_idx: -1, c12_idx: 1}),
            ({c_idx: 1, a_idx: -1}, {e_idx: 1, f_idx: 1}, {c22_idx: 1}),
            ({b_idx: 1, d_idx: -1}, {g_idx: 1, h_idx: 1}, {c11_idx: 1})
        ]
        
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = col_offset + t_idx
            for idx, val in u_dict.items():
                U[idx, col] += val
            for idx, val in v_dict.items():
                V[idx, col] += val
            for idx, val in w_dict.items():
                W[idx, col] += val

    # 1. A1 * B11
    # A1: i=0..1, j=0..1. B11: j=0..1, k=0..1. C11: i=0..1, k=0..1.
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    
    # 2. A2 * B21
    # A2: i=0..1, j=2..3. B21: j=2..3, k=0..1. C11: i=0..1, k=0..1.
    # Note: C11 is the same block in C.
    # The contributions add up.
    add_strassen_block(0, 2, 2, 0, 0, 0, 7)
    
    # 3. A1 * B12
    # A1: i=0..1, j=0..1. B12: j=0..1, k=2..3. C12: i=0..1, k=2..3.
    add_strassen_block(0, 0, 0, 2, 0, 2, 14)
    
    # 4. A2 * B22
    # A2: i=0..1, j=2..3. B22: j=2..3, k=2..3. C12: i=0..1, k=2..3.
    add_strassen_block(0, 2, 2, 2, 0, 2, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
