import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-43 tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Uses Strassen's algorithm for two 2x2x2 sub-blocks to reduce rank from 45 to 43.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # We will build U, V, W as lists of lists
    # U: 9 x rank
    # V: 15 x rank
    # W: 15 x rank
    
    # We don't know the final rank yet, so we'll collect terms in a list
    # Each term is (u_vec, v_vec, w_vec) where vectors are sparse dicts or lists
    # Using lists of size 9, 15, 15 is efficient enough for small ranks
    
    terms = []
    
    def add_schoolbook_term(i, j, k):
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        
        terms.append((u, v, w))

    def add_strassen_block(i_offsets, j_offsets, k_offsets):
        """
        Adds the 7 Strassen terms for a 2x2x2 block.
        i_offsets: list of 2 global i indices
        j_offsets: list of 2 global j indices
        k_offsets: list of 2 global k indices
        """
        i0, i1 = i_offsets
        j0, j1 = j_offsets
        k0, k1 = k_offsets
        
        # Local indices 0, 1 map to global i0, i1 etc.
        # u vector: size 9 (n*m)
        # v vector: size 15 (m*p)
        # w vector: size 15 (p*n)
        
        def get_u(i_l, j_l, val):
            u = [0] * (n * m)
            i_g = i_offsets[i_l]
            j_g = j_offsets[j_l]
            u[i_g * m + j_g] = val
            return u

        def get_v(j_l, k_l, val):
            v = [0] * (m * p)
            j_g = j_offsets[j_l]
            k_g = k_offsets[k_l]
            v[j_g * p + k_g] = val
            return v

        def get_w(k_l, i_l, val):
            w = [0] * (p * n)
            k_g = k_offsets[k_l]
            i_g = i_offsets[i_l]
            w[k_g * n + i_g] = val
            return w
        
        def add_vecs(v1, v2, val1=1, val2=1):
            return [v1[i] * val1 + v2[i] * val2 for i in range(len(v1))]
        
        def sub_vecs(v1, v2):
            return [v1[i] - v2[i] for i in range(len(v1))]

        # Define the 7 terms
        # M1 = (A00 + A11)(B00 + B11)
        # A00: u(0,0), A11: u(1,1)
        # B00: v(0,0), B11: v(1,1)
        # C00: w(0,0), C11: w(1,1)
        u1 = add_vecs(get_u(0,0,1), get_u(1,1,1))
        v1 = add_vecs(get_v(0,0,1), get_v(1,1,1))
        w1 = add_vecs(get_w(0,0,1), get_w(1,1,1))
        terms.append((u1, v1, w1))
        
        # M2 = (A10 + A11)B00
        u2 = add_vecs(get_u(1,0,1), get_u(1,1,1))
        v2 = get_v(0,0,1)
        # C10: w(0,1) -> k=0, i=1
        w2 = get_w(0,1,1)
        terms.append((u2, v2, w2))
        
        # M3 = A00(B10 - B11)
        u3 = get_u(0,0,1)
        v3 = sub_vecs(get_v(1,0,1), get_v(1,1,1))
        # C10: w(0,1)
        w3 = get_w(0,1,1)
        terms.append((u3, v3, w3))
        
        # M4 = A11(B01 - B00)
        u4 = get_u(1,1,1)
        v4 = sub_vecs(get_v(0,1,1), get_v(0,0,1))
        # C00: w(0,0)
        w4 = get_w(0,0,1)
        terms.append((u4, v4, w4))
        
        # M5 = (A00 + A01)B11
        u5 = add_vecs(get_u(0,0,1), get_u(0,1,1))
        v5 = get_v(1,1,1)
        # C01: w(1,0) -> k=1, i=0
        w5 = get_w(1,0,1)
        terms.append((u5, v5, w5))
        
        # M6 = (A10 - A00)(B00 + B01)
        u6 = sub_vecs(get_u(1,0,1), get_u(0,0,1))
        v6 = add_vecs(get_v(0,0,1), get_v(0,1,1))
        # C01: w(1,0)
        w6 = get_w(1,0,1)
        terms.append((u6, v6, w6))
        
        # M7 = (A01 - A11)(B10 + B11)
        u7 = sub_vecs(get_u(0,1,1), get_u(1,1,1))
        v7 = add_vecs(get_v(1,0,1), get_v(1,1,1))
        # C00: w(0,0), C11: w(1,1)
        w7 = add_vecs(get_w(0,0,1), get_w(1,1,1))
        terms.append((u7, v7, w7))

    # Identify schoolbook terms to keep
    # Total 45 terms.
    # Remove those in Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Remove those in Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    
    block1_indices = {(i, j, k) for i in [0, 1] for j in [0, 1] for k in [0, 1]}
    block2_indices = {(i, j, k) for i in [0, 1] for j in [0, 1] for k in [2, 3]}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in block1_indices or (i, j, k) in block2_indices:
                    continue
                add_schoolbook_term(i, j, k)
                
    # Add Strassen blocks
    add_strassen_block([0, 1], [0, 1], [0, 1])
    add_strassen_block([0, 1], [0, 1], [2, 3])
    
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u, v, w = terms[c]
        for i in range(n * m):
            if u[i] != 0:
                U[i][c] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][c] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][c] = w[k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
