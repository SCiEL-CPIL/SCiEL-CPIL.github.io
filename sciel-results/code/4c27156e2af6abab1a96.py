"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # We will build U, V, W as lists of lists.
    # U: (n*m) x rank
    # V: (m*p) x rank
    # W: (p*n) x rank
    
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        u_dict: {local_index: coeff}
        v_dict: {local_index: coeff}
        w_dict: {local_index: coeff}
        """
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        
        for idx, coeff in u_dict.items():
            if coeff != 0:
                u_row[idx] += coeff
        for idx, coeff in v_dict.items():
            if coeff != 0:
                v_row[idx] += coeff
        for idx, coeff in w_dict.items():
            if coeff != 0:
                w_row[idx] += coeff
                
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)

    # Recursive Strassen decomposition generator
    # Returns a list of (u_dict, v_dict, w_dict) for the given dimensions
    def strassen_decomp(n, m, p):
        if n == 2 and m == 2 and p == 2:
            # Local indices for 2x2:
            # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            
            terms = []
            
            # M1 = (A00+A11)(B00+B11) -> C00(+), C11(+)
            terms.append((
                {0: 1, 3: 1}, 
                {0: 1, 3: 1}, 
                {0: 1, 3: 1}
            ))
            
            # M2 = (A10+A11)B00 -> C10(+), C11(-)
            terms.append((
                {2: 1, 3: 1}, 
                {0: 1}, 
                {2: 1, 3: -1}
            ))
            
            # M3 = A00(B01-B11) -> C01(+), C11(+)
            terms.append((
                {0: 1}, 
                {1: 1, 3: -1}, 
                {1: 1, 3: 1}
            ))
            
            # M4 = A11(B10-B00) -> C10(+)
            terms.append((
                {3: 1}, 
                {2: 1, 0: -1}, 
                {2: 1}
            ))
            
            # M5 = (A00+A01)B11 -> C00(-), C01(+)
            terms.append((
                {0: 1, 1: 1}, 
                {3: 1}, 
                {0: -1, 1: 1}
            ))
            
            # M6 = (A10-A00)(B00+B01) -> C11(+)
            terms.append((
                {2: 1, 0: -1}, 
                {0: 1, 1: 1}, 
                {3: 1}
            ))
            
            # M7 = (A01-A11)(B10+B11) -> C00(+)
            terms.append((
                {1: 1, 3: -1}, 
                {2: 1, 3: 1}, 
                {0: 1}
            ))
            
            return terms

        # Recursive step
        # Assumes n, m, p are even
        n2, m2, p2 = n // 2, m // 2, p // 2
        
        sub_terms = strassen_decomp(n2, m2, p2)
        
        # We need to map local indices of sub-problem to global indices
        # and define the 7 Strassen combinations.
        
        # Block offsets in A (n x m)
        # A11: (0, 0), A12: (0, m2), A21: (n2, 0), A22: (n2, m2)
        # Local index (i, j) in block -> Global (i + r*n2, j + c*m2)
        # But the sub_terms use a flattened index for n2*m2.
        # We need a helper to map local flat index to global flat index for a specific block.
        
        def map_index(local_idx, block_r, block_c, block_n, block_m, total_m):
            # local_idx is 0..block_n*block_m-1
            i = local_idx // block_m
            j = local_idx % block_m
            global_i = i + block_r * block_n
            global_j = j + block_c * block_m
            return global_i * total_m + global_j

        def map_index_C(local_idx, block_r, block_c, block_n, block_p, total_n):
            # C is p x n. But wait.
            # In the problem, W is (p*n). Index is k*n + i.
            # So C rows are k (0..p-1), C cols are i (0..n-1).
            # In Strassen, C is n x p?
            # Standard matrix mult: C = A B. A is n x m, B is m x p. C is n x p.
            # The tensor index for W is k*n + i.
            # This implies W is indexed by (k, i).
            # k is col of B (0..p-1). i is row of A (0..n-1).
            # So C_{k,i} = sum_j A_{i,j} B_{j,k}.
            # This matches C being p x n in the W matrix shape?
            # W shape is (p*n).
            # Yes.
            
            # So for C blocks, the "row" is k and "col" is i.
            # Block structure of C (p x n):
            # C11 (top-left of C in standard n x p view? No.)
            # Let's stick to the indices.
            # A is n x m. B is m x p.
            # C is n x p.
            # W index: k*n + i.
            # So W[r] corresponds to C[k, i] where r = k*n + i.
            
            # In Strassen, we split A (n x m), B (m x p), C (n x p).
            # n, m, p split in half.
            # A11 (n2 x m2), A12 (n2 x m2), A21 (n2 x m2), A22 (n2 x m2).
            # B11 (m2 x p2), B12 (m2 x p2), B21 (m2 x p2), B22 (m2 x p2).
            # C11 (n2 x p2), C12 (n2 x p2), C21 (n2 x p2), C22 (n2 x p2).
            
            # C11 corresponds to rows 0..n2-1, cols 0..p2-1 of C.
            # In W index (k*n + i):
            # k is col of C (0..p2-1). i is row of C (0..n2-1).
            # So C11 uses k in 0..p2-1, i in 0..n2-1.
            # C12 uses k in p2..p-1, i in 0..n2-1.
            # C21 uses k in 0..p2-1, i in n2..n-1.
            # C22 uses k in p2..p-1, i in n2..n-1.
            
            i = local_idx // p2  # Wait, local C is n2 x p2.
            # Local C index: row i_local (0..n2-1), col k_local (0..p2-1).
            # Flat index: i_local * p2 + k_local.
            # So i_local = local_idx // p2.
            # k_local = local_idx % p2.
            
            i_local = local_idx // p2
            k_local = local_idx % p2
            
            global_i = i_local + block_r * n2
            global_k = k_local + block_c * p2
            
            return global_k * n + global_i

        # Define the 7 combinations
        # Each combination is a list of (A_block, B_block, C_block, sign)
        # But A and B can have multiple blocks with same sign.
        
        # M1: (A11+A22)(B11+B22) -> C11(+), C22(+)
        # M2: (A21+A22)B11 -> C21(+), C22(-)
        # M3: A11(B12-B22) -> C12(+), C22(+)
        # M4: A22(B21-B11) -> C21(+)
        # M5: (A11+A12)B22 -> C11(-), C12(+)
        # M6: (A21-A11)(B11+B12) -> C22(+)
        # M7: (A12-A22)(B21+B22) -> C11(+)
        
        # We iterate over the sub_terms.
        # For each sub_term (u_sub, v_sub, w_sub), we create 7 global terms.
        
        global_terms = []
        
        for u_sub, v_sub, w_sub in sub_terms:
            
            # M1
            u1 = {}
            v1 = {}
            w1 = {}
            for l_idx, c in u_sub.items():
                # A11 (0,0) and A22 (1,1)
                u1[map_index(l_idx, 0, 0, n2, m2, m)] = c
                u1[map_index(l_idx, 1, 1, n2, m2, m)] = u1.get(map_index(l_idx, 1, 1, n2, m2, m), 0) + c
            for l_idx, c in v_sub.items():
                # B11 (0,0) and B22 (1,1)
                # B is m x p. Block (r, c) where r is row block, c is col block.
                # B11 is r=0, c=0. B22 is r=1, c=1.
                # map_index for B: local_idx -> global.
                # B local is m2 x p2.
                # i_local = l_idx // p2, j_local = l_idx % p2.
                # global_i = i_local + r*m2, global_j = j_local + c*p2.
                # global_idx = global_i * p + global_j.
                
                def map_index_B(l_idx, block_r, block_c):
                    i_local = l_idx // p2
                    j_local = l_idx % p2
                    global_i = i_local + block_r * m2
                    global_j = j_local + block_c * p2
                    return global_i * p + global_j
                
                v1[map_index_B(l_idx, 0, 0)] = v1.get(map_index_B(l_idx, 0, 0), 0) + c
                v1[map_index_B(l_idx, 1, 1)] = v1.get(map_index_B(l_idx, 1, 1), 0) + c
            for l_idx, c in w_sub.items():
                # C11 (r=0, c=0) and C22 (r=1, c=1)
                # C is n x p.
                # C11: rows 0..n2-1, cols 0..p2-1.
                # C22: rows n2..n-1, cols p2..p-1.
                w1[map_index_C(l_idx, 0, 0, n2, p2, n)] = w1.get(map_index_C(l_idx, 0, 0, n2, p2, n), 0) + c
                w1[map_index_C(l_idx, 1, 1, n2, p2, n)] = w1.get(map_index_C(l_idx, 1, 1, n2, p2, n), 0) + c
            global_terms.append((u1, v1, w1))
            
            # M2: (A21+A22)B11 -> C21(+), C22(-)
            u2 = {}
            v2 = {}
            w2 = {}
            for l_idx, c in u_sub.items():
                u2[map_index(l_idx, 1, 0, n2, m2, m)] = u2.get(map_index(l_idx, 1, 0, n2, m2, m), 0) + c
                u2[map_index(l_idx, 1, 1, n2, m2, m)] = u2.get(map_index(l_idx, 1, 1, n2, m2, m), 0) + c
            for l_idx, c in v_sub.items():
                v2[map_index_B(l_idx, 0, 0)] = v2.get(map_index_B(l_idx, 0, 0), 0) + c
            for l_idx, c in w_sub.items():
                w2[map_index_C(l_idx, 1, 0, n2, p2, n)] = w2.get(map_index_C(l_idx, 1, 0, n2, p2, n), 0) + c
                w2[map_index_C(l_idx, 1, 1, n2, p2, n)] = w2.get(map_index_C(l_idx, 1, 1, n2, p2, n), 0) - c
            global_terms.append((u2, v2, w2))
            
            # M3: A11(B12-B22) -> C12(+), C22(+)
            u3 = {}
            v3 = {}
            w3 = {}
            for l_idx, c in u_sub.items():
                u3[map_index(l_idx, 0, 0, n2, m2, m)] = c
            for l_idx, c in v_sub.items():
                v3[map_index_B(l_idx, 0, 1)] = v3.get(map_index_B(l_idx, 0, 1), 0) + c
                v3[map_index_B(l_idx, 1, 1)] = v3.get(map_index_B(l_idx, 1, 1), 0) - c
            for l_idx, c in w_sub.items():
                w3[map_index_C(l_idx, 0, 1, n2, p2, n)] = w3.get(map_index_C(l_idx, 0, 1, n2, p2, n), 0) + c
                w3[map_index_C(l_idx, 1, 1, n2, p2, n)] = w3.get(map_index_C(l_idx, 1, 1, n2, p2, n), 0) + c
            global_terms.append((u3, v3, w3))
            
            # M4: A22(B21-B11) -> C21(+)
            u4 = {}
            v4 = {}
            w4 = {}
            for l_idx, c in u_sub.items():
                u4[map_index(l_idx, 1, 1, n2, m2, m)] = c
            for l_idx, c in v_sub.items():
                v4[map_index_B(l_idx, 1, 0)] = v4.get(map_index_B(l_idx, 1, 0), 0) + c
                v4[map_index_B(l_idx, 0, 0)] = v4.get(map_index_B(l_idx, 0, 0), 0) - c
            for l_idx, c in w_sub.items():
                w4[map_index_C(l_idx, 1, 0, n2, p2, n)] = w4.get(map_index_C(l_idx, 1, 0, n2, p2, n), 0) + c
            global_terms.append((u4, v4, w4))
            
            # M5: (A11+A12)B22 -> C11(-), C12(+)
            u5 = {}
            v5 = {}
            w5 = {}
            for l_idx, c in u_sub.items():
                u5[map_index(l_idx, 0, 0, n2, m2, m)] = u5.get(map_index(l_idx, 0, 0, n2, m2, m), 0) + c
                u5[map_index(l_idx, 0, 1, n2, m2, m)] = u5.get(map_index(l_idx, 0, 1, n2, m2, m), 0) + c
            for l_idx, c in v_sub.items():
                v5[map_index_B(l_idx, 1, 1)] = v5.get(map_index_B(l_idx, 1, 1), 0) + c
            for l_idx, c in w_sub.items():
                w5[map_index_C(l_idx, 0, 0, n2, p2, n)] = w5.get(map_index_C(l_idx, 0, 0, n2, p2, n), 0) - c
                w5[map_index_C(l_idx, 0, 1, n2, p2, n)] = w5.get(map_index_C(l_idx, 0, 1, n2, p2, n), 0) + c
            global_terms.append((u5, v5, w5))
            
            # M6: (A21-A11)(B11+B12) -> C22(+)
            u6 = {}
            v6 = {}
            w6 = {}
            for l_idx, c in u_sub.items():
                u6[map_index(l_idx, 1, 0, n2, m2, m)] = u6.get(map_index(l_idx, 1, 0, n2, m2, m), 0) + c
                u6[map_index(l_idx, 0, 0, n2, m2, m)] = u6.get(map_index(l_idx, 0, 0, n2, m2, m), 0) - c
            for l_idx, c in v_sub.items():
                v6[map_index_B(l_idx, 0, 0)] = v6.get(map_index_B(l_idx, 0, 0), 0) + c
                v6[map_index_B(l_idx, 0, 1)] = v6.get(map_index_B(l_idx, 0, 1), 0) + c
            for l_idx, c in w_sub.items():
                w6[map_index_C(l_idx, 1, 1, n2, p2, n)] = w6.get(map_index_C(l_idx, 1, 1, n2, p2, n), 0) + c
            global_terms.append((u6, v6, w6))
            
            # M7: (A12-A22)(B21+B22) -> C11(+)
            u7 = {}
            v7 = {}
            w7 = {}
            for l_idx, c in u_sub.items():
                u7[map_index(l_idx, 0, 1, n2, m2, m)] = u7.get(map_index(l_idx, 0, 1, n2, m2, m), 0) + c
                u7[map_index(l_idx, 1, 1, n2, m2, m)] = u7.get(map_index(l_idx, 1, 1, n2, m2, m), 0) - c
            for l_idx, c in v_sub.items():
                v7[map_index_B(l_idx, 1, 0)] = v7.get(map_index_B(l_idx, 1, 0), 0) + c
                v7[map_index_B(l_idx, 1, 1)] = v7.get(map_index_B(l_idx, 1, 1), 0) + c
            for l_idx, c in w_sub.items():
                w7[map_index_C(l_idx, 0, 0, n2, p2, n)] = w7.get(map_index_C(l_idx, 0, 0, n2, p2, n), 0) + c
            global_terms.append((u7, v7, w7))
            
        return global_terms

    # Part 1: 4x4 x 4x4 (B cols 0-3)
    terms_1 = strassen_decomp(4, 4, 4)
    for u, v, w in terms_1:
        add_term(u, v, w)
        
    # Part 2: 4x4 x 4x3 (B cols 4-6)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = {i * 4 + j: 1.0}
                v = {j * 7 + (4 + k): 1.0}
                w = {(4 + k) * 4 + i: 1.0}
                add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
