import time
import sys

def construct():
    n, m, p = 4, 4, 9
    # We will build the decomposition for 4x4x9
    # Using:
    # 1. Strassen (recursively to 1x1) for cols 0-3 (Rank 49)
    # 2. Strassen (recursively to 1x1) for cols 4-7 (Rank 49)
    # 3. Schoolbook for col 8 (Rank 16)
    # Total Rank: 114
    
    def get_strassen_terms(size):
        """
        Returns a list of terms for size x size x size matrix multiplication.
        Each term is a tuple (U_dict, V_dict, W_dict).
        Keys are (i, j) or (k, i) etc? 
        Let's use flat indices for the local size x size matrices.
        U index: i * size + j
        V index: j * size + k
        W index: k * size + i
        """
        if size == 1:
            return [({0: 1}, {0: 1}, {0: 1})]
        
        half = size // 2
        # Get 7 sets of linear combinations
        # A = [[A00, A01], [A10, A11]]
        # B = [[B00, B01], [B10, B11]]
        
        # The 7 Strassen products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map the linear combinations to the local indices.
        # Local indices for size x size:
        # A[i, j] is at i*size + j.
        
        def block_indices(block_type):
            # block_type: 00, 01, 10, 11
            if block_type == 0: # 00
                return [(i, j) for i in range(half) for j in range(half)]
            elif block_type == 1: # 01
                return [(i, j+half) for i in range(half) for j in range(half)]
            elif block_type == 2: # 10
                return [(i+half, j) for i in range(half) for j in range(half)]
            elif block_type == 3: # 11
                return [(i+half, j+half) for i in range(half) for j in range(half)]

        # Define the 7 cases as (U_combos, V_combos, W_block)
        # U_combos: list of (block, coeff)
        # V_combos: list of (block, coeff)
        # W_block: 0,1,2,3 (which output block the result goes into, with sign)
        
        # M1: (A00+A11) (B00+B11) -> C00 (+1), C11 (+1) ... wait
        # C00 = M1 + M4 - M5 - M7
        # C01 = M5 + M6
        # C10 = M2 + M3
        # C11 = M1 - M2 + M3 + M4
        
        cases = [
            # (U_blocks, V_blocks, [(w_block, coeff)])
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, 1)]), # M2
            ([(0, 1)], [(2, 1), (3, -1)], [(2, 1), (3, 1)]), # M3
            ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (3, 1)]), # M4
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (2, 1)]), # M5
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1), (2, 1)]), # M6
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1), (3, -1)]), # M7
        ]
        
        terms = []
        # Recursively get terms for half size
        sub_terms = get_strassen_terms(half)
        
        for u_blocks, v_blocks, w_blocks in cases:
            for u_term, v_term, w_term in sub_terms:
                # Create new U, V, W dicts
                new_u = {}
                new_v = {}
                new_w = {}
                
                # Apply U linear combinations
                for (b_idx, coeff) in u_blocks:
                    for (i, j), val in u_term.items():
                        # Map local (i,j) to global (i+j*half) based on block
                        if b_idx == 0: # 00
                            gi, gj = i, j
                        elif b_idx == 1: # 01
                            gi, gj = i, j+half
                        elif b_idx == 2: # 10
                            gi, gj = i+half, j
                        elif b_idx == 3: # 11
                            gi, gj = i+half, j+half
                        
                        g_idx = gi * size + gj
                        new_u[g_idx] = new_u.get(g_idx, 0) + val * coeff
                        
                # Apply V linear combinations
                for (b_idx, coeff) in v_blocks:
                    for (j, k), val in v_term.items():
                        if b_idx == 0: # 00
                            gj, gk = j, k
                        elif b_idx == 1: # 01
                            gj, gk = j, k+half
                        elif b_idx == 2: # 10
                            gj, gk = j+half, k
                        elif b_idx == 3: # 11
                            gj, gk = j+half, k+half
                        
                        g_idx = gj * size + gk
                        new_v[g_idx] = new_v.get(g_idx, 0) + val * coeff
                        
                # Apply W linear combinations
                for (b_idx, coeff) in w_blocks:
                    for (k, i), val in w_term.items():
                        if b_idx == 0: # 00
                            gk, gi = k, i
                        elif b_idx == 1: # 01
                            gk, gi = k+half, i
                        elif b_idx == 2: # 10
                            gk, gi = k, i+half
                        elif b_idx == 3: # 11
                            gk, gi = k+half, i+half
                        
                        g_idx = gk * size + gi
                        new_w[g_idx] = new_w.get(g_idx, 0) + val * coeff

                # Prune zeros
                new_u = {k: v for k, v in new_u.items() if v != 0}
                new_v = {k: v for k, v in new_v.items() if v != 0}
                new_w = {k: v for k, v in new_w.items() if v != 0}
                
                if new_u and new_v and new_w:
                    terms.append((new_u, new_v, new_w))
                    
        return terms

    # Get 49 terms for 4x4x4
    terms_4x4 = get_strassen_terms(4)
    
    # Get 16 terms for 4x4x1 (Schoolbook)
    def get_schoolbook_4x1():
        terms = []
        for i in range(4):
            for j in range(4):
                u = {i*4+j: 1}
                v = {j*9+8: 1} # k=8
                w = {8*4+i: 1} # k=8
                terms.append((u, v, w))
        return terms
        
    terms_4x1 = get_schoolbook_4x1()
    
    # Assemble Final Matrices
    rank = len(terms_4x4) * 2 + len(terms_4x1) # 49 + 49 + 16 = 114
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    col_offset = 0
    # Block 1: Cols 0-3
    for u_term, v_term, w_term in terms_4x4:
        for idx, val in u_term.items():
            U[idx][col_offset] = val
        for idx, val in v_term.items():
            # v_idx is j*4+k. Global V is j*9+k. k is 0-3.
            j = idx // 4
            k = idx % 4
            g_idx = j * 9 + k
            V[g_idx][col_offset] = val
        for idx, val in w_term.items():
            # w_idx is k*4+i. Global W is k*4+i. k is 0-3.
            k = idx // 4
            i = idx % 4
            g_idx = k * 4 + i
            W[g_idx][col_offset] = val
        col_offset += 1
        
    # Block 2: Cols 4-7
    for u_term, v_term, w_term in terms_4x4:
        for idx, val in u_term.items():
            U[idx][col_offset] = val
        for idx, val in v_term.items():
            j = idx // 4
            k = idx % 4
            g_k = k + 4
            g_idx = j * 9 + g_k
            V[g_idx][col_offset] = val
        for idx, val in w_term.items():
            k = idx // 4
            i = idx % 4
            g_k = k + 4
            g_idx = g_k * 4 + i
            W[g_idx][col_offset] = val
        col_offset += 1
        
    # Block 3: Col 8
    for u_term, v_term, w_term in terms_4x1:
        for idx, val in u_term.items():
            U[idx][col_offset] = val
        for idx, val in v_term.items():
            V[idx][col_offset] = val
        for idx, val in w_term.items():
            W[idx][col_offset] = val
        col_offset += 1
        
    return {"U": U, "V": V, "W": W}

