    def construct():
        n, m, p = 6, 6, 6
        rank = 189
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        # Define the 7 Strassen terms
        # Each term is a dict:
        # 'A': list of (block_id, coeff)
        # 'B': list of (block_id, coeff)
        # 'C': list of (block_id, coeff)
        
        terms = [
            { # M1
                'A': [(0, 1), (3, 1)], # 11, 22
                'B': [(0, 1), (3, 1)], # 11, 22
                'C': [(0, 1), (3, 1)]  # 11, 22
            },
            { # M2
                'A': [(2, 1), (3, 1)], # 21, 22
                'B': [(0, 1)],         # 11
                'C': [(2, 1)]          # 21
            },
            { # M3
                'A': [(0, 1)],         # 11
                'B': [(1, 1), (3, -1)],# 12, -22
                'C': [(1, 1), (3, -1)] # 12, -22
            },
            { # M4
                'A': [(3, 1)],         # 22
                'B': [(2, 1), (0, -1)],# 21, -11
                'C': [(0, 1), (2, 1)]  # 11, 21
            },
            { # M5
                'A': [(0, 1), (1, 1)], # 11, 12
                'B': [(3, 1)],         # 22
                'C': [(0, -1), (1, 1)] # -11, 12
            },
            { # M6
                'A': [(2, 1), (0, -1)],# 21, -11
                'B': [(0, 1), (1, 1)], # 11, 12
                'C': [(3, 1)]          # 22
            },
            { # M7
                'A': [(1, 1), (3, -1)],# 12, -22
                'B': [(2, 1), (3, 1)], # 21, 22
                'C': [(0, 1), (3, 1)]  # 11, 22
            }
        ]
        
        col = 0
        for term in terms:
            # For each of the 27 scalar multiplications in the 3x3 block mult
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        # U part: A_block[i,k]
                        for b_id, coeff in term['A']:
                            # Block b_id is 3x3.
                            # Offset:
                            # 0 (11): row 0, col 0
                            # 1 (12): row 0, col 3
                            # 2 (21): row 3, col 0
                            # 3 (22): row 3, col 3
                            r_off = 3 * (b_id // 2)
                            c_off = 3 * (b_id % 2)
                            a_idx = (r_off + i) * 3 + (c_off + k)
                            U[a_idx][col] += coeff
                        
                        # V part: B_block[k,j]
                        for b_id, coeff in term['B']:
                            r_off = 3 * (b_id // 2)
                            c_off = 3 * (b_id % 2)
                            # B is 6x6. Flattened index: row * 6 + col?
                            # Wait, V shape is (m*p, rank) = (36, rank).
                            # B is 6x6.
                            # B_block is 3x3.
                            # b_id 0 (11): rows 0-2, cols 0-2.
                            # b_id 1 (12): rows 0-2, cols 3-5.
                            # b_id 2 (21): rows 3-5, cols 0-2.
                            # b_id 3 (22): rows 3-5, cols 3-5.
                            b_row = r_off + k
                            b_col = c_off + j
                            b_idx = b_row * 6 + b_col
                            V[b_idx][col] += coeff
                        
                        # W part: C_block[i,j]
                        for b_id, coeff in term['C']:
                            r_off = 3 * (b_id // 2)
                            c_off = 3 * (b_id % 2)
                            c_row = r_off + i
                            c_col = c_off + j
                            # W shape (p*n, rank) = (36, rank).
                            # W is indexed by k*n+i,r?
                            # Task says: W[k*n+i,r].
                            # This usually means W is flattened column-major? Or just index mapping.
                            # "W[k*n+i,r]" where k is col? i is row?
                            # Standard C[i,j].
                            # If W is flattened C, index = row * n + col?
                            # Let's assume standard row-major flattening for C as well,
                            # unless "trace convention" implies something else.
                            # "Output factor follows the official trace convention W[k*n+i,r]"
                            # This phrasing is tricky.
                            # Usually, tensor decomposition for matmul:
                            # T_{i,j,k} = sum U_{i,k'} V_{j,k'} W_{k,k'}?
                            # Here indices are:
                            # U: (n*m, rank) -> A index, rank
                            # V: (m*p, rank) -> B index, rank
                            # W: (p*n, rank) -> C index, rank
                            # C is p x n? No, C is n x p?
                            # A is n x m. B is m x p. C is n x p.
                            # So C index should be n*p.
                            # But W is (p*n, rank).
                            # And the convention is W[k*n+i, r].
                            # This suggests W is indexed by (k, i) where k is col (0..p-1) and i is row (0..n-1)?
                            # If p=6, n=6.
                            # Index = k*6 + i.
                            # This is column-major flattening of C?
                            # C[i, k] is at index k*6 + i.
                            # Yes, this is column-major.
                            c_idx = c_col * 6 + c_row
                            W[c_idx][col] += coeff
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    