"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # We will build the matrices column by column.
    # Total rank is 88.
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeffs):
        nonlocal col
        # u_indices: list of indices in U
        # v_indices: list of indices in V
        # w_indices: list of indices in W
        # coeffs: list of coefficients corresponding to the indices
        # Note: u_indices, v_indices, w_indices must be same length
        if len(u_indices) != len(v_indices) or len(u_indices) != len(w_indices) or len(u_indices) != len(coeffs):
            raise ValueError("Mismatched lengths in add_term")
        for idx_u, idx_v, idx_w, c in zip(u_indices, v_indices, w_indices, coeffs):
            U[idx_u][col] += c
            V[idx_v][col] += c # Wait, V coeff is usually 1?
            # In the decomposition T = sum u_r v_r w_r^T,
            # U[i,r] V[j,r] W[k,r]
            # If we define the "multiplication" as (U_col . V_col) * W_col contribution?
            # No, the standard form is sum_r (u_r[i] * v_r[j] * w_r[k]).
            # So U[i,r] and V[j,r] and W[k,r] are the coefficients.
            # My add_term logic above was mixing "indices of non-zero entries" with "coefficients".
            # Let's restart the helper logic.
            pass
        col += 1

    # The previous helper was flawed because U, V, W entries are the coefficients.
    # A term is defined by:
    # u_vec: sparse vector of length n*m
    # v_vec: sparse vector of length m*p
    # w_vec: sparse vector of length p*n
    # We add these vectors as columns col.
    
    def add_term_vec(u_sparse, v_sparse, w_sparse):
        nonlocal col
        for idx, val in u_sparse.items():
            U[idx][col] = val
        for idx, val in v_sparse.items():
            V[idx][col] = val
        for idx, val in w_sparse.items():
            W[idx][col] = val
        col += 1

    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # --- Block 1: Top 2 rows (i=0,1) ---
    # A_top (2x4) = [A_L (2x2) | A_R (2x2)]
    # B (4x8) = [B_L (2x8); B_R (2x8)]
    # C_top = A_L B_L + A_R B_R
    
    # Process A_L B_L
    # A_L indices:
    # A[0,0] -> u_idx(0,0)=0
    # A[0,1] -> u_idx(0,1)=1
    # A[1,0] -> u_idx(1,0)=4
    # A[1,1] -> u_idx(1,1)=5
    
    # B_L is B[0:2, :]. Split into 4 blocks of 2x2.
    for b_block in range(4):
        # B_L block columns are 2*b_block, 2*b_block+1
        # B[0, 2*b_block] -> v_idx(0, 2*b_block)
        # B[0, 2*b_block+1] -> v_idx(0, 2*b_block+1)
        # B[1, 2*b_block] -> v_idx(1, 2*b_block)
        # B[1, 2*b_block+1] -> v_idx(1, 2*b_block+1)
        
        # C entries for this block:
        # C[0, 2*b_block] -> w_idx(2*b_block, 0)
        # C[0, 2*b_block+1] -> w_idx(2*b_block+1, 0)
        # C[1, 2*b_block] -> w_idx(2*b_block, 1)
        # C[1, 2*b_block+1] -> w_idx(2*b_block+1, 1)
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11)
        add_term_vec(
            {u_idx(0,0): 1, u_idx(1,1): 1},
            {v_idx(0, 2*b_block): 1, v_idx(1, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): 1, w_idx(2*b_block+1, 1): 1}
        )
        
        # M2 = (A10+A11)B00
        add_term_vec(
            {u_idx(1,0): 1, u_idx(1,1): 1},
            {v_idx(0, 2*b_block): 1},
            {w_idx(2*b_block, 1): 1, w_idx(2*b_block+1, 1): -1}
        )
        
        # M3 = A00(B10-B00)
        add_term_vec(
            {u_idx(0,0): 1},
            {v_idx(1, 2*b_block): 1, v_idx(0, 2*b_block): -1},
            {w_idx(2*b_block+1, 0): 1, w_idx(2*b_block+1, 1): 1}
        )
        
        # M4 = A11(B01-B11)
        add_term_vec(
            {u_idx(1,1): 1},
            {v_idx(0, 2*b_block+1): 1, v_idx(1, 2*b_block+1): -1},
            {w_idx(2*b_block, 0): 1, w_idx(2*b_block, 1): 1}
        )
        
        # M5 = (A00+A01)B11
        add_term_vec(
            {u_idx(0,0): 1, u_idx(0,1): 1},
            {v_idx(1, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): -1, w_idx(2*b_block+1, 0): 1}
        )
        
        # M6 = (A10-A00)(B00+B01)
        add_term_vec(
            {u_idx(1,0): 1, u_idx(0,0): -1},
            {v_idx(0, 2*b_block): 1, v_idx(0, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 1): 1}
        )
        
        # M7 = (A01-A11)(B10+B11)
        add_term_vec(
            {u_idx(0,1): 1, u_idx(1,1): -1},
            {v_idx(1, 2*b_block): 1, v_idx(1, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): 1}
        )

    # Process A_R B_R
    # A_R indices:
    # A[0,2] -> u_idx(0,2)=2
    # A[0,3] -> u_idx(0,3)=3
    # A[1,2] -> u_idx(1,2)=6
    # A[1,3] -> u_idx(1,3)=7
    
    # B_R is B[2:4, :]. Split into 4 blocks of 2x2.
    for b_block in range(4):
        # B[2, 2*b_block] -> v_idx(2, 2*b_block)
        # B[2, 2*b_block+1] -> v_idx(2, 2*b_block+1)
        # B[3, 2*b_block] -> v_idx(3, 2*b_block)
        # B[3, 2*b_block+1] -> v_idx(3, 2*b_block+1)
        
        # C entries same as before (rows 0,1)
        
        # Strassen 7 terms
        # M1 = (A00+A11)(B00+B11) -> (A[0,2]+A[1,3])(B[2,2b]+B[3,2b+1])
        add_term_vec(
            {u_idx(0,2): 1, u_idx(1,3): 1},
            {v_idx(2, 2*b_block): 1, v_idx(3, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): 1, w_idx(2*b_block+1, 1): 1}
        )
        
        # M2 = (A10+A11)B00 -> (A[1,2]+A[1,3])B[2,2b]
        add_term_vec(
            {u_idx(1,2): 1, u_idx(1,3): 1},
            {v_idx(2, 2*b_block): 1},
            {w_idx(2*b_block, 1): 1, w_idx(2*b_block+1, 1): -1}
        )
        
        # M3 = A00(B10-B00) -> A[0,2](B[3,2b]-B[2,2b])
        add_term_vec(
            {u_idx(0,2): 1},
            {v_idx(3, 2*b_block): 1, v_idx(2, 2*b_block): -1},
            {w_idx(2*b_block+1, 0): 1, w_idx(2*b_block+1, 1): 1}
        )
        
        # M4 = A11(B01-B11) -> A[1,3](B[2,2b+1]-B[3,2b+1])
        add_term_vec(
            {u_idx(1,3): 1},
            {v_idx(2, 2*b_block+1): 1, v_idx(3, 2*b_block+1): -1},
            {w_idx(2*b_block, 0): 1, w_idx(2*b_block, 1): 1}
        )
        
        # M5 = (A00+A01)B11 -> (A[0,2]+A[0,3])B[3,2b+1]
        add_term_vec(
            {u_idx(0,2): 1, u_idx(0,3): 1},
            {v_idx(3, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): -1, w_idx(2*b_block+1, 0): 1}
        )
        
        # M6 = (A10-A00)(B00+B01) -> (A[1,2]-A[0,2])(B[2,2b]+B[2,2b+1])
        add_term_vec(
            {u_idx(1,2): 1, u_idx(0,2): -1},
            {v_idx(2, 2*b_block): 1, v_idx(2, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 1): 1}
        )
        
        # M7 = (A01-A11)(B10+B11) -> (A[0,3]-A[1,3])(B[3,2b]+B[3,2b+1])
        add_term_vec(
            {u_idx(0,3): 1, u_idx(1,3): -1},
            {v_idx(3, 2*b_block): 1, v_idx(3, 2*b_block+1): 1},
            {w_idx(2*b_block, 0): 1}
        )

    # --- Block 2: Bottom 1 row (i=2) ---
    # A_bot (1x4) = [A_bot_L (1x2) | A_bot_R (1x2)]
    # B (4x8) = [B_L (2x8); B_R (2x8)]
    # C_bot = A_bot_L B_L + A_bot_R B_R
    
    # Process A_bot_L B_L
    # A[2,0] -> u_idx(2,0)=8
    # A[2,1] -> u_idx(2,1)=9
    
    for b_block in range(4):
        # B[0, 2*b_block] -> v_idx(0, 2*b_block)
        # B[1, 2*b_block] -> v_idx(1, 2*b_block)
        # B[0, 2*b_block+1] -> v_idx(0, 2*b_block+1)
        # B[1, 2*b_block+1] -> v_idx(1, 2*b_block+1)
        
        # C[2, 2*b_block] -> w_idx(2*b_block, 2)
        # C[2, 2*b_block+1] -> w_idx(2*b_block+1, 2)
        
        # Term 1: A[2,0] * B[0, 2*b_block] -> C[2, 2*b_block]
        add_term_vec(
            {u_idx(2,0): 1},
            {v_idx(0, 2*b_block): 1},
            {w_idx(2*b_block, 2): 1}
        )
        # Term 2: A[2,1] * B[1, 2*b_block] -> C[2, 2*b_block]
        add_term_vec(
            {u_idx(2,1): 1},
            {v_idx(1, 2*b_block): 1},
            {w_idx(2*b_block, 2): 1}
        )
        # Term 3: A[2,0] * B[0, 2*b_block+1] -> C[2, 2*b_block+1]
        add_term_vec(
            {u_idx(2,0): 1},
            {v_idx(0, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 2): 1}
        )
        # Term 4: A[2,1] * B[1, 2*b_block+1] -> C[2, 2*b_block+1]
        add_term_vec(
            {u_idx(2,1): 1},
            {v_idx(1, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 2): 1}
        )

    # Process A_bot_R B_R
    # A[2,2] -> u_idx(2,2)=10
    # A[2,3] -> u_idx(2,3)=11
    
    for b_block in range(4):
        # B[2, 2*b_block] -> v_idx(2, 2*b_block)
        # B[3, 2*b_block] -> v_idx(3, 2*b_block)
        # B[2, 2*b_block+1] -> v_idx(2, 2*b_block+1)
        # B[3, 2*b_block+1] -> v_idx(3, 2*b_block+1)
        
        # C[2, 2*b_block] -> w_idx(2*b_block, 2)
        # C[2, 2*b_block+1] -> w_idx(2*b_block+1, 2)
        
        # Term 1: A[2,2] * B[2, 2*b_block] -> C[2, 2*b_block]
        add_term_vec(
            {u_idx(2,2): 1},
            {v_idx(2, 2*b_block): 1},
            {w_idx(2*b_block, 2): 1}
        )
        # Term 2: A[2,3] * B[3, 2*b_block] -> C[2, 2*b_block]
        add_term_vec(
            {u_idx(2,3): 1},
            {v_idx(3, 2*b_block): 1},
            {w_idx(2*b_block, 2): 1}
        )
        # Term 3: A[2,2] * B[2, 2*b_block+1] -> C[2, 2*b_block+1]
        add_term_vec(
            {u_idx(2,2): 1},
            {v_idx(2, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 2): 1}
        )
        # Term 4: A[2,3] * B[3, 2*b_block+1] -> C[2, 2*b_block+1]
        add_term_vec(
            {u_idx(2,3): 1},
            {v_idx(3, 2*b_block+1): 1},
            {w_idx(2*b_block+1, 2): 1}
        )

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
