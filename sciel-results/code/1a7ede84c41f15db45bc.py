"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_entry, V_entry, W_entry) where each entry is a dict
    mapping (row, col) -> coefficient for the respective factor.
    
    U[i,j] corresponds to A[i,j], V[j,k] to B[j,k], W[k,i] to C[k,i] (trace convention).
    For 2x2: i,j,k in {0,1}
    U index: i*2+j, V index: j*2+k, W index: k*2+i
    """
    # Strassen 7 products for C = A*B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10-B11)
    # M4 = (A11)(B01+B11)
    # M5 = (A00+A10)(B11-B10)
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A00-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For tensor decomposition, each Strassen product M = A_sub * B_sub contributes
    # rank-1 terms. But A_sub and B_sub are linear combinations of A and B blocks.
    # So each M expands to multiple rank-1 terms in the full tensor.
    
    # Actually, for the 2x2x2 tensor T[i,j,k] = 1 (for all i,j,k in 0..1),
    # the Strassen decomposition gives 7 rank-1 terms where:
    # U_c[i,j] = coefficient of A[i,j] in the first factor of M_c
    # V_c[j,k] = coefficient of B[j,k] in the second factor of M_c  
    # W_c[k,i] = coefficient such that sum over c gives the correct C
    
    # Let me just directly encode the 7 terms.
    # For each Strassen product, the A-part is a 2x2 matrix of coefficients,
    # the B-part is a 2x2 matrix of coefficients, and the W-part distributes
    # the result to C entries.
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
    # U: A00=1, A11=1 -> U[0,0]=1, U[1,1]=1
    # V: B00=1, B11=1 -> V[0,0]=1, V[1,1]=1
    # W: C00 gets +1, C11 gets +1 -> W[0,0]=1, W[1,1]=1
    
    terms = []
    
    # M1: (A00+A11)(B00+B11), C00+=1, C11+=1
    terms.append({
        'U': {(0,0):1, (1,1):1},
        'V': {(0,0):1, (1,1):1},
        'W': {(0,0):1, (1,1):1}
    })
    
    # M2: (A10+A11)(B00), C10+=1
    terms.append({
        'U': {(1,0):1, (1,1):1},
        'V': {(0,0):1},
        'W': {(0,1):1}  # C[1,0] -> W[k=0, i=1]
    })
    
    # M3: (A00)(B10-B11), C01+=1
    terms.append({
        'U': {(0,0):1},
        'V': {(1,0):1, (1,1):-1},
        'W': {(1,0):1}  # C[0,1] -> W[k=1, i=0]
    })
    
    # M4: (A11)(B01+B11), C00+=1, C10+=1
    terms.append({
        'U': {(1,1):1},
        'V': {(0,1):1, (1,1):1},
        'W': {(0,0):1, (0,1):1}  # C[0,0] and C[1,0]
    })
    
    # M5: (A00+A10)(B11-B10), C00-=1, C01+=1
    terms.append({
        'U': {(0,0):1, (1,0):1},
        'V': {(1,1):1, (1,0):-1},
        'W': {(0,0):-1, (1,0):1}  # C[0,0]-=1, C[0,1]+=1
    })
    
    # M6: (A11-A10)(B00+B10), C11+=1
    terms.append({
        'U': {(1,1):1, (1,0):-1},
        'V': {(0,0):1, (1,0):1},
        'W': {(1,1):1}  # C[1,1]
    })
    
    # M7: (A00-A11)(B01+B11), C00+=1, C11-=1
    terms.append({
        'U': {(0,0):1, (1,1):-1},
        'V': {(0,1):1, (1,1):1},
        'W': {(0,0):1, (1,1):-1}  # C[0,0]+=1, C[1,1]-=1
    })
    
    return terms

def _build_4x4x4_strassen(U, V, W, rank_start, n, m, p):
    """
    Build a rank-49 decomposition for the 4x4x4 part (first 4 cols of B)
    using Strassen 2x2x2 recursively.
    Returns number of terms added.
    """
    # 4x4 matrices split into 2x2 blocks of 2x2 submatrices.
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B, each Ci is 2x2
    
    # Block row/col offsets:
    # Block (0,0): rows 0-1, cols 0-1
    # Block (0,1): rows 0-1, cols 2-3
    # Block (1,0): rows 2-3, cols 0-1
    # Block (1,1): rows 2-3, cols 2-3
    
    block_offsets = [(0,0), (0,2), (2,0), (2,2)]
    block_ids = [(0,0), (0,1), (1,0), (1,1)]
    
    # The 7 Strassen products, each a 2x2 matrix multiplication.
    # For each, we need to expand into 4 schoolbook terms for the inner 2x2 product.
    
    # Strassen steps: (A_block_coeffs, B_block_coeffs)
    # A_block_coeffs: dict from block_id -> coeff
    # B_block_coeffs: dict from block_id -> coeff
    strassen_steps = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}),            # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}),           # M3
        ({(1,1):1}, {(0,1):1, (1,1):1}),            # M4
        ({(0,0):1, (1,0):1}, {(1,1):1, (1,0):-1}),  # M5
        ({(1,1):1, (1,0):-1}, {(0,0):1, (1,0):1}),  # M6
        ({(0,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),  # M7
    ]
    
    # C block contributions from each Strassen product:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    c_contributions = [
        # For each Strassen product index, list of (c_block_id, coefficient)
        [((0,0),1), ((1,1),1)],           # M1
        [((1,0),1)],                      # M2
        [((0,1),1)],                      # M3
        [((0,0),1), ((1,0),1)],           # M4
        [((0,0),-1), ((0,1),1)],          # M5
        [((1,1),1)],                      # M6
        [((0,0),1), ((1,1),-1)],          # M7
    ]
    
    current_rank = rank_start
    terms_added = 0
    
    for s_idx, (a_coeffs, b_coeffs) in enumerate(strassen_steps):
        for c_block, c_coeff in c_contributions[s_idx]:
            # This Strassen product contributes c_coeff * (A_sub * B_sub) to C[c_block]
            # A_sub is a 2x2 matrix = sum over a_blocks of a_coeffs[ab] * A[ab]
            # B_sub is a 2x2 matrix = sum over b_blocks of b_coeffs[bb] * B[bb]
            
            # The 2x2 product A_sub * B_sub has 4 schoolbook terms.
            # For each (i_local, j_local, k_local) in 2x2x2:
            #   U[i_local, j_local] gets contribution from A_sub
            #   V[j_local, k_local] gets contribution from B_sub
            #   W[k_local, i_local] gets c_coeff
            
            # A_sub[i_local, j_local] = sum_{ab} a_coeffs[ab] * A[ab][i_local, j_local]
            #   where A[ab] is the 2x2 block at position ab
            #   Global A index: (ab_row*2 + i_local, ab_col*2 + j_local)
            
            # B_sub[j_local, k_local] = sum_{bb} b_coeffs[bb] * B[bb][j_local, k_local]
            #   Global B index: (bb_row*2 + j_local, bb_col*2 + k_local)
            
            # C contribution to C[c_block][i_local, k_local]
            #   Global C index: (c_block_row*2 + k_local, c_block_col*2 + i_local)
            #   W index: k_global * n + i_global = (c_block_row*2+k_local)*4 + (c_block_col*2+i_local)
            
            for i_loc in range(2):
                for j_loc in range(2):
                    for k_loc in range(2):
                        # U entry: sum over a_blocks
                        u_val = 0
                        for ab, ac in a_coeffs.items():
                            ab_row, ab_col = ab
                            i_global = ab_row * 2 + i_loc
                            j_global = ab_col * 2 + j_loc
                            # A[i_global, j_global] coefficient
                            u_val += ac
                        if u_val == 0:
                            continue
                        
                        # V entry: sum over b_blocks
                        v_val = 0
                        for bb, bc in b_coeffs.items():
                            bb_row, bb_col = bb
                            j_global = bb_row * 2 + j_loc
                            k_global = bb_col * 2 + k_loc
                            # B[j_global, k_global] coefficient
                            v_val += bc
                        if v_val == 0:
                            continue
                        
                        # W entry
                        cb_row, cb_col = c_block
                        k_global = cb_row * 2 + k_loc
                        i_global = cb_col * 2 + i_loc
                        w_val = c_coeff
                        
                        U[i_global * m + j_global][current_rank] = u_val
                        V[j_global * p + k_global][current_rank] = v_val
                        W[k_global * n + i_global][current_rank] = w_val
                        
                        current_rank += 1
                        terms_added += 1
    
    return terms_added

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 
    # Part 1: 4x4x4 using recursive Strassen (rank 49)
    # Part 2: 4x4x2 using schoolbook (rank 8)
    # Total: rank 57
    
    max_rank = 60
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Part 1: 4x4x4 (first 4 columns of B)
    current_rank += _build_4x4x4_strassen(U, V, W, current_rank, n, m, p)
    
    # Part 2: 4x4x2 (columns 4 and 5 of B), schoolbook
    for k in [4, 5]:
        for i in range(n):
            for j in range(m):
                if current_rank >= max_rank:
                    break
                U[i * m + j][current_rank] = 1
                V[j * p + k][current_rank] = 1
                W[k * n + i][current_rank] = 1
                current_rank += 1
        if current_rank >= max_rank:
            break
    
    # Trim
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify the decomposition reconstructs the matrix multiplication tensor."""
    rank = len(U[0])
    
    # Target tensor: T[i,j,k] = 1 for all i,j,k
    target = np.ones((n, m, p), dtype=int)
    
    # Reconstruct
    recon = np.zeros((n, m, p), dtype=int)
    for c in range(rank):
        u_vec = np.array(U[:, c]).reshape(n, m)
        v_vec = np.array(V[:, c]).reshape(m, p)
        w_vec = np.array(W[:, c]).reshape(p, n)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    recon[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    return np.array_equal(recon, target)
# EVOLVE-BLOCK-END
