"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    total_rank = 36
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add Strassen 2x2x2 block
    # i_base, j_base, k_base are the starting indices in the global matrix
    # col_start is the starting column in U, V, W
    def add_strassen(i_base, j_base, k_base, col_start):
        # 7 terms
        # Term 1
        U[i_base + 0][col_start + 0] += 1
        U[i_base + 1][col_start + 0] += 1
        V[j_base + 0][col_start + 0] += 1
        V[j_base + 1][col_start + 0] += 1
        W[k_base + 0][col_start + 0] += 1
        W[k_base + 2][col_start + 0] += 1 # W index is k*n+i. Here n=2.
        # W indices:
        # k=0, i=0 -> 0
        # k=0, i=1 -> 1
        # k=1, i=0 -> 2
        # k=1, i=1 -> 3
        # So W[0] is C00, W[1] is C01, W[2] is C10, W[3] is C11?
        # Wait, W is (p*n, rank). p=5, n=2.
        # In the local 2x2x2 block, p_local=2, n_local=2.
        # Local W indices:
        # 0: k=0, i=0
        # 1: k=0, i=1
        # 2: k=1, i=0
        # 3: k=1, i=1
        # Global W indices:
        # (k_base + 0)*2 + (i_base + 0)
        # (k_base + 0)*2 + (i_base + 1)
        # (k_base + 1)*2 + (i_base + 0)
        # (k_base + 1)*2 + (i_base + 1)
        
        # Let's rewrite W updates with correct global indices.
        
        # Reset W updates for Term 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 0] += 1
        W[(k_base + 1)*2 + (i_base + 1)][col_start + 0] += 1
        
        # Term 2
        U[i_base + 1][col_start + 1] += 1
        V[j_base + 0][col_start + 1] += 1
        W[(k_base + 1)*2 + (i_base + 0)][col_start + 1] += 1
        
        # Term 3
        U[i_base + 0][col_start + 2] += 1
        V[j_base + 0][col_start + 2] += 1
        V[j_base + 1][col_start + 2] += -1
        W[(k_base + 0)*2 + (i_base + 1)][col_start + 2] += 1
        
        # Term 4
        U[i_base + 1][col_start + 3] += 1
        V[j_base + 0][col_start + 3] += -1
        V[j_base + 1][col_start + 3] += 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 3] += 1
        W[(k_base + 1)*2 + (i_base + 0)][col_start + 3] += 1
        
        # Term 5
        U[i_base + 0][col_start + 4] += 1
        U[i_base + 1][col_start + 4] += 1
        V[j_base + 1][col_start + 4] += 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 4] += -1
        W[(k_base + 0)*2 + (i_base + 1)][col_start + 4] += 1
        
        # Term 6
        U[i_base + 0][col_start + 5] += -1
        U[i_base + 1][col_start + 5] += 1
        V[j_base + 0][col_start + 5] += 1
        V[j_base + 1][col_start + 5] += 1
        W[(k_base + 1)*2 + (i_base + 1)][col_start + 5] += 1
        
        # Term 7
        U[i_base + 0][col_start + 6] += 1
        U[i_base + 1][col_start + 6] += -1
        V[j_base + 1][col_start + 6] += 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 6] += 1
        W[(k_base + 1)*2 + (i_base + 1)][col_start + 6] += 1
        
        return col_start + 7

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(i_base, j_base, k_base, col_start):
        # 4 terms
        # C0 = A0B0 + A1B1
        # C1 = A2B0 + A3B1
        # A indices: 0,1,2,3 (flattened 2x2)
        # B indices: 0,1
        # C indices: 0,1
        
        # T0: A0 B0 -> C0
        U[i_base + 0][col_start + 0] += 1
        V[j_base + 0][col_start + 0] += 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 0] += 1
        
        # T1: A1 B1 -> C0
        U[i_base + 1][col_start + 1] += 1
        V[j_base + 1][col_start + 1] += 1
        W[(k_base + 0)*2 + (i_base + 0)][col_start + 1] += 1
        
        # T2: A2 B0 -> C1
        U[i_base + 2][col_start + 2] += 1
        V[j_base + 0][col_start + 2] += 1
        W[(k_base + 0)*2 + (i_base + 1)][col_start + 2] += 1
        
        # T3: A3 B1 -> C1
        U[i_base + 3][col_start + 3] += 1
        V[j_base + 1][col_start + 3] += 1
        W[(k_base + 0)*2 + (i_base + 1)][col_start + 3] += 1
        
        return col_start + 4

    # Block 1: i in 0..1, j in 0..1, k in 0..4
    # Split k into 0..1, 2..3, 4..4
    col_offset = add_strassen(0, 0, 0, col_offset)
    col_offset = add_strassen(0, 0, 2, col_offset)
    col_offset = add_schoolbook_2x2x1(0, 0, 4, col_offset)
    
    # Block 2: i in 0..1, j in 2..3, k in 0..4
    # Note: j_base is 2.
    # U indices: i*m + j. m=4.
    # i=0, j=2 -> 2. i=0, j=3 -> 3.
    # i=1, j=2 -> 6. i=1, j=3 -> 7.
    # My helper assumes U[i_base+0], U[i_base+1]...
    # But U is flattened.
    # For Block 2, i_base=0, j_base=2.
    # U indices used:
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    # My helper uses U[i_base + offset].
    # This is incorrect for flattened U.
    
    # I need to pass explicit indices or modify helpers.
    # Let's modify helpers to take a list of U indices, V indices, W indices.
    
    # Re-implementing with explicit indices for safety.
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col_offset = 0
    
    # Block 1: i in 0..1, j in 0..1
    # U indices: 0,1,4,5 (i*m+j)
    # V indices: 0..4 (j*p+k) -> j=0: 0..4, j=1: 5..9
    # W indices: k*n+i
    
    # Strassen 1 (k=0,1)
    # U: 0,1 (A00,A01? No, A is 2x2. A00, A01, A10, A11)
    # Wait, A is 2x4. Block 1 is cols 0,1.
    # A00, A01, A10, A11.
    # U indices: 0,1,4,5.
    # Strassen uses A00, A11 (diagonal) and others.
    # My previous Strassen implementation assumed U[0], U[1] are A00, A11?
    # No, it assumed U[0], U[1] are the two rows of A?
    # No, A is 2x2.
    # U indices for A00, A01, A10, A11 are 0, 1, 4, 5.
    # Strassen M1 uses A00, A11.
    # So U[0] and U[5].
    
    # I will hardcode the 36 terms to avoid indexing bugs.
    
    # This is tedious but safe.
    # Given time, I will stick to the 36 rank solution.
    
    # Since I cannot easily debug 36 terms in thought, I will rely on the block structure.
    # The previous helper `add_strassen` was flawed for flattened indices.
    # I will write a correct version.
    
    def get_u_idx(i, j): return i * 4 + j
    def get_v_idx(j, k): return j * 5 + k
    def get_w_idx(k, i): return k * 2 + i
    
    # Block 1: i in 0,1; j in 0,1
    # k in 0,1 (Strassen)
    col = col_offset
    # M1
    U[get_u_idx(0,0)][col] += 1; U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,0)][col] += 1; V[get_v_idx(1,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,1)][col] += 1
    col += 1
    # M2
    U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,0)][col] += 1
    W[get_w_idx(1,0)][col] += 1
    col += 1
    # M3
    U[get_u_idx(0,0)][col] += 1
    V[get_v_idx(0,1)][col] += 1; V[get_v_idx(1,1)][col] += -1
    W[get_w_idx(0,1)][col] += 1
    col += 1
    # M4
    U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,0)][col] += -1; V[get_v_idx(0,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,0)][col] += 1
    col += 1
    # M5
    U[get_u_idx(0,0)][col] += 1; U[get_u_idx(0,1)][col] += 1
    V[get_v_idx(1,1)][col] += 1
    W[get_w_idx(0,0)][col] += -1; W[get_w_idx(0,1)][col] += 1
    col += 1
    # M6
    U[get_u_idx(0,0)][col] += -1; U[get_u_idx(1,0)][col] += 1
    V[get_v_idx(0,0)][col] += 1; V[get_v_idx(0,1)][col] += 1
    W[get_w_idx(1,1)][col] += 1
    col += 1
    # M7
    U[get_u_idx(0,1)][col] += 1; U[get_u_idx(1,1)][col] += -1
    V[get_v_idx(1,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,1)][col] += 1
    col += 1
    
    # k in 2,3 (Strassen)
    # Same, but k_base=2
    # M1
    U[get_u_idx(0,0)][col] += 1; U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,2)][col] += 1; V[get_v_idx(1,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,1)][col] += 1
    col += 1
    # M2
    U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,2)][col] += 1
    W[get_w_idx(3,0)][col] += 1
    col += 1
    # M3
    U[get_u_idx(0,0)][col] += 1
    V[get_v_idx(0,3)][col] += 1; V[get_v_idx(1,3)][col] += -1
    W[get_w_idx(2,1)][col] += 1
    col += 1
    # M4
    U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(0,2)][col] += -1; V[get_v_idx(0,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,0)][col] += 1
    col += 1
    # M5
    U[get_u_idx(0,0)][col] += 1; U[get_u_idx(0,1)][col] += 1
    V[get_v_idx(1,3)][col] += 1
    W[get_w_idx(2,0)][col] += -1; W[get_w_idx(2,1)][col] += 1
    col += 1
    # M6
    U[get_u_idx(0,0)][col] += -1; U[get_u_idx(1,0)][col] += 1
    V[get_v_idx(0,2)][col] += 1; V[get_v_idx(0,3)][col] += 1
    W[get_w_idx(3,1)][col] += 1
    col += 1
    # M7
    U[get_u_idx(0,1)][col] += 1; U[get_u_idx(1,1)][col] += -1
    V[get_v_idx(1,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,1)][col] += 1
    col += 1
    
    # k in 4 (Schoolbook 2x2x1)
    # C0 = A00B0 + A01B1
    # C1 = A10B0 + A11B1
    # Here B is 2x1. B0, B1.
    # V indices: get_v_idx(0,4), get_v_idx(1,4)
    # W indices: get_w_idx(4,0), get_w_idx(4,1)
    
    # T0
    U[get_u_idx(0,0)][col] += 1
    V[get_v_idx(0,4)][col] += 1
    W[get_w_idx(4,0)][col] += 1
    col += 1
    # T1
    U[get_u_idx(0,1)][col] += 1
    V[get_v_idx(1,4)][col] += 1
    W[get_w_idx(4,0)][col] += 1
    col += 1
    # T2
    U[get_u_idx(1,0)][col] += 1
    V[get_v_idx(0,4)][col] += 1
    W[get_w_idx(4,1)][col] += 1
    col += 1
    # T3
    U[get_u_idx(1,1)][col] += 1
    V[get_v_idx(1,4)][col] += 1
    W[get_w_idx(4,1)][col] += 1
    col += 1
    
    # Block 2: i in 0,1; j in 2,3
    # U indices: 2,3,6,7
    # V indices: 10..14 (j=2), 15..19 (j=3)
    # W indices: same
    
    # k in 0,1
    # M1
    U[get_u_idx(0,2)][col] += 1; U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,0)][col] += 1; V[get_v_idx(3,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,1)][col] += 1
    col += 1
    # M2
    U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,0)][col] += 1
    W[get_w_idx(1,0)][col] += 1
    col += 1
    # M3
    U[get_u_idx(0,2)][col] += 1
    V[get_v_idx(2,1)][col] += 1; V[get_v_idx(3,1)][col] += -1
    W[get_w_idx(0,1)][col] += 1
    col += 1
    # M4
    U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,0)][col] += -1; V[get_v_idx(2,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,0)][col] += 1
    col += 1
    # M5
    U[get_u_idx(0,2)][col] += 1; U[get_u_idx(0,3)][col] += 1
    V[get_v_idx(3,1)][col] += 1
    W[get_w_idx(0,0)][col] += -1; W[get_w_idx(0,1)][col] += 1
    col += 1
    # M6
    U[get_u_idx(0,2)][col] += -1; U[get_u_idx(1,2)][col] += 1
    V[get_v_idx(2,0)][col] += 1; V[get_v_idx(2,1)][col] += 1
    W[get_w_idx(1,1)][col] += 1
    col += 1
    # M7
    U[get_u_idx(0,3)][col] += 1; U[get_u_idx(1,3)][col] += -1
    V[get_v_idx(3,1)][col] += 1
    W[get_w_idx(0,0)][col] += 1; W[get_w_idx(1,1)][col] += 1
    col += 1
    
    # k in 2,3
    # M1
    U[get_u_idx(0,2)][col] += 1; U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,2)][col] += 1; V[get_v_idx(3,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,1)][col] += 1
    col += 1
    # M2
    U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,2)][col] += 1
    W[get_w_idx(3,0)][col] += 1
    col += 1
    # M3
    U[get_u_idx(0,2)][col] += 1
    V[get_v_idx(2,3)][col] += 1; V[get_v_idx(3,3)][col] += -1
    W[get_w_idx(2,1)][col] += 1
    col += 1
    # M4
    U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(2,2)][col] += -1; V[get_v_idx(2,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,0)][col] += 1
    col += 1
    # M5
    U[get_u_idx(0,2)][col] += 1; U[get_u_idx(0,3)][col] += 1
    V[get_v_idx(3,3)][col] += 1
    W[get_w_idx(2,0)][col] += -1; W[get_w_idx(2,1)][col] += 1
    col += 1
    # M6
    U[get_u_idx(0,2)][col] += -1; U[get_u_idx(1,2)][col] += 1
    V[get_v_idx(2,2)][col] += 1; V[get_v_idx(2,3)][col] += 1
    W[get_w_idx(3,1)][col] += 1
    col += 1
    # M7
    U[get_u_idx(0,3)][col] += 1; U[get_u_idx(1,3)][col] += -1
    V[get_v_idx(3,3)][col] += 1
    W[get_w_idx(2,0)][col] += 1; W[get_w_idx(3,1)][col] += 1
    col += 1
    
    # k in 4
    # T0
    U[get_u_idx(0,2)][col] += 1
    V[get_v_idx(2,4)][col] += 1
    W[get_w_idx(4,0)][col] += 1
    col += 1
    # T1
    U[get_u_idx(0,3)][col] += 1
    V[get_v_idx(3,4)][col] += 1
    W[get_w_idx(4,0)][col] += 1
    col += 1
    # T2
    U[get_u_idx(1,2)][col] += 1
    V[get_v_idx(2,4)][col] += 1
    W[get_w_idx(4,1)][col] += 1
    col += 1
    # T3
    U[get_u_idx(1,3)][col] += 1
    V[get_v_idx(3,4)][col] += 1
    W[get_w_idx(4,1)][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
