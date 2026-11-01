"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a rank-18 decomposition for 2x2x5 matrix multiplication.
    Uses Strassen (rank 7) for two 2x2 blocks and schoolbook (rank 4) for the 2x1 block.
    """
    n, m, p = DIMENSIONS  # 2, 2, 5
    rank = 18
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Helper to add a block decomposition ---
    def add_block(offset, U_block, V_block, W_block, v_row_map, w_row_map):
        """
        Adds a block decomposition to the global matrices.
        U_block: (4, r)
        V_block: (4, r) - corresponds to local B indices
        W_block: (4, r) - corresponds to local C indices
        v_row_map: list of 4 global row indices for V
        w_row_map: list of 4 global row indices for W
        """
        r = U_block.shape[1]
        for c in range(r):
            # U is shared, so we just add (or assign if disjoint, but A is shared)
            # Since A is the same for all blocks, the U factors from different blocks
            # are combined by summation? NO.
            # The decomposition is a sum of rank-1 tensors.
            # T = sum_r U_r V_r W_r.
            # If we use Strassen for block 1, we add 7 terms.
            # If we use Strassen for block 2, we add 7 terms.
            # The U part for block 1 is U1_r.
            # The U part for block 2 is U2_r.
            # These are added to the global U at the same columns?
            # No, each block contributes distinct rank-1 components.
            # So we assign them to distinct columns in U, V, W.
            
            # U indices are the same for all blocks (i, j).
            # So U[:, offset+c] = U_block[:, c]
            
            # V indices depend on k.
            # V_block has rows corresponding to local j, k_local.
            # We map local rows to global rows.
            
            # W indices depend on k.
            
            U[:, offset + c] = U_block[:, c]
            
            # Map V
            for local_idx, global_idx in enumerate(v_row_map):
                V[global_idx, offset + c] = V_block[local_idx, c]
                
            # Map W
            for local_idx, global_idx in enumerate(w_row_map):
                W[global_idx, offset + c] = W_block[local_idx, c]

    # --- Strassen Decomposition for 2x2x2 ---
    # Local indices:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # Strassen factors
    # M1 = (A00+A11)(B00+B10) -> C00 += M1, C11 -= M1? No.
    # Standard Strassen:
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 - M4 + M6 + M7
    # C01 = M3 + M4 + M5
    # C10 = M2 + M4 - M7
    # C11 = M1 - M2 + M3 + M5
    
    # We need U, V, W such that sum U V W = T.
    # For each M, we have a rank-1 term? No, M is a product of linear forms.
    # M1 = (a+d)(e+g).
    # Term 1: U=(1,0,0,1), V=(1,0,1,0), W=(1,1,0,0) ?
    # Let's check:
    # sum_j U_j V_j W_k ? No.
    # The tensor entry is U_{ij} V_{jk} W_{ki}.
    # For M1:
    # U part: a+d -> (1,0,0,1)
    # V part: e+g -> (1,0,1,0) (B00, B10)
    # W part: contributes to C00 and C11?
    # C00 gets M1. C11 gets M1.
    # So W part must be (1, 0, 0, 1) for C00, C11?
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
    # W = (1, 0, 0, 1).
    # Check term:
    # sum_{i,j,k} U_{ij} V_{jk} W_{ki} A_{ij} B_{jk}
    # = (A00+A11)(B00+B10)(C00_coeff + C11_coeff)
    # This produces (A00+A11)(B00+B10) * (C00 + C11).
    # We want C00 += M1 and C11 += M1.
    # So yes, this works.
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # Col 0: M1
    U_strassen[:, 0] = [1, 0, 0, 1]
    V_strassen[:, 0] = [1, 0, 1, 0]
    W_strassen[:, 0] = [1, 0, 0, 1]
    
    # Col 1: -M4
    # M4 = d(h-f)
    # U: d -> (0,0,0,1)
    # V: h-f -> (0,1,0,-1) (B01, B11)
    # W: -M4 in C00, +M4 in C01?
    # C00 = ... - M4
    # C01 = ... + M4
    # C10 = ... + M4
    # C11 = ...
    # Wait, C11 = M1 - M2 + M3 + M5. M4 is not in C11.
    # C10 = M2 + M4 - M7.
    # So M4 is in C00 (-), C01 (+), C10 (+).
    # W = [-1, 1, 1, 0]
    U_strassen[:, 1] = [0, 0, 0, 1]
    V_strassen[:, 1] = [0, 1, 0, -1]
    W_strassen[:, 1] = [-1, 1, 1, 0]
    
    # Col 2: M6
    # M6 = (c-a)(e+f)
    # U: c-a -> (0,-1,1,0)
    # V: e+f -> (1,1,0,0)
    # W: M6 in C00 (+)
    # C00 = M1 - M4 + M6 + M7
    # W = [1, 0, 0, 0]
    U_strassen[:, 2] = [0, -1, 1, 0]
    V_strassen[:, 2] = [1, 1, 0, 0]
    W_strassen[:, 2] = [1, 0, 0, 0]
    
    # Col 3: M7
    # M7 = (b-d)(g+h)
    # U: b-d -> (0,1,0,-1)
    # V: g+h -> (0,0,1,1)
    # W: M7 in C00 (+), C10 (-)
    # W = [1, 0, -1, 0]
    U_strassen[:, 3] = [0, 1, 0, -1]
    V_strassen[:, 3] = [0, 0, 1, 1]
    W_strassen[:, 3] = [1, 0, -1, 0]
    
    # Col 4: M3
    # M3 = a(f-g)
    # U: a -> (1,0,0,0)
    # V: f-g -> (0,1,-1,0)
    # W: M3 in C01 (+), C11 (+)
    # W = [0, 1, 0, 1]
    U_strassen[:, 4] = [1, 0, 0, 0]
    V_strassen[:, 4] = [0, 1, -1, 0]
    W_strassen[:, 4] = [0, 1, 0, 1]
    
    # Col 5: M5
    # M5 = (a+b)h
    # U: a+b -> (1,1,0,0)
    # V: h -> (0,0,0,1)
    # W: M5 in C01 (+), C11 (+)
    # W = [0, 1, 0, 1]
    U_strassen[:, 5] = [1, 1, 0, 0]
    V_strassen[:, 5] = [0, 0, 0, 1]
    W_strassen[:, 5] = [0, 1, 0, 1]
    
    # Col 6: M2
    # M2 = (c+d)e
    # U: c+d -> (0,0,1,1)
    # V: e -> (1,0,0,0)
    # W: M2 in C10 (+), C11 (-)
    # W = [0, 0, 1, -1]
    U_strassen[:, 6] = [0, 0, 1, 1]
    V_strassen[:, 6] = [1, 0, 0, 0]
    W_strassen[:, 6] = [0, 0, 1, -1]

    # --- Block 1: k=0,1 ---
    # V global rows:
    # j=0, k=0 -> 0*5+0 = 0
    # j=0, k=1 -> 0*5+1 = 1
    # j=1, k=0 -> 1*5+0 = 5
    # j=1, k=1 -> 1*5+1 = 6
    v_map_1 = [0, 1, 5, 6]
    
    # W global rows:
    # k=0, i=0 -> 0*2+0 = 0
    # k=0, i=1 -> 0*2+1 = 1
    # k=1, i=0 -> 1*2+0 = 2
    # k=1, i=1 -> 1*2+1 = 3
    w_map_1 = [0, 1, 2, 3]
    
    add_block(0, U_strassen, V_strassen, W_strassen, v_map_1, w_map_1)

    # --- Block 2: k=2,3 ---
    # Same Strassen factors.
    # V global rows:
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 7
    # j=1, k=3 -> 8
    v_map_2 = [2, 3, 7, 8]
    
    # W global rows:
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    w_map_2 = [4, 5, 6, 7]
    
    add_block(7, U_strassen, V_strassen, W_strassen, v_map_2, w_map_2)

    # --- Block 3: k=4 ---
    # Schoolbook 2x2x1
    # C_i4 = A_i0 B_04 + A_i1 B_14
    # Terms:
    # 1. A00 B04 -> C04
    # 2. A01 B14 -> C04
    # 3. A10 B04 -> C14
    # 4. A11 B14 -> C14
    
    U_school = np.zeros((4, 4))
    V_school = np.zeros((2, 4)) # Only 2 rows for B (j=0,1)
    W_school = np.zeros((2, 4)) # Only 2 rows for C (i=0,1)
    
    # Col 0
    U_school[:, 0] = [1, 0, 0, 0]
    V_school[:, 0] = [1, 0]
    W_school[:, 0] = [1, 0]
    
    # Col 1
    U_school[:, 1] = [0, 1, 0, 0]
    V_school[:, 1] = [0, 1]
    W_school[:, 1] = [1, 0]
    
    # Col 2
    U_school[:, 2] = [0, 0, 1, 0]
    V_school[:, 2] = [1, 0]
    W_school[:, 2] = [0, 1]
    
    # Col 3
    U_school[:, 3] = [0, 0, 0, 1]
    V_school[:, 3] = [0, 1]
    W_school[:, 3] = [0, 1]
    
    # V global rows for k=4:
    # j=0, k=4 -> 4
    # j=1, k=4 -> 9
    v_map_3 = [4, 9]
    
    # W global rows for k=4:
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    w_map_3 = [8, 9]
    
    # Note: U_school is 4x4, V_school is 2x4, W_school is 2x4.
    # add_block expects V_block and W_block to have 4 rows?
    # My add_block maps 4 local rows.
    # For schoolbook, local V rows are just 2 (j=0,1).
    # I can pad V_school and W_school to 4 rows with zeros.
    
    V_school_padded = np.zeros((4, 4))
    V_school_padded[0:2, :] = V_school
    W_school_padded = np.zeros((4, 4))
    W_school_padded[0:2, :] = W_school
    
    add_block(14, U_school, V_school_padded, W_school_padded, v_map_3, w_map_3)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
