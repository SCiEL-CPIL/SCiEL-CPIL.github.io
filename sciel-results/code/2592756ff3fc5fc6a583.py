import numpy as np
import time

# EVOLVE-BLOCK-START
_N, _M, _P = 4, 4, 4
_RANK = 49
_START_TIME = time.time()
_SAFETY_MARGIN = 10  # seconds before the 120s limit

def _construct_decomposition():
    """
    Build a rank-49 tensor decomposition for 4x4 matrix multiplication
    using a recursive Strassen-like approach.
    
    The decomposition follows the trace convention for W: W[k*n+i, r]
    where k is the column index in C and i is the row index.
    
    Returns:
        dict with keys 'U', 'V', 'W' as factor matrices
    """
    U = np.zeros((_N * _M, _RANK), dtype=float)
    V = np.zeros((_M * _P, _RANK), dtype=float)
    W = np.zeros((_P * _N, _RANK), dtype=float)
    
    def _idx_u(i, j):
        return i * _M + j
    
    def _idx_v(i, j):
        return i * _P + j
    
    def _idx_w(i, j):
        # Trace convention: W[k*n+i, r] where k=col, i=row
        return j * _N + i
    
    # Define the 7 Strassen combinations for 2x2 blocks
    # Each combination is defined by coefficients for the 4 sub-blocks
    # For 4x4, we split into 2x2 blocks of 2x2 matrices
    
    # Block positions: (0,0), (0,1), (1,0), (1,1)
    # Each 2x2 block has 4 elements at positions (r,c) where r,c in {0,1}
    
    # For each of the 7 Strassen terms, we need to define:
    # - Which A sub-blocks contribute and with what coefficients
    # - Which B sub-blocks contribute and with what coefficients
    # - How the product contributes to C sub-blocks
    
    # Standard Strassen for 2x2:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    
    # C00 = P1 - P2 + P4 + P6
    # C01 = P3 + P5
    # C10 = P2 + P7
    # C11 = P1 - P5 + P3 + P7
    
    # For 4x4, each 2x2 sub-block is itself a 2x2 matrix
    # So we need to expand each term into 7 more multiplications
    # giving 7*7 = 49 total rank-1 terms
    
    # Define the base Strassen patterns
    # A_patterns[term][block] = coefficient
    # B_patterns[term][block] = coefficient
    # C_contrib[term][block] = list of (block, coefficient)
    
    A_patterns = [
        # Term 0: P1
        [(0, 1), (1, 1)],   # A00 + A11
        [(1, 1), (1, 1)],   # A10 + A11
        [(0, 1), (1, 1)],   # A00 + A11... wait, let me be more careful
    ]
    
    # Let me redefine more carefully
    # For each Strassen term t (0-6), define which 2x2 sub-blocks of A and B
    # are used, with coefficients
    
    # sub_block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Similarly for B
    
    A_sub = [
        {0: 1, 3: 1},    # P1: A00 + A11
        {2: 1, 3: 1},    # P2: A10 + A11
        {0: 1},           # P3: A00
        {3: 1},           # P4: A11
        {0: 1, 1: 1},    # P5: A00 + A01
        {2: 1, 0: -1},   # P6: A10 - A00
        {1: 1, 3: -1},   # P7: A01 - A11
    ]
    
    B_sub = [
        {0: 1, 3: 1},    # P1: B00 + B11
        {0: 1},           # P2: B00
        {1: 1, 3: -1},   # P3: B01 - B11
        {2: 1, 0: -1},   # P4: B10 - B00
        {3: 1},           # P5: B11
        {0: 1, 1: 1},    # P6: B00 + B01
        {2: 1, 3: 1},    # P7: B10 + B11
    ]
    
    C_sub = [
        {0: 1, 3: 1},    # P1 contributes to C00 and C11
        {2: 1, 3: -1},   # P2 contributes to C10 and C11 (with -1)
        {1: 1, 3: 1},    # P3 contributes to C01 and C11
        {0: 1, 2: 1},    # P4 contributes to C00 and C10
        {0: -1, 1: 1},   # P5 contributes to C00 (-1) and C01
        {3: 1},          # P6 contributes to C11
        {0: 1},          # P7 contributes to C00
    ]
    
    # Now for each of the 7 outer Strassen terms, we expand the 2x2 sub-block
    # multiplications using another level of Strassen (or direct 7-term expansion)
    
    # For a 2x2 matrix multiplication C = A*B where each is 2x2,
    # we can use 7 multiplications (Strassen) or 4 (naive)
    # To get rank 49 = 7*7, we use Strassen at both levels
    
    # Inner Strassen patterns (same as above)
    # For inner level, we work with scalar elements
    
    # Element positions in a 2x2 block: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    elem_idx = lambda r, c: r * 2 + c
    
    # Inner Strassen: for multiplying two 2x2 matrices X*Y = Z
    # Using the same 7-term pattern
    X_sub_inner = A_sub  # Same pattern
    Y_sub_inner = B_sub
    Z_sub_inner = C_sub
    
    # Build the full decomposition
    for t_outer in range(7):
        a_blocks = A_sub[t_outer]
        b_blocks = B_sub[t_outer]
        c_contrib = C_sub[t_outer]
        
        for t_inner in range(7):
            rank_idx = t_outer * 7 + t_inner
            
            # For this rank-1 term, we need to specify:
            # U[row, rank_idx] = coefficient of A element
            # V[col, rank_idx] = coefficient of B element  
            # W[row, rank_idx] = coefficient for C element
            
            # The outer term multiplies combinations of A sub-blocks and B sub-blocks
            # Each sub-block is 2x2, so we need to expand the 2x2 multiplication
            
            # For the inner Strassen term t_inner, we combine elements from
            # the selected sub-blocks according to the inner pattern
            
            u_coeffs = {}  # (i, j) -> coefficient for U
            v_coeffs = {}  # (i, j) -> coefficient for V
            w_coeffs = {}  # (i, j) -> coefficient for W
            
            # For each selected A sub-block, apply the inner X pattern
            for ablk, a_coeff in a_blocks.items():
                # ablk: 0=A00, 1=A01, 2=A10, 3=A11
                br, bc = divmod(ablk, 2)  # block row, block col
                for ir in range(2):
                    for ic in range(2):
                        # This element is at global position (br*2+ir, bc*2+ic)
                        ei, ej = elem_idx(ir, ic)
                        # Apply inner X pattern
                        for inner_blk, inner_coeff in X_sub_inner[t_inner].items():
                            ir2, ic2 = divmod(inner_blk, 2)
                            if ir == ir2 and ic == ic2:
                                u_coeffs[(br*2+ir, bc*2+ic)] = u_coeffs.get((br*2+ir, bc*2+ic), 0) + a_coeff * inner_coeff
            
            # Similarly for B sub-blocks
            for bblk, b_coeff in b_blocks.items():
                br, bc = divmod(bblk, 2)
                for ir in range(2):
                    for ic in range(2):
                        ei, ej = elem_idx(ir, ic)
                        for inner_blk, inner_coeff in Y_sub_inner[t_inner].items():
                            ir2, ic2 = divmod(inner_blk, 2)
                            if ir == ir2 and ic == ic2:
                                v_coeffs[(br*2+ir, bc*2+ic)] = v_coeffs.get((br*2+ir, bc*2+ic), 0) + b_coeff * inner_coeff
            
            # For C contribution, the inner Z pattern tells us which elements
            # of the 2x2 result contribute
            for cblk, c_coeff in c_contrib.items():
                br, bc = divmod(cblk, 2)
                for ir in range(2):
                    for ic in range(2):
                        ei, ej = elem_idx(ir, ic)
                        for inner_blk, inner_coeff in Z_sub_inner[t_inner].items():
                            ir2, ic2 = divmod(inner_blk, 2)
                            if ir == ir2 and ic == ic2:
                                w_coeffs[(br*2+ir, bc*2+ic)] = w_coeffs.get((br*2+ir, bc*2+ic), 0) + c_coeff * inner_coeff
            
            # Fill in the matrices
            for (i, j), coeff in u_coeffs.items():
                U[_idx_u(i, j), rank_idx] = coeff
            
            for (i, j), coeff in v_coeffs.items():
                V[_idx_v(i, j), rank_idx] = coeff
            
            for (i, j), coeff in w_coeffs.items():
                W[_idx_w(i, j), rank_idx] = coeff
    
    return {"U": U, "V": V, "W": W}

# Precompute at module load time
_RESULT = _construct_decomposition()

def construct():
    """
    Return the rank-49 tensor decomposition for 4x4 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' as numpy arrays of shape (16, 49)
    """
    return _RESULT
# EVOLVE-BLOCK-END
