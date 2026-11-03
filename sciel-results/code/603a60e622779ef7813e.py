"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy:
    - Split m=3 into m1=1 and m2=2.
    - Part 1: [2,1,8] multiplication, rank 16 (schoolbook, optimal for this shape).
    - Part 2: [2,2,8] multiplication, decomposed into 4 blocks of [2,2,2],
              each using Strassen (rank 7), total rank 28.
    - Total rank: 16 + 28 = 44.
    
    The decomposition is exact with integer entries.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # --- Part 1: 2x1x8 (j=0) ---
    # A is 2x1, B is 1x8.
    # C[i, k] = A[i, 0] * B[0, k]
    # Rank 16.
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    # --- Part 2: 2x2x8 (j=1,2) ---
    # A is 2x2 (cols 1,2), B is 2x8 (rows 1,2).
    # Split B into 4 blocks of 2 columns.
    # Use Strassen (rank 7) for each 2x2x2 block.
    
    # Precompute Strassen step templates for a 2x2x2 block
    # Local indices: A[0,0],A[0,1],A[1,0],A[1,1] -> mapped to global
    #                B[0,0],B[0,1],B[1,0],B[1,1] -> mapped to global
    #                C[0,0],C[0,1],C[1,0],C[1,1] -> mapped to global
    
    def strassen_2x2x2(k_start):
        """Return list of (U_dict, V_dict, W_dict) for Strassen on block starting at column k_start."""
        # Global index mappings:
        # A local [0,0] -> global U idx 0*m+1 = 1
        # A local [0,1] -> global U idx 0*m+2 = 2
        # A local [1,0] -> global U idx 1*m+1 = 4
        # A local [1,1] -> global U idx 1*m+2 = 5
        #
        # B local [0,0] -> global V idx 1*p+k_start = 8+k_start
        # B local [0,1] -> global V idx 1*p+(k_start+1) = 9+k_start
        # B local [1,0] -> global V idx 2*p+k_start = 16+k_start
        # B local [1,1] -> global V idx 2*p+(k_start+1) = 17+k_start
        #
        # C local [0,0] -> global W idx k_start*n+0 = 2*k_start
        # C local [0,1] -> global W idx (k_start+1)*n+0 = 2*(k_start+1)
        # C local [1,0] -> global W idx k_start*n+1 = 2*k_start+1
        # C local [1,1] -> global W idx (k_start+1)*n+1 = 2*(k_start+1)+1
        
        A00 = 1  # 0*m+1
        A01 = 2  # 0*m+2
        A10 = 4  # 1*m+1
        A11 = 5  # 1*m+2
        
        B00 = 8 + k_start       # 1*p+k_start
        B01 = 9 + k_start       # 1*p+(k_start+1)
        B10 = 16 + k_start      # 2*p+k_start
        B11 = 17 + k_start      # 2*p+(k_start+1)
        
        C00 = 2*k_start         # k_start*n+0
        C01 = 2*(k_start+1)     # (k_start+1)*n+0
        C10 = 2*k_start+1       # k_start*n+1
        C11 = 2*(k_start+1)+1   # (k_start+1)*n+1
        
        steps = []
        
        # M1 = (A00 + A11) * (B00 + B11)
        # C00 += M1, C11 += M1
        steps.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00: 1, C11: 1}))
        
        # M2 = (A10 + A11) * B00
        # C10 += M2, C11 -= M2
        steps.append(({A10: 1, A11: 1}, {B00: 1}, {C10: 1, C11: -1}))
        
        # M3 = A00 * (B01 - B11)
        # C01 += M3, C11 += M3
        steps.append(({A00: 1}, {B01: 1, B11: -1}, {C01: 1, C11: 1}))
        
        # M4 = A11 * (B10 - B00)
        # C00 += M4, C10 += M4
        steps.append(({A11: 1}, {B10: 1, B00: -1}, {C00: 1, C10: 1}))
        
        # M5 = (A00 + A01) * B11
        # C00 -= M5, C01 += M5
        steps.append(({A00: 1, A01: 1}, {B11: 1}, {C00: -1, C01: 1}))
        
        # M6 = (A10 - A00) * (B00 + B01)
        # C11 += M6
        steps.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11: 1}))
        
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 += M7
        steps.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00: 1}))
        
        return steps
    
    for block_idx in range(4):
        k_start = block_idx * 2
        steps = strassen_2x2x2(k_start)
        
        for u_dict, v_dict, w_dict in steps:
            for u_idx, val in u_dict.items():
                U[u_idx][col] += val
            for v_idx, val in v_dict.items():
                V[v_idx][col] += val
            for w_idx, val in w_dict.items():
                W[w_idx][col] += val
            col += 1

    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
