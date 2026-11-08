"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Build the rank-42 block-Strassen solution
    U, V, W = _build_strassen_solution(n, m, p)
    rank = len(U[0])
    
    # Verify the solution
    if not _verify_solution(U, V, W, n, m, p):
        # Fallback to trivial rank n*m*p
        U, V, W = _build_trivial_solution(n, m, p)
        
    return {"U": U, "V": V, "W": W}

def _build_strassen_solution(n, m, p):
    """Build the standard block-Strassen solution with rank 42."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U coefficients: (i, j) -> coeff
    # V coefficients: (j, k) -> coeff
    # W coefficients: (i, k) -> coeff
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]

    col = 0
    # We have 2 blocks of A (2x2) and 3 blocks of B (2x2)
    # Total 6 combinations, each using 7 multiplications
    for a in range(2):
        for b in range(3):
            for s in range(7):
                # U: A block a
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V: B block b
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc # Wait, j in V must match j in U for the same block?
                    # In the decomposition C = A_L B_L + A_L B_M + A_L B_R + A_R B_L + ...
                    # The middle index j is shared.
                    # For block a (A) and block b (B), the j index is fixed to the block's j range.
                    # A block a covers j in [2a, 2a+1].
                    # B block b covers j in [2b, 2b+1]? No.
                    # B is 4x6. Rows are j=0..3.
                    # We split B rows into two groups: top (j=0,1) and bottom (j=2,3).
                    # So B block index should correspond to the row group.
                    # But we have 3 B blocks horizontally?
                    # B = [B_top, B_bot] where B_top is 2x6, B_bot is 2x6.
                    # B_top = [B_T0, B_T1, B_T2] where each is 2x2.
                    # B_bot = [B_B0, B_B1, B_B2] where each is 2x2.
                    # A = [A_L, A_R] where A_L is 2x2, A_R is 2x2.
                    # C = A_L B_top + A_R B_bot.
                    # A_L B_top = A_L B_T0 + A_L B_T1 + A_L B_T2.
                    # Each term is 2x2 * 2x2.
                    # So for a=0 (A_L), b=0,1,2 (B_T0, B_T1, B_T2):
                    # j is always 0,1.
                    # For a=1 (A_R), b=0,1,2 (B_B0, B_B1, B_B2):
                    # j is always 2,3.
                    
                    # So j_loc in V_coeffs is relative to the block.
                    # j = a * 2 + j_loc is correct for U.
                    # For V, j should also be a * 2 + j_loc?
                    # Yes, because the multiplication is A_block * B_block.
                    # The j index in A_block and B_block must match.
                    # A_block a has rows i=0,1 and cols j=2a, 2a+1.
                    # B_block (row group a) has rows j=2a, 2a+1 and cols k=2b, 2b+1.
                    # So j in V is 2a + j_loc.
                    
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # W: C block
                # C is 2x6.
                # C[i, k] = sum ...
                # For A_block a and B_block b (row group a, col group b):
                # i is 0,1.
                # k is 2b, 2b+1.
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1

    return U, V, W

def _build_trivial_solution(n, m, p):
    """Build the trivial rank n*m*p solution."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
                
    return U, V, W

def _verify_solution(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
