"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a rank-51 decomposition for 3x3x6 matrix multiplication.
    
    Strategy:
    Split the output dimension p=6 into 3 blocks of size 2.
    For each block, solve the 3x3x2 multiplication using:
    - Strassen's 7-term algorithm for the 2x2x2 core (A[0:2, 0:2] * B[0:2, :])
    - 4 terms for the cross product A[0:2, 2] * B[2, :]
    - 4 terms for the cross product A[2, 0:2] * B[0:2, :]
    - 2 terms for the corner product A[2, 2] * B[2, :]
    Total 17 terms per block * 3 blocks = 51 rank.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        if u_val != 0:
            U[u_idx][col] += u_val
        if v_val != 0:
            V[v_idx][col] += v_val
        if w_val != 0:
            W[w_idx][col] += w_val

    for b in range(3):
        # b is the block index for the output columns
        # Global column offset for V and W: k_global = 2*b + k_local
        col_base = b * 17
        
        # --- Strassen Core (7 terms) ---
        # Operates on A[0:2, 0:2] and B[0:2, 2b:2b+2]
        # A indices: 0,1,3,4 (i*3+j)
        # B indices: 6*(0)+2b, 6*(0)+2b+1, 6*(1)+2b, 6*(1)+2b+1
        # C indices (W): 3*(2b)+i, 3*(2b+1)+i
        
        # m1 = (a00+a11)(b00+b11)
        add_term(0, 1, 6*0+2*b, 1, 3*(2*b)+0, 1, col_base+0)
        add_term(4, 1, 6*1+(2*b+1), 1, 3*(2*b+1)+1, 1, col_base+0)
        
        # m2 = (a10+a11)b00
        add_term(3, 1, 6*0+2*b, 1, 3*(2*b)+1, 1, col_base+1)
        add_term(4, 1, 0, 0, 0, 0, col_base+1) # a11 is already added? No, U[4] needs 1.
        # Wait, add_term adds to specific index. I need to handle multiple U/V/W entries per term.
        # My add_term only handles ONE U, ONE V, ONE W.
        # Strassen terms are products of SUMS.
        # (a00+a11) means U[0]=1 AND U[4]=1 in the same column.
        # My add_term function above is insufficient for sums.
        
        # Let's redefine the loop to handle multiple indices per term.
        
        # I will rewrite the construction logic to be more explicit.
        
    # Reset and do it properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        col_base = b * 17
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices for B[0:2, k0:k1+1]
        v_00 = 6 * 0 + k0
        v_01 = 6 * 0 + k1
        v_10 = 6 * 1 + k0
        v_11 = 6 * 1 + k1
        
        # V indices for B[2, k0:k1+1]
        v_20 = 6 * 2 + k0
        v_21 = 6 * 2 + k1
        
        # W indices for C[0:2, k0:k1+1]
        w_00 = 3 * k0 + 0
        w_01 = 3 * k1 + 0
        w_10 = 3 * k0 + 1
        w_11 = 3 * k1 + 1
        
        # W indices for C[2, k0:k1+1]
        w_20 = 3 * k0 + 2
        w_21 = 3 * k1 + 2
        
        # --- Term 0: m1 = (a00+a11)(b00+b11) ---
        col = col_base + 0
        U[0][col] += 1
        U[4][col] += 1
        V[v_00][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        
        # --- Term 1: m2 = (a10+a11)b00 ---
        col = col_base + 1
        U[3][col] += 1
        U[4][col] += 1
        V[v_00][col] += 1
        W[w_10][col] += 1
        
        # --- Term 2: m3 = a00(b01-b11) ---
        col = col_base + 2
        U[0][col] += 1
        V[v_01][col] += 1
        V[v_11][col] += -1
        W[w_01][col] += 1
        
        # --- Term 3: m4 = a11(b10-b00) ---
        col = col_base + 3
        U[4][col] += 1
        V[v_10][col] += 1
        V[v_00][col] += -1
        W[w_10][col] += 1
        
        # --- Term 4: m5 = (a00+a01)b11 ---
        col = col_base + 4
        U[0][col] += 1
        U[1][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += 1
        W[w_01][col] += 1
        
        # --- Term 5: m6 = (a10-a00)(b00+b01) ---
        col = col_base + 5
        U[3][col] += 1
        U[0][col] += -1
        V[v_00][col] += 1
        V[v_01][col] += 1
        W[w_11][col] += 1
        
        # --- Term 6: m7 = (a01-a11)(b10+b11) ---
        col = col_base + 6
        U[1][col] += 1
        U[4][col] += -1
        V[v_10][col] += 1
        V[v_11][col] += 1
        W[w_00][col] += -1
        W[w_11][col] += -1
        
        # --- Term 7: a02*b20 ---
        col = col_base + 7
        U[2][col] += 1
        V[v_20][col] += 1
        W[w_00][col] += 1
        
        # --- Term 8: a02*b21 ---
        col = col_base + 8
        U[2][col] += 1
        V[v_21][col] += 1
        W[w_01][col] += 1
        
        # --- Term 9: a12*b20 ---
        col = col_base + 9
        U[5][col] += 1
        V[v_20][col] += 1
        W[w_10][col] += 1
        
        # --- Term 10: a12*b21 ---
        col = col_base + 10
        U[5][col] += 1
        V[v_21][col] += 1
        W[w_11][col] += 1
        
        # --- Term 11: a20*b00 ---
        col = col_base + 11
        U[6][col] += 1
        V[v_00][col] += 1
        W[w_20][col] += 1
        
        # --- Term 12: a20*b01 ---
        col = col_base + 12
        U[6][col] += 1
        V[v_01][col] += 1
        W[w_21][col] += 1
        
        # --- Term 13: a21*b10 ---
        col = col_base + 13
        U[7][col] += 1
        V[v_10][col] += 1
        W[w_20][col] += 1
        
        # --- Term 14: a21*b11 ---
        col = col_base + 14
        U[7][col] += 1
        V[v_11][col] += 1
        W[w_21][col] += 1
        
        # --- Term 15: a22*b20 ---
        col = col_base + 15
        U[8][col] += 1
        V[v_20][col] += 1
        W[w_20][col] += 1
        
        # --- Term 16: a22*b21 ---
        col = col_base + 16
        U[8][col] += 1
        V[v_21][col] += 1
        W[w_21][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
