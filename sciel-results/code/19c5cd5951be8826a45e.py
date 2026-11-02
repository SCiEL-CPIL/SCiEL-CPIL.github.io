"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 6 rows
    V = [[0] * rank for _ in range(m * p)]   # 27 rows
    W = [[0] * rank for _ in range(p * n)]   # 18 rows

    col = 0

    # --- Block A_1 * B_1 : i in {0,1}, j=0, k in {0..8} : 2*9 = 18 terms ---
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1          # U[i, 0]
            V[0 * p + k][col] = 1          # V[0, k]
            W[k * n + i][col] = 1          # W[k, i]
            col += 1
    # col is now 18

    # --- Block A_2 * B_2 : i in {0,1}, j in {1,2}, k in {0..8} ---
    # Split k into 4 blocks of size 2 and 1 block of size 1
    # Strassen for 2x2x2 blocks (rank 7 each)
    # Schoolbook for 2x2x1 block (rank 4)
    
    def add_strassen_block(k_start):
        """
        Adds Strassen terms for A_2 (2x2) * B_2_sub (2x2)
        A_2 indices: i in {0,1}, j in {1,2}
        B_2_sub indices: j in {1,2}, k in {k_start, k_start+1}
        C_sub indices: i in {0,1}, k in {k_start, k_start+1}
        """
        nonlocal col
        # Local indices for U: i*m + (1+jl) where jl in {0,1} maps to j in {1,2}
        # Local indices for V: (1+jl)*p + (k_start+kl) where jl in {0,1}, kl in {0,1}
        # Local indices for W: (k_start+kl)*n + i where kl in {0,1}, i in {0,1}
        
        def u_idx(i, jl):
            return i * m + (1 + jl)
        def v_idx(jl, kl):
            return (1 + jl) * p + (k_start + kl)
        def w_idx(kl, i):
            return (k_start + kl) * n + i

        # M1 = (A00+A11)*(B00+B11)
        U[u_idx(0, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += 1  # C00
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1

        # M2 = (A10+A11)*B00
        U[u_idx(1, 0)][col] += 1
        U[u_idx(1, 1)][col] += 1
        V[v_idx(0, 0)][col] += 1
        W[w_idx(0, 1)][col] += 1  # C10 (row 1, col 0 of C block) -> W index for k=0, i=1
        col += 1

        # M3 = A00*(B01-B11)
        U[u_idx(0, 0)][col] += 1
        V[v_idx(0, 1)][col] += 1
        V[v_idx(1, 1)][col] -= 1
        W[w_idx(1, 0)][col] += 1  # C01 (row 0, col 1 of C block) -> W index for k=1, i=0
        col += 1

        # M4 = A11*(B10-B00)
        U[u_idx(1, 1)][col] += 1
        V[v_idx(1, 0)][col] += 1
        V[v_idx(0, 0)][col] -= 1
        W[w_idx(0, 1)][col] += 1  # C10
        W[w_idx(0, 0)][col] += 1  # C00
        col += 1

        # M5 = (A00+A01)*B11
        U[u_idx(0, 0)][col] += 1
        U[u_idx(0, 1)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(1, 1)][col] += 1  # C11
        W[w_idx(1, 0)][col] += 1  # C01
        col += 1

        # M6 = (A10-A00)*(B00+B01)
        U[u_idx(1, 0)][col] += 1
        U[u_idx(0, 0)][col] -= 1
        V[v_idx(0, 0)][col] += 1
        V[v_idx(0, 1)][col] += 1
        W[w_idx(1, 1)][col] += 1  # C11
        col += 1

        # M7 = (A01-A10)*(B10+B11)
        U[u_idx(0, 1)][col] += 1
        U[u_idx(1, 0)][col] -= 1
        V[v_idx(1, 0)][col] += 1
        V[v_idx(1, 1)][col] += 1
        W[w_idx(0, 0)][col] += 1  # C00
        col += 1

    # Apply Strassen to four 2x2x2 blocks
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    add_strassen_block(6)
    
    # col is now 18 + 28 = 46

    # --- Last k-block: k=8, width 1: schoolbook 2x2x1 = 4 terms ---
    # A_2 (2x2) * B_2_sub (2x1)
    # C_sub (2x1)
    # Terms:
    # C_0,8 = A_0,1 B_1,8 + A_0,2 B_2,8
    # C_1,8 = A_1,1 B_1,8 + A_1,2 B_2,8
    
    # Term 1: A_0,1 * B_1,8 -> C_0,8
    U[0 * m + 1][col] = 1
    V[1 * p + 8][col] = 1
    W[8 * n + 0][col] = 1
    col += 1
    
    # Term 2: A_0,2 * B_2,8 -> C_0,8
    U[0 * m + 2][col] = 1
    V[2 * p + 8][col] = 1
    W[8 * n + 0][col] = 1
    col += 1
    
    # Term 3: A_1,1 * B_1,8 -> C_1,8
    U[1 * m + 1][col] = 1
    V[1 * p + 8][col] = 1
    W[8 * n + 1][col] = 1
    col += 1
    
    # Term 4: A_1,2 * B_2,8 -> C_1,8
    U[1 * m + 2][col] = 1
    V[2 * p + 8][col] = 1
    W[8 * n + 1][col] = 1
    col += 1

    # Verify col == 50
    assert col == 50, f"Expected 50 columns, got {col}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
