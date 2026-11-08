"""Matrix multiplication 3x4x8 rank-88 decomposition (Strassen tiling)."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 4, 8

    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 12 x 88
    V = [[0] * rank for _ in range(m * p)]  # 32 x 88
    W = [[0] * rank for _ in range(p * n)]  # 24 x 88

    col = 0

    # Part 1: i=0, schoolbook for 1x4x8 -> 32 terms
    for j in range(4):
        for k in range(8):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Part 2: i=1,2 using Strassen on 2x4x8
    # Split j into 2 blocks of 2, k into 4 blocks of 2
    # 7 Strassen terms per (j_block, k_block) -> 2*4*7 = 56 terms

    # Strassen 2x2x2 factors
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    strassen_U = [
        (1, 0, 0, 1),   # M1: A00 + A11
        (0, 0, 1, 1),   # M2: A10 + A11
        (1, 0, 0, 0),   # M3: A00
        (0, 0, 0, 1),   # M4: A11
        (1, 1, 0, 0),   # M5: A00 + A01
        (-1, 0, 1, 0),  # M6: A10 - A00
        (0, 1, 0, -1)   # M7: A01 - A11
    ]
    strassen_V = [
        (1, 0, 0, 1),   # M1: B00 + B11
        (1, 0, 0, 0),   # M2: B00
        (0, 1, 0, -1),  # M3: B10 - B11
        (-1, 0, 1, 0),  # M4: B01 - B00
        (0, 0, 0, 1),   # M5: B11
        (1, 1, 0, 0),   # M6: B00 + B01
        (0, 0, 1, 1)    # M7: B10 + B11
    ]
    strassen_W = [
        (1, 0, 0, 1),   # M1: C00 + C11
        (0, 0, 1, -1),  # M2: C10 - C11
        (0, 1, 0, 1),   # M3: C01 + C11
        (1, 0, 1, 0),   # M4: C00 + C10
        (-1, 1, 0, 0),  # M5: -C00 + C01
        (0, 0, 0, 1),   # M6: C11
        (1, 0, 0, 0)    # M7: C00
    ]

    # Precompute local index mappings
    # For U: l -> (local_i, local_j) where l = local_i*2 + local_j
    # For V: l -> (local_j, local_k) where l = local_j*2 + local_k  
    # For W: l -> (local_i, local_k) where l = local_i*2 + local_k
    # Precompute non-zero entries for each factor
    U_entries = []
    V_entries = []
    W_entries = []
    for t in range(7):
        ue = [(l, strassen_U[t][l]) for l in range(4) if strassen_U[t][l] != 0]
        ve = [(l, strassen_V[t][l]) for l in range(4) if strassen_V[t][l] != 0]
        we = [(l, strassen_W[t][l]) for l in range(4) if strassen_W[t][l] != 0]
        U_entries.append(ue)
        V_entries.append(ve)
        W_entries.append(we)

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2

            for t in range(7):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}

                # U: A-side
                for l, val in U_entries[t]:
                    local_i = l >> 1
                    local_j = l & 1
                    global_i = 1 + local_i
                    global_j = j_start + local_j
                    U[global_i * m + global_j][col] += val

                # V: B-side
                for l, val in V_entries[t]:
                    local_j = l >> 1
                    local_k = l & 1
                    global_j = j_start + local_j
                    global_k = k_start + local_k
                    V[global_j * p + global_k][col] += val

                # W: C-side
                for l, val in W_entries[t]:
                    local_i = l >> 1
                    local_k = l & 1
                    global_i = 1 + local_i
                    global_k = k_start + local_k
                    W[global_k * n + global_i][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
