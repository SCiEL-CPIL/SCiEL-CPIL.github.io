import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # 3 output blocks (k_base in 0, 2, 4)
    # For each:
    #   Top row (i_base=0, n_sub=1): 2 schoolbook blocks (j_base in 0, 2) -> 2 * 4 = 8 rank
    #   Bottom block (i_base=1, n_sub=2): 2 strassen blocks (j_base in 0, 2) -> 2 * 7 = 14 rank
    # Total per output block: 22
    # Total rank: 3 * 22 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_row, v_row, w_row, val):
        nonlocal col
        if val != 0:
            U[u_row][col] += val
            V[v_row][col] += val
            W[w_row][col] += val
        # Note: We need to advance col after adding all terms for a specific factor
        # But the structure above adds multiple U/V/W entries for one factor.
        # So we should define the whole factor addition, then col += 1.

    def add_factor(u_entries, v_entries, w_entries):
        nonlocal col
        for u_r, val in u_entries:
            U[u_r][col] += val
        for v_r, val in v_entries:
            V[v_r][col] += val
        for w_r, val in w_entries:
            W[w_r][col] += val
        col += 1

    # Loop through output blocks
    for k_base in range(0, p, 2):
        # Top row: i_base = 0, n_sub = 1
        # Schoolbook 1x2x2
        for j_base in range(0, m, 2):
            # i_glob = 0
            # j_glob in j_base, j_base+1
            # k_glob in k_base, k_base+1
            
            # Term 1: a0 b0 -> c0
            # U[0*m + j_base]
            # V[j_base*p + k_base]
            # W[k_base*n + 0]
            add_factor(
                [(0 * m + j_base, 1)],
                [(j_base * p + k_base, 1)],
                [(k_base * n + 0, 1)]
            )
            
            # Term 2: a0 b1 -> c1
            # U[0*m + j_base]
            # V[j_base*p + (k_base+1)]
            # W[(k_base+1)*n + 0]
            add_factor(
                [(0 * m + j_base, 1)],
                [(j_base * p + (k_base + 1), 1)],
                [((k_base + 1) * n + 0, 1)]
            )
            
            # Term 3: a1 b2 -> c0
            # U[0*m + (j_base+1)]
            # V[(j_base+1)*p + k_base]
            # W[k_base*n + 0]
            add_factor(
                [(0 * m + (j_base + 1), 1)],
                [((j_base + 1) * p + k_base, 1)],
                [(k_base * n + 0, 1)]
            )
            
            # Term 4: a1 b3 -> c1
            # U[0*m + (j_base+1)]
            # V[(j_base+1)*p + (k_base+1)]
            # W[(k_base+1)*n + 0]
            add_factor(
                [(0 * m + (j_base + 1), 1)],
                [((j_base + 1) * p + (k_base + 1), 1)],
                [((k_base + 1) * n + 0, 1)]
            )

        # Bottom block: i_base = 1, n_sub = 2
        # Strassen 2x2x2
        for j_base in range(0, m, 2):
            # i_glob in 1, 2
            # j_glob in j_base, j_base+1
            # k_glob in k_base, k_base+1
            
            # Helper indices
            i0 = 1
            i1 = 2
            j0 = j_base
            j1 = j_base + 1
            k0 = k_base
            k1 = k_base + 1
            
            # U indices
            u00 = i0 * m + j0
            u01 = i0 * m + j1
            u10 = i1 * m + j0
            u11 = i1 * m + j1
            
            # V indices
            v00 = j0 * p + k0
            v01 = j0 * p + k1
            v10 = j1 * p + k0
            v11 = j1 * p + k1
            
            # W indices (k*n + i)
            w00 = k0 * n + i0
            w01 = k1 * n + i0
            w10 = k0 * n + i1
            w11 = k1 * n + i1
            
            # M1: (a00 + a11)(b00 + b11)
            # U: u00, u11
            # V: v00, v11
            # W: w00, w11
            add_factor(
                [(u00, 1), (u11, 1)],
                [(v00, 1), (v11, 1)],
                [(w00, 1), (w11, 1)]
            )
            
            # M2: (a10 + a11) b00
            # U: u10, u11
            # V: v00
            # W: w10, w11 (-1)
            add_factor(
                [(u10, 1), (u11, 1)],
                [(v00, 1)],
                [(w10, 1), (w11, -1)]
            )
            
            # M3: a00 (b01 - b11)
            # U: u00
            # V: v01, v11 (-1)
            # W: w01, w11
            add_factor(
                [(u00, 1)],
                [(v01, 1), (v11, -1)],
                [(w01, 1), (w11, 1)]
            )
            
            # M4: a11 (b10 - b00)
            # U: u11
            # V: v10, v00 (-1)
            # W: w00, w10
            add_factor(
                [(u11, 1)],
                [(v10, 1), (v00, -1)],
                [(w00, 1), (w10, 1)]
            )
            
            # M5: (a00 + a01) b11
            # U: u00, u01
            # V: v11
            # W: w00 (-1), w01
            add_factor(
                [(u00, 1), (u01, 1)],
                [(v11, 1)],
                [(w00, -1), (w01, 1)]
            )
            
            # M6: (a10 - a00) (b00 + b01)
            # U: u10, u00 (-1)
            # V: v00, v01
            # W: w11
            add_factor(
                [(u10, 1), (u00, -1)],
                [(v00, 1), (v01, 1)],
                [(w11, 1)]
            )
            
            # M7: (a01 - a11) (b10 + b11)
            # U: u01, u11 (-1)
            # V: v10, v11
            # W: w00
            add_factor(
                [(u01, 1), (u11, -1)],
                [(v10, 1), (v11, 1)],
                [(w00, 1)]
            )
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
