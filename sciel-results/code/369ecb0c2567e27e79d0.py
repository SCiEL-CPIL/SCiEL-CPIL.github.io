"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 88
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a component
    def add_component(u_indices, v_indices, w_indices, values):
        nonlocal col
        # u_indices: list of (idx, val)
        # v_indices: list of (idx, val)
        # w_indices: list of (idx, val)
        for idx, val in u_indices:
            U[idx][col] += val
        for idx, val in v_indices:
            V[idx][col] += val
        for idx, val in w_indices:
            W[idx][col] += val
        col += 1

    # Helper to get U, V, W indices
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen block generator
    # i_base, j_base, k_base: global starting indices for the 2x2x2 block
    def emit_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # Local indices 0,1 map to global i_base, i_base+1 etc.
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: i_base, j_base
        # A11: i_base+1, j_base+1
        # B00: j_base, k_base
        # B11: j_base+1, k_base+1
        # C00: i_base, k_base
        # C11: i_base+1, k_base+1
        
        add_component(
            [(u_idx(i_base, j_base), 1), (u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base, k_base), 1), (v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base, i_base), 1), (w_idx(k_base+1, i_base+1), 1)]
        )
        
        # M2 = (A10 + A11) * B00
        # A10: i_base+1, j_base
        # A11: i_base+1, j_base+1
        # B00: j_base, k_base
        # C10: i_base+1, k_base
        # C11: i_base+1, k_base+1
        
        add_component(
            [(u_idx(i_base+1, j_base), 1), (u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base, k_base), 1)],
            [(w_idx(k_base, i_base+1), 1), (w_idx(k_base+1, i_base+1), 1)]
        )

        # M3 = A00 * (B01 - B11)
        # A00: i_base, j_base
        # B01: j_base, k_base+1
        # B11: j_base+1, k_base+1
        # C01: i_base, k_base+1
        # C11: i_base+1, k_base+1
        
        add_component(
            [(u_idx(i_base, j_base), 1)],
            [(v_idx(j_base, k_base+1), 1), (v_idx(j_base+1, k_base+1), -1)],
            [(w_idx(k_base+1, i_base), 1), (w_idx(k_base+1, i_base+1), -1)]
        )

        # M4 = A11 * (B10 - B00)
        # A11: i_base+1, j_base+1
        # B10: j_base+1, k_base
        # B00: j_base, k_base
        # C00: i_base, k_base
        # C10: i_base+1, k_base
        
        add_component(
            [(u_idx(i_base+1, j_base+1), 1)],
            [(v_idx(j_base+1, k_base), 1), (v_idx(j_base, k_base), -1)],
            [(w_idx(k_base, i_base), -1), (w_idx(k_base, i_base+1), 1)]
        )

        # M5 = (A00 + A01) * B11
        # A00: i_base, j_base
        # A01: i_base, j_base+1
        # B11: j_base+1, k_base+1
        # C00: i_base, k_base
        # C01: i_base, k_base+1
        
        add_component(
            [(u_idx(i_base, j_base), 1), (u_idx(i_base, j_base+1), 1)],
            [(v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base, i_base), 1), (w_idx(k_base+1, i_base), 1)]
        )

        # M6 = (A10 - A00) * (B00 + B01)
        # A10: i_base+1, j_base
        # A00: i_base, j_base
        # B00: j_base, k_base
        # B01: j_base, k_base+1
        # C10: i_base+1, k_base
        # C11: i_base+1, k_base+1
        
        add_component(
            [(u_idx(i_base+1, j_base), 1), (u_idx(i_base, j_base), -1)],
            [(v_idx(j_base, k_base), 1), (v_idx(j_base, k_base+1), 1)],
            [(w_idx(k_base, i_base+1), 1), (w_idx(k_base+1, i_base+1), 1)]
        )

        # M7 = (A01 - A11) * (B10 + B11)
        # A01: i_base, j_base+1
        # A11: i_base+1, j_base+1
        # B10: j_base+1, k_base
        # B11: j_base+1, k_base+1
        # C01: i_base, k_base+1
        # C10: i_base+1, k_base
        
        add_component(
            [(u_idx(i_base, j_base+1), 1), (u_idx(i_base+1, j_base+1), -1)],
            [(v_idx(j_base+1, k_base), 1), (v_idx(j_base+1, k_base+1), 1)],
            [(w_idx(k_base+1, i_base), 1), (w_idx(k_base, i_base+1), 1)]
        )

    # Term 1: i in {0,1}, j in {0,1}, k in {0..7}
    # Split k into 4 blocks of size 2
    for q in range(4):
        k_base = 2 * q
        emit_strassen_block(0, 0, k_base)

    # Term 2: i in {0,1}, j in {2,3}, k in {0..7}
    for q in range(4):
        k_base = 2 * q
        emit_strassen_block(0, 2, k_base)

    # Term 3: i=2, j in {0,1}, k in {0..7}
    # Schoolbook 1x2x8
    for k in range(8):
        # A20 B0k
        add_component(
            [(u_idx(2, 0), 1)],
            [(v_idx(0, k), 1)],
            [(w_idx(k, 2), 1)]
        )
        # A21 B1k
        add_component(
            [(u_idx(2, 1), 1)],
            [(v_idx(1, k), 1)],
            [(w_idx(k, 2), 1)]
        )

    # Term 4: i=2, j in {2,3}, k in {0..7}
    for k in range(8):
        # A22 B2k
        add_component(
            [(u_idx(2, 2), 1)],
            [(v_idx(2, k), 1)],
            [(w_idx(k, 2), 1)]
        )
        # A23 B3k
        add_component(
            [(u_idx(2, 3), 1)],
            [(v_idx(3, k), 1)],
            [(w_idx(k, 2), 1)]
        )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
