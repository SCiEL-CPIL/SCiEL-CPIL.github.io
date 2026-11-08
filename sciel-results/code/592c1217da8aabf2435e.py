import time
import numpy as np

_DIMS = (3, 4, 6)
_DEADLINE = time.time() + 110

def construct():
    """
    Construct a rank-66 exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy: Split p=6 into three blocks of 2. For each block:
    - i=0: M(1,4,2) with rank 8 (schoolbook)
    - i=1,2: M(2,4,2) with rank 14 (two M(2,2,2) Strassen blocks)
    Total per block: 22, times 3 = 66.
    """
    n, m, p = _DIMS
    rank = 66
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_1_4_2(i_off, j_off, k_off):
        """Add M(1,4,2) block: i fixed, j in [j_off, j_off+3], k in [k_off, k_off+1]"""
        nonlocal col
        for j in range(4):
            for k in range(2):
                u_idx = i_off * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1

    def add_2_2_2(i_off, j_off, k_off):
        """Add M(2,2,2) block using 7 rank-1 terms (Strassen)."""
        nonlocal col
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        # 7 terms for M(2,2,2) - standard Strassen decomposition
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1]),
            ([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1]),
            ([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1]),
            ([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1]),
            ([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1]),
            ([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1]),
            ([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1]),
        ]
        
        for u_idx_list, u_vals, v_idx_list, v_vals, w_idx_list, w_vals in terms:
            for idx, val in zip(u_idx_list, u_vals):
                U[idx, col] += val
            for idx, val in zip(v_idx_list, v_vals):
                V[idx, col] += val
            for idx, val in zip(w_idx_list, w_vals):
                W[idx, col] += val
            col += 1

    def add_2_4_2(i_off, j_off, k_off):
        """Add M(2,4,2) block by splitting j into two M(2,2,2)."""
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)

    # Main loop: 3 k-blocks of size 2
    for block_k in range(3):
        k_off = block_k * 2
        # i=0: M(1,4,2)
        add_1_4_2(0, 0, k_off)
        # i=1,2: M(2,4,2)
        add_2_4_2(1, 0, k_off)

    # Verify rank
    if col != rank:
        raise RuntimeError(f"Expected rank {rank}, got {col}")

    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
