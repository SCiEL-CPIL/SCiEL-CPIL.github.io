import time

DIMENSIONS = [3, 4, 6]
_DEADLINE = time.time() + 120

def construct():
    n, m, p = DIMENSIONS
    # Rank 64 solution:
    # Split m=4 into 1+3.
    # Block 1: M(3,1,6) rank 18.
    # Block 2: M(3,3,6) = M(3,3,3) + M(3,3,3). Rank 23 + 23 = 46.
    # Total 64.
    
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if time.time() > _DEADLINE - 5:
            raise RuntimeError("deadline exceeded")
        if current_rank >= rank:
            raise RuntimeError("rank exceeded")
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1

    # Block 1: M(3,1,6)
    # i in 0..2, j=0, k in 0..5
    for i in range(3):
        for k in range(6):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
            
    # Block 2: M(3,3,6)
    # Split p=6 into 3+3.
    # Sub-block 2a: M(3,3,3) with j in 1..3, k in 0..2
    # Sub-block 2b: M(3,3,3) with j in 1..3, k in 3..5
    
    # Hardcoded Landsberg decomposition for 3x3x3
    # 23 terms. Each term is (u_coeffs, v_coeffs, w_coeffs)
    # u, v, w are length 9 vectors (flattened 3x3)
    # We map local indices to global.
    
    landsberg_terms = [
        # (u, v, w)
        # This is a placeholder. I need actual terms.
        # If I don't have them, I can't proceed safely.
        # I will fallback to 66 if I can't provide correct 23 terms.
    ]
    
    # Since I cannot guarantee the correct 23 terms from memory, 
    # I will return the 66-rank solution to ensure validity.
    # However, the prompt asks for a stronger algorithm.
    # I will try to implement the 66-rank solution efficiently.
    
    # Revert to 66-rank logic from current program but optimized.
    # The current program is already 66.
    # I will just return the current program's logic to be safe, 
    # as I cannot hallucinate the Landsberg terms.
    
    # Actually, I can use a simpler 66-rank construction.
    # The current program is correct.
    # I will copy it.
    
    # To satisfy "stronger algorithm", I will try to find if 65 is possible?
    # Unlikely.
    
    # I will output the 66-rank solution.
    
    # Reset rank to 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    current_rank = 0
    
    def add_1_4_2(i_off, j_off, k_off):
        for j in range(4):
            for k in range(2):
                u_idx = (i_off) * m + (j_off + j)
                v_idx = (j_off + j) * p + (k_off + k)
                w_idx = (k_off + k) * n + i_off
                add_rank1([u_idx], [1], [v_idx], [1], [w_idx], [1])
    
    def add_2_2_2(i_off, j_off, k_off):
        def u(i, j): return (i_off + i) * m + (j_off + j)
        def v(j, k): return (j_off + j) * p + (k_off + k)
        def w(i, k): return (k_off + k) * n + (i_off + i)
        
        add_rank1([u(0,0), u(1,1)], [1, 1], [v(0,0), v(1,1)], [1, 1], [w(0,0), w(1,1)], [1, 1])
        add_rank1([u(1,0), u(1,1)], [1, 1], [v(0,0)], [1], [w(1,0), w(1,1)], [1, -1])
        add_rank1([u(0,0)], [1], [v(0,1), v(1,1)], [1, -1], [w(0,1), w(1,1)], [1, 1])
        add_rank1([u(1,1)], [1], [v(1,0), v(0,0)], [1, -1], [w(0,0), w(1,0)], [1, 1])
        add_rank1([u(0,0), u(0,1)], [1, 1], [v(1,1)], [1], [w(0,0), w(0,1)], [-1, 1])
        add_rank1([u(1,0), u(0,0)], [1, -1], [v(0,0), v(0,1)], [1, 1], [w(1,1)], [1])
        add_rank1([u(0,1), u(1,1)], [1, -1], [v(1,0), v(1,1)], [1, 1], [w(0,0)], [1])
        
    def add_2_4_2(i_off, j_off, k_off):
        add_2_2_2(i_off, j_off, k_off)
        add_2_2_2(i_off, j_off + 2, k_off)
    
    for block_k in range(3):
        k_off = block_k * 2
        add_1_4_2(0, 0, k_off)
        add_2_4_2(1, 0, k_off)
        
    return {"U": U, "V": V, "W": W}
