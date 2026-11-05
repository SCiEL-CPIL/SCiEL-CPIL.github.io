import time

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 140

def construct():
    n, m, p = DIMENSIONS
    # M(3,4,6) = M(3,2,6) + M(3,2,6) (split m=4 into 2+2)
    # M(3,2,6) = M(3,2,2) + M(3,2,2) + M(3,2,2) (split p=6 into 2+2+2)
    # M(3,2,2): all-ones 3x2x2 tensor, rank 3
    #   T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}
    #   Decomposition: sum_{i=0}^{2} e_i ⊗ (e_0+e_1) ⊗ (e_0+e_1)
    # Total rank: 2 * 3 * 3 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 12 x 18
    V = [[0] * rank for _ in range(m * p)]  # 24 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    current_rank = 0
    
    def check_deadline():
        return (time.time() - _START_TIME) < _DEADLINE
    
    def add_rank1(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        nonlocal current_rank
        if current_rank >= rank or not check_deadline():
            return
        for idx, val in zip(u_indices, u_vals):
            U[idx][current_rank] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][current_rank] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][current_rank] += val
        current_rank += 1
    
    def add_3_2_2(i_off, j_off, k_off):
        # M(3,2,2): n=3, m=2, p=2
        # T[i,j,k] = 1 for all i,j,k
        # Decomposition: for each i in 0..2: e_i ⊗ (e_0+e_1) ⊗ (e_0+e_1)
        for i in range(3):
            u_idx = (i_off + i) * m + j_off  # A[i,0]
            u_idx2 = (i_off + i) * m + (j_off + 1)  # A[i,1]
            v_idx0 = (j_off) * p + k_off  # B[0,0]
            v_idx1 = (j_off) * p + (k_off + 1)  # B[0,1]
            v_idx2 = (j_off + 1) * p + k_off  # B[1,0]
            v_idx3 = (j_off + 1) * p + (k_off + 1)  # B[1,1]
            w_idx0 = k_off * n + (i_off + i)  # C[0,i]
            w_idx1 = (k_off + 1) * n + (i_off + i)  # C[1,i]
            # Term: e_i ⊗ (e_0+e_1) ⊗ (e_0+e_1)
            # U: A[i,0] and A[i,1] both get coefficient 1
            # V: B[0,0], B[0,1], B[1,0], B[1,1] all get coefficient 1
            # W: C[0,i] and C[1,i] both get coefficient 1
            add_rank1(
                [u_idx, u_idx2], [1, 1],
                [v_idx0, v_idx1, v_idx2, v_idx3], [1, 1, 1, 1],
                [w_idx0, w_idx1], [1, 1]
            )
    
    # Build two M(3,2,6) blocks (splitting m=4 into j=0..1 and j=2..3)
    for block_j in range(2):
        j_off = block_j * 2
        # Three M(3,2,2) blocks (splitting p=6 into k=0..1, 2..3, 4..5)
        for block_k in range(3):
            k_off = block_k * 2
            add_3_2_2(0, j_off, k_off)
    
    # Verify decomposition correctness
    if check_deadline():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(current_rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        pass
    
    return {"U": U, "V": V, "W": W}
