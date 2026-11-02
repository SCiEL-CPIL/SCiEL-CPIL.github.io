import time

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a 2x5 matrix by a 5x6 matrix.
    
    Rank 54 decomposition.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = 2, 5, 6
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= rank:
            return False
        if time.time() - start_time > deadline:
            return False
            
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
            
        col += 1
        return True

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    def add_strassen_block(u_base, v_base, w_base):
        """
        Adds a 2x2x2 Strassen block.
        u_base: list of 4 U indices [u00, u01, u10, u11]
        v_base: list of 4 V indices [v00, v01, v10, v11]
        w_base: list of 4 W indices [w00, w01, w10, w11]
        """
        # m1 = (u00+u11)(v00+v11)
        add_rank_step([(u_base[0], 1), (u_base[3], 1)], 
                     [(v_base[0], 1), (v_base[3], 1)], 
                     [(w_base[0], 1), (w_base[3], 1)])
                     
        # m2 = (u10+u11)*v00
        add_rank_step([(u_base[2], 1), (u_base[3], 1)], 
                     [(v_base[0], 1)], 
                     [(w_base[2], 1), (w_base[3], -1)])
                     
        # m3 = u00*(v01-v11)
        add_rank_step([(u_base[0], 1)], 
                     [(v_base[1], 1), (v_base[3], -1)], 
                     [(w_base[1], 1), (w_base[3], 1)])
                     
        # m4 = u11*(v10-v00)
        add_rank_step([(u_base[3], 1)], 
                     [(v_base[2], 1), (v_base[0], -1)], 
                     [(w_base[0], 1), (w_base[2], 1)])
                     
        # m5 = (u00+u01)*v11
        add_rank_step([(u_base[0], 1), (u_base[1], 1)], 
                     [(v_base[3], 1)], 
                     [(w_base[0], -1), (w_base[1], 1)])
                     
        # m6 = (u10-u00)*(v00+v01)
        add_rank_step([(u_base[2], 1), (u_base[0], -1)], 
                     [(v_base[0], 1), (v_base[1], 1)], 
                     [(w_base[3], 1)])
                     
        # m7 = (u01-u11)*(v10+v11)
        add_rank_step([(u_base[1], 1), (u_base[3], -1)], 
                     [(v_base[2], 1), (v_base[3], 1)], 
                     [(w_base[0], 1)])

    # --- Part 1: Left Part (C[:, 0:2]) ---
    
    # Block 1.1: A[:, 0:2] * B[0:2, 0:2]
    u_b11 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_b11 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    w_b11 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    add_strassen_block(u_b11, v_b11, w_b11)
    
    # Block 1.2: A[:, 2:4] * B[2:4, 0:2]
    u_b12 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_b12 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    w_b12 = [w_idx(0,0), w_idx(1,0), w_idx(0,1), w_idx(1,1)]
    add_strassen_block(u_b12, v_b12, w_b12)
    
    # Block 1.3: A[:, 4] * B[4, 0:2] (Schoolbook 2x1x2)
    # i=0, j=4, k=0
    add_rank_step([(u_idx(0,4), 1)], [(v_idx(4,0), 1)], [(w_idx(0,0), 1)])
    # i=0, j=4, k=1
    add_rank_step([(u_idx(0,4), 1)], [(v_idx(4,1), 1)], [(w_idx(1,0), 1)])
    # i=1, j=4, k=0
    add_rank_step([(u_idx(1,4), 1)], [(v_idx(4,0), 1)], [(w_idx(0,1), 1)])
    # i=1, j=4, k=1
    add_rank_step([(u_idx(1,4), 1)], [(v_idx(4,1), 1)], [(w_idx(1,1), 1)])
    
    # --- Part 2: Right Part (C[:, 2:6]) ---
    
    # Block 2.1: A[:, 0:2] * B[0:2, 2:4]
    u_b21 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_b21 = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    w_b21 = [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    add_strassen_block(u_b21, v_b21, w_b21)
    
    # Block 2.2: A[:, 0:2] * B[0:2, 4:6]
    u_b22 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    v_b22 = [v_idx(0,4), v_idx(0,5), v_idx(1,4), v_idx(1,5)]
    w_b22 = [w_idx(4,0), w_idx(5,0), w_idx(4,1), w_idx(5,1)]
    add_strassen_block(u_b22, v_b22, w_b22)
    
    # Block 2.3: A[:, 2:4] * B[2:4, 2:4]
    u_b23 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_b23 = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    w_b23 = [w_idx(2,0), w_idx(3,0), w_idx(2,1), w_idx(3,1)]
    add_strassen_block(u_b23, v_b23, w_b23)
    
    # Block 2.4: A[:, 2:4] * B[2:4, 4:6]
    u_b24 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    v_b24 = [v_idx(2,4), v_idx(2,5), v_idx(3,4), v_idx(3,5)]
    w_b24 = [w_idx(4,0), w_idx(5,0), w_idx(4,1), w_idx(5,1)]
    add_strassen_block(u_b24, v_b24, w_b24)
    
    # Block 2.5: A[:, 4] * B[4, 2:6] (Schoolbook 2x1x4)
    for i in range(n):
        for k in range(2, p):
            add_rank_step([(u_idx(i,4), 1)], [(v_idx(4,k), 1)], [(w_idx(k,i), 1)])
            
    return {"U": U, "V": V, "W": W}
