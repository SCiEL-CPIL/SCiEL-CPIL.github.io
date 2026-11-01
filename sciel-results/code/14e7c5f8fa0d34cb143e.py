import time

def construct():
    """
    Construct a rank-58 exact tensor decomposition
    for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The output matrix C is 2x6.
    U has shape (2*5, 58) = (10, 58)
    V has shape (5*6, 58) = (30, 58)
    W has shape (6*2, 58) = (12, 58)
    
    Indexing:
    U[i*m + j, r]  where m=5
    V[j*p + k, r]  where p=6
    W[k*n + i, r]  where n=2 (official trace convention)
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    n, m, p = 2, 5, 6
    rank = 58
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        """Add one rank-1 term to the decomposition."""
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

    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: a00=u(0,0)=0, a01=u(0,1)=1, a10=u(1,0)=5, a11=u(1,1)=6
    # B indices: b00=v(0,0)=0, b01=v(0,1)=1, b10=v(1,0)=6, b11=v(1,1)=7
    # C indices: c00=w(0,0)=0, c10=w(0,1)=1, c01=w(1,0)=2, c11=w(1,1)=3
    
    # m1 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    add_rank_step([(u_idx(0,0), 1), (u_idx(1,1), 1)], 
                  [(v_idx(0,0), 1), (v_idx(1,1), 1)], 
                  [(w_idx(0,0), 1), (w_idx(1,1), 1)])
    
    # m2 = (a10+a11)*b00 -> C10 += 1, C11 -= 1
    add_rank_step([(u_idx(1,0), 1), (u_idx(1,1), 1)], 
                  [(v_idx(0,0), 1)], 
                  [(w_idx(0,1), 1), (w_idx(1,1), -1)])
    
    # m3 = a00*(b01-b11) -> C01 += 1, C11 += 1
    add_rank_step([(u_idx(0,0), 1)], 
                  [(v_idx(0,1), 1), (v_idx(1,1), -1)], 
                  [(w_idx(1,0), 1), (w_idx(1,1), 1)])
    
    # m4 = a11*(b10-b00) -> C00 += 1, C10 += 1
    add_rank_step([(u_idx(1,1), 1)], 
                  [(v_idx(1,0), 1), (v_idx(0,0), -1)], 
                  [(w_idx(0,0), 1), (w_idx(0,1), 1)])
    
    # m5 = (a00+a01)*b11 -> C00 -= 1, C01 += 1
    add_rank_step([(u_idx(0,0), 1), (u_idx(0,1), 1)], 
                  [(v_idx(1,1), 1)], 
                  [(w_idx(0,0), -1), (w_idx(1,0), 1)])
    
    # m6 = (a10-a00)*(b00+b01) -> C11 += 1
    add_rank_step([(u_idx(1,0), 1), (u_idx(0,0), -1)], 
                  [(v_idx(0,0), 1), (v_idx(0,1), 1)], 
                  [(w_idx(1,1), 1)])
    
    # m7 = (a01-a11)*(b10+b11) -> C00 += 1
    add_rank_step([(u_idx(0,1), 1), (u_idx(1,1), -1)], 
                  [(v_idx(1,0), 1), (v_idx(1,1), 1)], 
                  [(w_idx(0,0), 1)])

    # Part 2: Strassen for A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A indices: a02=u(0,2)=2, a03=u(0,3)=3, a12=u(1,2)=7, a13=u(1,3)=8
    # B indices: b20=v(2,0)=12, b21=v(2,1)=13, b30=v(3,0)=18, b31=v(3,1)=19
    # C indices: same as Part 1
    
    # m1 = (a02+a13)(b20+b31) -> C00 += 1, C11 += 1
    add_rank_step([(u_idx(0,2), 1), (u_idx(1,3), 1)], 
                  [(v_idx(2,0), 1), (v_idx(3,1), 1)], 
                  [(w_idx(0,0), 1), (w_idx(1,1), 1)])
    
    # m2 = (a12+a13)*b20 -> C10 += 1, C11 -= 1
    add_rank_step([(u_idx(1,2), 1), (u_idx(1,3), 1)], 
                  [(v_idx(2,0), 1)], 
                  [(w_idx(0,1), 1), (w_idx(1,1), -1)])
    
    # m3 = a02*(b21-b31) -> C01 += 1, C11 += 1
    add_rank_step([(u_idx(0,2), 1)], 
                  [(v_idx(2,1), 1), (v_idx(3,1), -1)], 
                  [(w_idx(1,0), 1), (w_idx(1,1), 1)])
    
    # m4 = a13*(b30-b20) -> C00 += 1, C10 += 1
    add_rank_step([(u_idx(1,3), 1)], 
                  [(v_idx(3,0), 1), (v_idx(2,0), -1)], 
                  [(w_idx(0,0), 1), (w_idx(0,1), 1)])
    
    # m5 = (a02+a03)*b31 -> C00 -= 1, C01 += 1
    add_rank_step([(u_idx(0,2), 1), (u_idx(0,3), 1)], 
                  [(v_idx(3,1), 1)], 
                  [(w_idx(0,0), -1), (w_idx(1,0), 1)])
    
    # m6 = (a12-a02)*(b20+b21) -> C11 += 1
    add_rank_step([(u_idx(1,2), 1), (u_idx(0,2), -1)], 
                  [(v_idx(2,0), 1), (v_idx(2,1), 1)], 
                  [(w_idx(1,1), 1)])
    
    # m7 = (a03-a13)*(b30+b31) -> C00 += 1
    add_rank_step([(u_idx(0,3), 1), (u_idx(1,3), -1)], 
                  [(v_idx(3,0), 1), (v_idx(3,1), 1)], 
                  [(w_idx(0,0), 1)])

    # Part 3: Schoolbook for A[0:2, 4] * B[4, 0:2] -> C[0:2, 0:2]
    # A indices: u(0,4)=4, u(1,4)=9
    # B indices: v(4,0)=24, v(4,1)=25
    
    # i=0, j=4, k=0: C[0,0] += A[0,4]*B[4,0]
    add_rank_step([(u_idx(0,4), 1)], [(v_idx(4,0), 1)], [(w_idx(0,0), 1)])
    # i=0, j=4, k=1: C[0,1] += A[0,4]*B[4,1]
    add_rank_step([(u_idx(0,4), 1)], [(v_idx(4,1), 1)], [(w_idx(1,0), 1)])
    # i=1, j=4, k=0: C[1,0] += A[1,4]*B[4,0]
    add_rank_step([(u_idx(1,4), 1)], [(v_idx(4,0), 1)], [(w_idx(0,1), 1)])
    # i=1, j=4, k=1: C[1,1] += A[1,4]*B[4,1]
    add_rank_step([(u_idx(1,4), 1)], [(v_idx(4,1), 1)], [(w_idx(1,1), 1)])

    # Part 4: Schoolbook for A[0:2, 0:5] * B[0:5, 2:6] -> C[0:2, 2:6]
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                if time.time() - start_time > deadline:
                    return {"U": U, "V": V, "W": W}
                add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])

    return {"U": U, "V": V, "W": W}
