DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 40
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank1(u_vec, v_vec, w_vec):
        nonlocal col
        for idx, val in u_vec.items():
            U[idx][col] += val
        for idx, val in v_vec.items():
            V[idx][col] += val
        for idx, val in w_vec.items():
            W[idx][col] += val
        col += 1

    def get_u(i, j): return i*m + j
    def get_v(j, k): return j*p + k
    def get_w(k, i): return k*n + i

    # Block 1: k in 0,1,2
    # M1
    add_rank1(
        {get_u(0,0): 1, get_u(1,1): 1},
        {get_v(0,0): 1, get_v(1,1): 1},
        {get_w(0,0): 1, get_w(1,1): 1}
    )
    # M2
    add_rank1(
        {get_u(1,0): 1},
        {get_v(0,0): 1, get_v(1,0): 1},
        {get_w(0,1): 1}
    )
    # M3
    add_rank1(
        {get_u(0,0): 1},
        {get_v(0,1): 1, get_v(1,1): -1},
        {get_w(1,0): 1, get_w(1,1): 1}
    )
    # M4
    add_rank1(
        {get_u(1,1): 1},
        {get_v(0,1): 1, get_v(1,1): 1},
        {get_w(0,0): 1, get_w(1,1): 1}
    )
    # M5
    add_rank1(
        {get_u(0,0): 1, get_u(1,0): 1},
        {get_v(1,1): 1},
        {get_w(0,0): -1, get_w(1,1): 1}
    )
    # M6
    add_rank1(
        {get_u(1,1): 1, get_u(0,0): -1},
        {get_v(0,0): 1, get_v(1,0): 1},
        {get_w(0,1): 1}
    )
    # M7
    add_rank1(
        {get_u(0,1): 1, get_u(1,1): -1},
        {get_v(0,1): 1, get_v(1,1): 1},
        {get_w(1,0): 1, get_w(0,0): -1}
    )
    
    # 1.2: 2x1x2
    for i in range(2):
        for k in range(2):
            add_rank1(
                {get_u(i,2): 1},
                {get_v(2,k): 1},
                {get_w(k,i): 1}
            )
            
    # 1.3: 2x2x1
    for i in range(2):
        for j in range(2):
            add_rank1(
                {get_u(i,j): 1},
                {get_v(j,2): 1},
                {get_w(2,i): 1}
            )
            
    # 1.4: 2x1x1
    for i in range(2):
        add_rank1(
            {get_u(i,2): 1},
            {get_v(2,2): 1},
            {get_w(2,i): 1}
        )

    # Block 2: k in 3,4,5
    # M1
    add_rank1(
        {get_u(0,0): 1, get_u(1,1): 1},
        {get_v(0,3): 1, get_v(1,4): 1},
        {get_w(3,0): 1, get_w(4,1): 1}
    )
    # M2
    add_rank1(
        {get_u(1,0): 1},
        {get_v(0,3): 1, get_v(1,3): 1},
        {get_w(3,1): 1}
    )
    # M3
    add_rank1(
        {get_u(0,0): 1},
        {get_v(0,4): 1, get_v(1,4): -1},
        {get_w(4,0): 1, get_w(4,1): 1}
    )
    # M4
    add_rank1(
        {get_u(1,1): 1},
        {get_v(0,4): 1, get_v(1,4): 1},
        {get_w(3,0): 1, get_w(4,1): 1}
    )
    # M5
    add_rank1(
        {get_u(0,0): 1, get_u(1,0): 1},
        {get_v(1,4): 1},
        {get_w(3,0): -1, get_w(4,1): 1}
    )
    # M6
    add_rank1(
        {get_u(1,1): 1, get_u(0,0): -1},
        {get_v(0,3): 1, get_v(1,3): 1},
        {get_w(3,1): 1}
    )
    # M7
    add_rank1(
        {get_u(0,1): 1, get_u(1,1): -1},
        {get_v(0,4): 1, get_v(1,4): 1},
        {get_w(4,0): 1, get_w(3,0): -1}
    )
    
    # 2.2: 2x1x2
    for i in range(2):
        for k in range(3, 5):
            add_rank1(
                {get_u(i,2): 1},
                {get_v(2,k): 1},
                {get_w(k,i): 1}
            )
            
    # 2.3: 2x2x1
    for i in range(2):
        for j in range(2):
            add_rank1(
                {get_u(i,j): 1},
                {get_v(j,5): 1},
                {get_w(5,i): 1}
            )
            
    # 2.4: 2x1x1
    for i in range(2):
        add_rank1(
            {get_u(i,2): 1},
            {get_v(2,5): 1},
            {get_w(5,i): 1}
        )

    # Block 3: k in 6
    for i in range(2):
        for j in range(3):
            add_rank1(
                {get_u(i,j): 1},
                {get_v(j,6): 1},
                {get_w(6,i): 1}
            )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
