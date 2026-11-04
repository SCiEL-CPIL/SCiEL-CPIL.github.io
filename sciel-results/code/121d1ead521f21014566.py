import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal rank
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        rank += 1

    # Part 1: j=0
    # A[:, 0] * B[0, :]
    # Rank 20
    for i in range(n):
        for k in range(p):
            if time.time() > deadline - 5:
                break
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            
            u[i * m + 0] = 1
            v[0 * p + k] = 1
            w[k * n + i] = 1
            
            add_term(u, v, w)
        if time.time() > deadline - 5:
            break

    # Part 2: j=1,2
    # A[:, 1:3] * B[1:3, :]
    # 5 blocks of Strassen
    for b in range(5):
        if time.time() > deadline - 5:
            break
        k_offset = 2 * b
        
        # M1
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[0 * m + 1] = 1
        u[1 * m + 2] = 1
        v[1 * p + k_offset] = 1
        v[2 * p + k_offset + 1] = 1
        w[k_offset * n + 0] = 1
        w[(k_offset + 1) * n + 1] = 1
        add_term(u, v, w)
        
        # M2
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[1 * m + 1] = 1
        u[1 * m + 2] = 1
        v[1 * p + k_offset] = 1
        w[k_offset * n + 1] = 1
        w[(k_offset + 1) * n + 1] = -1
        add_term(u, v, w)
        
        # M3
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[0 * m + 1] = 1
        v[1 * p + k_offset + 1] = 1
        v[2 * p + k_offset + 1] = -1
        w[(k_offset + 1) * n + 0] = 1
        w[(k_offset + 1) * n + 1] = 1
        add_term(u, v, w)
        
        # M4
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[1 * m + 2] = 1
        v[2 * p + k_offset] = 1
        v[1 * p + k_offset] = -1
        w[k_offset * n + 0] = 1
        w[k_offset * n + 1] = 1
        add_term(u, v, w)
        
        # M5
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[0 * m + 1] = 1
        u[1 * m + 1] = 1
        v[2 * p + k_offset + 1] = 1
        w[k_offset * n + 0] = -1
        w[(k_offset + 1) * n + 0] = 1
        add_term(u, v, w)
        
        # M6
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[0 * m + 2] = 1
        u[0 * m + 1] = -1
        v[1 * p + k_offset] = 1
        v[2 * p + k_offset] = 1
        w[(k_offset + 1) * n + 1] = 1
        add_term(u, v, w)
        
        # M7
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[1 * m + 1] = 1
        u[1 * m + 2] = -1
        v[1 * p + k_offset + 1] = 1
        v[2 * p + k_offset + 1] = 1
        w[k_offset * n + 0] = 1
        add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
