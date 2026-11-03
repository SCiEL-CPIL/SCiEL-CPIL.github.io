import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 6
    
    # Rank calculation: 7 outer steps * (7 inner Strassen + 4 inner Schoolbook) = 7 * 11 = 77
    rank = 77

    U = [[0] * rank for _ in range(n * m)]   # 16 x 77
    V = [[0] * rank for _ in range(m * p)]   # 24 x 77
    W = [[0] * rank for _ in range(p * n)]   # 24 x 77

    # Outer Strassen Steps for 4x4 * 4x6
    # A (4x4) -> 2x2 blocks of 2x2
    # B (4x6) -> 2x2 blocks of 2x3
    # C (4x6) -> 2x2 blocks of 2x3
    
    outer_steps = [
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]

    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Inner Strassen for 2x2 * 2x2 (cols 0,1)
    inner_strassen = [
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]

    # Inner Schoolbook for 2x2 * 2x1 (col 2)
    # A(r,s) * B(s,0) -> C(r,0)
    # Mapped to global: A(r,s), B(s,2), C(r,2)
    # Note: In get_global_B/C, k=2 corresponds to the 3rd column of the 2x3 block.
    inner_schoolbook = [
        (([(0,0,1)], [(0,2,1)], [(0,2,1)])),
        (([(0,1,1)], [(1,2,1)], [(0,2,1)])),
        (([(1,0,1)], [(0,2,1)], [(1,2,1)])),
        (([(1,1,1)], [(1,2,1)], [(1,2,1)]))
    ]

    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_strassen):
            r_idx = base_rank + j
            for (r, s, ca) in A_inner:
                for (b, cb) in A_outer:
                    I, J = get_global_A(b, r, s)
                    U[I * m + J][r_idx] += ca * cb
            for (s, k, cb) in B_inner:
                for (b, cb2) in B_outer:
                    J, K = get_global_B(b, s, k)
                    V[J * p + K][r_idx] += cb * cb2
            for (r, k, cc) in C_inner:
                for (b, cb2) in C_outer:
                    I, K = get_global_C(b, r, k)
                    W[K * n + I][r_idx] += cc * cb2

        for j, (A_inner, B_inner, C_inner) in enumerate(inner_schoolbook):
            r_idx = base_rank + 7 + j
            for (r, s, ca) in A_inner:
                for (b, cb) in A_outer:
                    I, J = get_global_A(b, r, s)
                    U[I * m + J][r_idx] += ca * cb
            for (s, k, cb) in B_inner:
                for (b, cb2) in B_outer:
                    J, K = get_global_B(b, s, k)
                    V[J * p + K][r_idx] += cb * cb2
            for (r, k, cc) in C_inner:
                for (b, cb2) in C_outer:
                    I, K = get_global_C(b, r, k)
                    W[K * n + I][r_idx] += cc * cb2

        if time.time() > deadline - 20:
            break

    # Verification
    if time.time() < deadline - 5:
        Um = np.array(U)
        Vm = np.array(V)
        Wm = np.array(W)
        T = np.zeros((n, m, p))
        for r in range(rank):
            u = Um[:, r]
            v = Vm[:, r]
            w = Wm[:, r]
            for i in range(n):
                for j in range(m):
                    if u[i*m+j] == 0:
                        continue
                    for k in range(p):
                        T[i, j, k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        if not np.allclose(T, 1.0):
            # Fallback to schoolbook rank 96
            rank2 = n * m * p
            U2 = [[0] * rank2 for _ in range(n * m)]
            V2 = [[0] * rank2 for _ in range(m * p)]
            W2 = [[0] * rank2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i*m+j][r] = 1
                        V2[j*p+k][r] = 1
                        W2[k*n+i][r] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
