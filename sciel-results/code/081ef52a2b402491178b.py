import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 120.0

    def create_decomposition():
        rank = 20
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        # Helper to map local indices to global indices
        # A is 2x4, B is 4x5, C is 2x5
        # Split m=4 into m1=1, m2=1, m3=2
        # Split p=5 into p1=5 (for m1), p2=5 (for m2), p3=2, p4=3 (for m3)
        
        # Block 1: m1=1 (col 0), p1=5 (cols 0-4)
        # A1 (2x1) * B1 (1x5) -> C1 (2x5)
        # Rank 1
        u = {0: 1, 2: 1} # A[0,0], A[1,0]
        v = {0: 1} # B[0,0] -> B[0,0..4]
        # Wait, B1 is 1x5. B[0,0], B[0,1], B[0,2], B[0,3], B[0,4]
        # The tensor term is sum_k A[i,0] B[0,k] C[i,k]
        # This is rank 1?
        # C[i,k] = A[i,0] B[0,k]
        # U_i = A[i,0], V_k = B[0,k], W_{ik} = C[i,k]
        # Yes.
        # u: {0: 1, 2: 1}
        # v: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
        # w: {0: 1, 1: 1, 2: 1, 3: 1, 4: 1} # C[0,0]..C[0,4], C[1,0]..C[1,4]
        # w indices: k*n + i.
        # k=0..4, i=0,1.
        # w: {0: 1, 2: 1, 1: 1, 3: 1, 4: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}
        # Wait, W[k*n+i, r].
        # C[0,0] -> 0. C[0,1] -> 1. C[0,2] -> 2. C[0,3] -> 3. C[0,4] -> 4.
        # C[1,0] -> 2. C[1,1] -> 3. C[1,2] -> 4. C[1,3] -> 5. C[1,4] -> 6.
        # My previous w dict was wrong.
        
        # Let's use a generic add_block function
        
        def add_block(m_start, m_len, p_start, p_len, rank_func):
            # m_start: starting col of A, row of B
            # p_start: starting col of B, row of C (wait, C is 2x5, so p is col of C)
            # rank_func: lambda u_idx, v_idx, w_idx: add_term(...)
            pass

        # 1. M_{2,1,5} (Rank 1)
        # A cols 0, B rows 0, C cols 0-4
        u = {0: 1, 2: 1}
        v = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
        w = {}
        for i in range(2):
            for k in range(5):
                w[k * 2 + i] = 1
        add_term(u, v, w)

        # 2. M_{2,1,5} (Rank 1)
        # A cols 1, B rows 1, C cols 0-4
        u = {1: 1, 3: 1}
        v = {5: 1, 6: 1, 7: 1, 8: 1, 9: 1} # B[1,0..4] -> indices 1*5+0..4
        w = {}
        for i in range(2):
            for k in range(5):
                w[k * 2 + i] = 1
        add_term(u, v, w)

        # 3. M_{2,2,2} (Rank 7)
        # A cols 2-3, B rows 2-3, C cols 0-1
        strassen_222 = [
            ({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}),
            ({2: 1, 1: -1}, {0: 1}, {2: 1, 1: 1}),
            ({0: 1}, {2: 1, 1: -1}, {0: 1, 2: 1}),
            ({1: 1}, {2: 1, 0: -1}, {1: 1, 2: -1}),
            ({0: 1, 3: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 3: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        # Map local (i,j) to global
        # A: i in 0..1, j in 0..1 (local) -> i, 2+j (global)
        # B: j in 0..1, k in 0..1 (local) -> 2+j, k (global, but p_start=0)
        # C: i in 0..1, k in 0..1 (local) -> k, i (global)
        
        for u_t, v_t, w_t in strassen_222:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * 4 + (2 + lj)] = val
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[(2 + lj) * 5 + lk] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[lk * 2 + li] = val
            add_term(u_g, v_g, w_g)

        # 4. M_{2,2,2} (Rank 7)
        # A cols 2-3, B rows 2-3, C cols 2-3
        for u_t, v_t, w_t in strassen_222:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * 4 + (2 + lj)] = val
            for idx, val in v_t.items():
                lj, lk = idx // 2, idx % 2
                v_g[(2 + lj) * 5 + (2 + lk)] = val
            for idx, val in w_t.items():
                li, lk = idx // 2, idx % 2
                w_g[(2 + lk) * 2 + li] = val
            add_term(u_g, v_g, w_g)

        # 5. M_{2,2,1} (Rank 4)
        # A cols 2-3, B rows 2-3, C col 4
        # C[i,4] = sum_j A[i,j] B[j,4]
        # j in 0..1 local -> 2..3 global
        # k=4
        terms = [
            ({0: 1}, {0: 1}, {0: 1}),
            ({1: 1}, {1: 1}, {0: 1}),
            ({2: 1}, {0: 1}, {1: 1}),
            ({3: 1}, {1: 1}, {1: 1}),
        ]
        for u_t, v_t, w_t in terms:
            u_g, v_g, w_g = {}, {}, {}
            for idx, val in u_t.items():
                li, lj = idx // 2, idx % 2
                u_g[li * 4 + (2 + lj)] = val
            for idx, val in v_t.items():
                lj = idx
                v_g[(2 + lj) * 5 + 4] = val
            for idx, val in w_t.items():
                li = idx
                w_g[4 * 2 + li] = val
            add_term(u_g, v_g, w_g)

        # 6. M_{2,2,1} (Rank 4)
        # A cols 2-3, B rows 2-3, C col 4?
        # Wait, I split p=5 into 2+2+1.
        # C cols 0,1 (M222), 2,3 (M222), 4 (M221).
        # I used M221 for C col 4.
        # Did I miss anything?
        # m=4.
        # m1=1 (col 0) -> M215
        # m2=1 (col 1) -> M215
        # m3=2 (cols 2,3) -> M225.
        # M225 = M222 (cols 0,1) + M222 (cols 2,3) + M221 (col 4).
        # Yes.
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    s = 0
                    for r in range(rank):
                        if U[a_idx][r] and V[b_idx][r] and W[c_idx][r]:
                            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
                    if s != 1:
                        return False
        return True

    best = create_decomposition()
    
    if verify(best["U"], best["V"], best["W"]):
        return best
    else:
        # Fallback
        return {"U": [[0]*40 for _ in range(8)], "V": [[0]*40 for _ in range(20)], "W": [[0]*40 for _ in range(10)]}
# EVOLVE-BLOCK-END
