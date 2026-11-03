import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = 4, 4, 6
    rank = 77

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Outer Strassen steps for 4x4 * 4x6 with 2x2 blocks (A) and 2x3 blocks (B, C)
    # Block IDs: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A blocks: 2x2 each. B blocks: 2x3 each. C blocks: 2x3 each.
    outer_steps = [
        # M1 = (A00+A11)*(B00+B11) -> C00+C11
        (([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)])),
        # M2 = (A10+A11)*B00 -> C10-C11
        (([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)])),
        # M3 = A00*(B01-B11) -> C01+C11
        (([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)])),
        # M4 = A11*(B10-B00) -> C00+C10
        (([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)])),
        # M5 = (A00+A01)*B11 -> C01-C00
        (([(0,1),(1,1)], [(3,1)], [(1,1),(0,-1)])),
        # M6 = (A10-A00)*(B00+B01) -> C11
        (([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)])),
        # M7 = (A01-A11)*(B10+B11) -> C00
        (([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]))
    ]

    # Inner decomposition for 2x2 * 2x3 -> 2x3
    # Strassen for first 2 columns (2x2*2x2), schoolbook for 3rd column (2x2*2x1)
    # Inner Strassen: A(r,s), B(s,k) with k in {0,1}, C(r,k) with k in {0,1}
    inner_strassen = [
        (([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])),
        (([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])),
        (([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])),
        (([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])),
        (([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(0,0,-1)])),
        (([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])),
        (([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]))
    ]

    # Schoolbook for 3rd column: A(r,s), B(s,0), C(r,0)
    # The local B column index 0 maps to global column offset 2 within the 2x3 block
    inner_schoolbook = [
        (([(0,0,1)], [(0,0,1)], [(0,0,1)])),
        (([(0,1,1)], [(1,0,1)], [(0,0,1)])),
        (([(1,0,1)], [(0,0,1)], [(1,0,1)])),
        (([(1,1,1)], [(1,0,1)], [(1,0,1)]))
    ]

    def global_A(block_id, r, s):
        """A block (2x2): block_id encodes (block_row, block_col)"""
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def global_B(block_id, s, k):
        """B block (2x3): s is local row (0,1), k is local col (0,1,2)"""
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def global_C(block_id, r, k):
        """C block (2x3): r is local row (0,1), k is local col (0,1,2)"""
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    for oi, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base = oi * 11

        # Inner Strassen terms (7)
        for ii, (A_in, B_in, C_in) in enumerate(inner_strassen):
            c = base + ii
            for (r, s, ca) in A_in:
                for (bid, cb) in A_outer:
                    I, J = global_A(bid, r, s)
                    U[I * m + J][c] += ca * cb
            for (s, k, cb) in B_in:
                for (bid, cc) in B_outer:
                    J, K = global_B(bid, s, k)
                    V[J * p + K][c] += cb * cc
            for (r, k, cc) in C_in:
                for (bid, cd) in C_outer:
                    I, K = global_C(bid, r, k)
                    W[K * n + I][c] += cc * cd

        # Inner Schoolbook terms (4) - 3rd column
        for ii, (A_in, B_in, C_in) in enumerate(inner_schoolbook):
            c = base + 7 + ii
            for (r, s, ca) in A_in:
                for (bid, cb) in A_outer:
                    I, J = global_A(bid, r, s)
                    U[I * m + J][c] += ca * cb
            for (s, k_local, cb) in B_in:
                # k_local=0 maps to global column offset 2 within the 2x3 block
                for (bid, cc) in B_outer:
                    J, K = global_B(bid, s, 2)
                    V[J * p + K][c] += cb * cc
            for (r, k_local, cc) in C_in:
                # k_local=0 maps to global column offset 2 within the 2x3 block
                for (bid, cd) in C_outer:
                    I, K = global_C(bid, r, 2)
                    W[K * n + I][c] += cc * cd

    # Verification with deadline guard
    if time.time() < deadline - 3.0:
        Um = np.array(U, dtype=np.float64)
        Vm = np.array(V, dtype=np.float64)
        Wm = np.array(W, dtype=np.float64)
        U_r = Um.reshape(n, m, rank)
        V_r = Vm.reshape(m, p, rank)
        W_r = Wm.reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        T_target = np.ones((n, m, p))
        if not np.allclose(T_recon, T_target, atol=1e-10):
            # Fallback: schoolbook
            r2 = n * m * p
            U2 = [[0] * r2 for _ in range(n * m)]
            V2 = [[0] * r2 for _ in range(m * p)]
            W2 = [[0] * r2 for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        U2[i * m + j][r] = 1
                        V2[j * p + k][r] = 1
                        W2[k * n + i][r] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
