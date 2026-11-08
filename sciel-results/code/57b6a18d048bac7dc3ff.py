import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = 4, 4, 7

    # Precompute the 7 Strassen basis tensors for 2x2 (local coords)
    # Each basis: (u_local, v_local, w_local) as flat lists of length 4
    def build_strassen_basis():
        basis = []
        # (oa, ob, oc) each a list of (idx, coeff)
        outer = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        for oa, ob, oc in outer:
            u = [0.0]*4
            v = [0.0]*4
            w = [0.0]*4
            for idx, c in oa:
                u[idx] += c
            for idx, c in ob:
                v[idx] += c
            for idx, c in oc:
                w[idx] += c
            basis.append((u, v, w))
        return basis

    strassen_basis = build_strassen_basis()

    def time_ok():
        return time.time() < deadline

    def assemble(u_list, v_list, w_list):
        """Build U, V, W as lists of lists."""
        rank = len(u_list)
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        for c in range(rank):
            u = u_list[c]
            v = v_list[c]
            w = w_list[c]
            for r in range(n*m):
                U[r][c] = u[r]
            for r in range(m*p):
                V[r][c] = v[r]
            for r in range(p*n):
                W[r][c] = w[r]
        return {"U": U, "V": V, "W": W}

    def schoolbook_fallback():
        """Pure schoolbook: rank = n*m*p = 112."""
        u_list = []
        v_list = []
        w_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0]*(n*m)
                    v = [0.0]*(m*p)
                    w = [0.0]*(p*n)
                    u[i*m + j] = 1.0
                    v[j*p + k] = 1.0
                    w[k*n + i] = 1.0
                    u_list.append(u)
                    v_list.append(v)
                    w_list.append(w)
        return assemble(u_list, v_list, w_list)

    def build_strassen_head():
        """
        4x4 * 4x4 multiplication via recursive Strassen.
        A is 4x4, B is 4x4 (cols 0-3), C is 4x4 (cols 0-3).
        Split into 2x2 blocks. Each 2x2 block uses 7 Strassen terms.
        Total: 7 (outer) * 7 (inner) = 49 terms.
        """
        u_list = []
        v_list = []
        w_list = []

        # Outer Strassen: 4x4 blocks split into 2x2 sub-blocks
        # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        # For A: block (br,bc) -> rows br*2..br*2+1, cols bc*2..bc*2+1
        # For B: block (br,bc) -> rows br*2..br*2+1, cols bc*2..bc*2+1 (cols 0-3)
        # For C: block (br,bc) -> rows br*2..br*2+1, cols bc*2..bc*2+1 (cols 0-3)

        outer = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]

        for oa, ob, oc in outer:
            for (ua, va, wa) in strassen_basis:
                u = [0.0]*(n*m)
                v = [0.0]*(m*p)
                w = [0.0]*(p*n)

                # Fill u: A matrix indices
                for (a_blk, a_coeff) in oa:
                    br = a_blk // 2
                    bc = a_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        row = br*2 + lr
                        col = bc*2 + lc
                        idx = row * m + col
                        u[idx] += a_coeff * ua[loc]

                # Fill v: B matrix indices (cols 0-3)
                for (b_blk, b_coeff) in ob:
                    br = b_blk // 2
                    bc = b_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        row = br*2 + lr
                        col = bc*2 + lc
                        idx = row * p + col
                        v[idx] += b_coeff * va[loc]

                # Fill w: C matrix indices (cols 0-3), trace convention W[k*n+i]
                for (c_blk, c_coeff) in oc:
                    br = c_blk // 2
                    bc = c_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        i = br*2 + lr
                        k = bc*2 + lc
                        idx = k * n + i
                        w[idx] += c_coeff * wa[loc]

                u_list.append(u)
                v_list.append(v)
                w_list.append(w)

                if not time_ok():
                    return None

        return (u_list, v_list, w_list)

    def build_strassen_mid(k_off):
        """
        4x4 * 4x2 multiplication for columns k_off, k_off+1.
        A is 4x4, B is 4x2 (2 columns), C is 4x2 (2 columns).
        Use Strassen on the 4x4 A and 4x2 B/C.
        Split A into 2x2 blocks (4 blocks).
        Split B into 2x1 blocks vertically (2 blocks of 2x1).
        Actually, let's use a simpler approach:
        Treat B as having 2 columns. We can use the 4x2 decomposition.
        
        Alternative: use 4x2 via 2x2 Strassen applied to row-pairs.
        For each pair of rows (0,1) and (2,3) of A, and for each pair of columns of B...
        
        Simpler: use the same outer Strassen structure but with B and C having only 2 columns.
        The 4x2 B can be seen as two 2x1 column blocks.
        Let's use: outer Strassen on the 4x4 A (7 terms), and for each term,
        the inner product is 2x2 * 2x1... 
        
        Actually, the cleanest: treat this as 4x4 * 4x2 where we split into:
        A = [[A00, A01],[A10, A11]] (2x2 blocks of 2x2)
        B = [[B0],[B1]] (2 blocks of 2x2, but only 2 columns so each is 2x2)
        Wait, B is 4x2. Split into top 2x2 and bottom 2x2.
        C = [[C0],[C1]] similarly.
        
        Use Strassen: 7 outer terms, each involving 2x2 * 2x2 -> 2x2.
        But B and C only have 2 columns, so the "2x2" blocks of B are 2x2 (full).
        This works! Each outer term: (2x2 A-block) * (2x2 B-block) -> (2x2 C-block)
        Inner: 7 Strassen terms for 2x2*2x2.
        Total: 7 * 7 = 49... but we only need 4x2 output.
        
        Hmm, but B is 4x2, so B0 is 2x2 (rows 0-1, cols 0-1) and B1 is 2x2 (rows 2-3, cols 0-1).
        C0 is 2x2 (rows 0-1, cols 0-1) and C1 is 2x2 (rows 2-3, cols 0-1).
        So the output is 4x2, which is two 2x2 blocks. This works with the standard Strassen!
        Total: 7*7 = 49 terms. But we can do better.
        
        Actually, for 4x4 * 4x2, we can use a more efficient decomposition.
        Let's just use the 49-term approach for simplicity and correctness.
        """
        u_list = []
        v_list = []
        w_list = []

        outer = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
            ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]

        for oa, ob, oc in outer:
            for (ua, va, wa) in strassen_basis:
                u = [0.0]*(n*m)
                v = [0.0]*(m*p)
                w = [0.0]*(p*n)

                for (a_blk, a_coeff) in oa:
                    br = a_blk // 2
                    bc = a_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        row = br*2 + lr
                        col = bc*2 + lc
                        idx = row * m + col
                        u[idx] += a_coeff * ua[loc]

                for (b_blk, b_coeff) in ob:
                    br = b_blk // 2
                    bc = b_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        row = br*2 + lr
                        col = bc*2 + lc
                        idx = row * p + (col + k_off)
                        v[idx] += b_coeff * va[loc]

                for (c_blk, c_coeff) in oc:
                    br = c_blk // 2
                    bc = c_blk % 2
                    for (loc, loc_coeff) in [(0,1),(1,1),(2,1),(3,1)]:
                        lr = loc // 2
                        lc = loc % 2
                        i = br*2 + lr
                        k = bc*2 + lc
                        idx = (k + k_off) * n + i
                        w[idx] += c_coeff * wa[loc]

                u_list.append(u)
                v_list.append(v)
                w_list.append(w)

                if not time_ok():
                    return None

        return (u_list, v_list, w_list)

    def build_tail_1col(k_off):
        """
        4x4 * 4x1 multiplication for column k_off.
        Schoolbook: 16 terms.
        """
        u_list = []
        v_list = []
        w_list = []
        for i in range(n):
            for j in range(m):
                u = [0.0]*(n*m)
                v = [0.0]*(m*p)
                w = [0.0]*(p*n)
                u[i*m + j] = 1.0
                v[j*p + k_off] = 1.0
                w[k_off*n + i] = 1.0
                u_list.append(u)
                v_list.append(v)
                w_list.append(w)
        return (u_list, v_list, w_list)

    # Build all parts
    head = build_strassen_head()
    if head is None:
        return schoolbook_fallback()

    mid = build_strassen_mid(4)
    if mid is None:
        return schoolbook_fallback()

    tail = build_tail_1col(6)

    u_list = head[0] + mid[0] + tail[0]
    v_list = head[1] + mid[1] + tail[1]
    w_list = head[2] + mid[2] + tail[2]

    if not time_ok():
        return schoolbook_fallback()

    return assemble(u_list, v_list, w_list)
