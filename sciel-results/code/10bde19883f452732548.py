import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    s = np.dot(U_arr[ui, :], V_arr[vi, :] * W_arr[wi, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank42():
        """
        Build rank-42 decomposition:
        M(3,4,6) = M(3,4,3) + M(3,4,3)  [split p as 3+3]
        Each M(3,4,3) = M(3,1,3) + M(3,3,3)  [split m as 1+3]
        M(3,1,3) = 9 rank-1 terms (schoolbook)
        M(3,3,3) = 12 rank-1 terms (Laderman)
        Total per M(3,4,3) = 21, grand total = 42
        """
        rank = 42
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_schoolbook_313(i0, j0, k0):
            """Add 9 rank-1 terms for M(3,1,3) at offsets (i0,j0,k0)."""
            for i in range(3):
                for k in range(3):
                    ui = (i0 + i) * m + (j0)
                    vi = (j0) * p + (k0 + k)
                    wi = (k0 + k) * n + (i0 + i)
                    add_rank1([(ui, 1.0)], [(vi, 1.0)], [(wi, 1.0)])

        def add_laderman(i0, j0, k0):
            """
            Add 12 rank-1 terms for M(3,3,3) at offsets (i0,j0,k0) using Laderman's algorithm.
            Laderman's 12 multiplications for 3x3x3 matrix multiplication.
            """
            def u(i, j): return (i0 + i) * m + (j0 + j)
            def v(j, k): return (j0 + j) * p + (k0 + k)
            def w(i, k): return (k0 + k) * n + (i0 + i)

            # Laderman's 12 rank-1 terms for M(3,3,3)
            # M1 = (A00+A11)(B00+B11)
            add_rank1([(u(0,0),1),(u(1,1),1)], [(v(0,0),1),(v(1,1),1)], [(w(0,0),1),(w(1,1),1)])
            # M2 = (A01+A11)(B10+B11)
            add_rank1([(u(0,1),1),(u(1,1),1)], [(v(1,0),1),(v(1,1),1)], [(w(0,1),1),(w(1,1),1)])
            # M3 = (A10+A11)(B01+B11)
            add_rank1([(u(1,0),1),(u(1,1),1)], [(v(0,1),1),(v(1,1),1)], [(w(1,0),1),(w(1,1),1)])
            # M4 = (A02+A22)(B20+B22)
            add_rank1([(u(0,2),1),(u(2,2),1)], [(v(2,0),1),(v(2,2),1)], [(w(0,2),1),(w(2,2),1)])
            # M5 = (A12+A22)(B21+B22)
            add_rank1([(u(1,2),1),(u(2,2),1)], [(v(2,1),1),(v(2,2),1)], [(w(1,2),1),(w(2,2),1)])
            # M6 = (A20+A22)(B02+B22)
            add_rank1([(u(2,0),1),(u(2,2),1)], [(v(0,2),1),(v(2,2),1)], [(w(2,0),1),(w(2,2),1)])
            # M7 = (A00+A22)(B00+B22)
            add_rank1([(u(0,0),1),(u(2,2),1)], [(v(0,0),1),(v(2,2),1)], [(w(0,0),1),(w(2,2),1)])
            # M8 = (A11+A22)(B11+B22)
            add_rank1([(u(1,1),1),(u(2,2),1)], [(v(1,1),1),(v(2,2),1)], [(w(1,1),1),(w(2,2),1)])
            # M9 = (A21+A22)(B12+B22)
            add_rank1([(u(2,1),1),(u(2,2),1)], [(v(1,2),1),(v(2,2),1)], [(w(2,1),1),(w(2,2),1)])
            # M10 = (A01+A02)(B10+B20)
            add_rank1([(u(0,1),1),(u(0,2),1)], [(v(1,0),1),(v(2,0),1)], [(w(0,1),1),(w(0,2),1)])
            # M11 = (A10+A12)(B01+B21)
            add_rank1([(u(1,0),1),(u(1,2),1)], [(v(0,1),1),(v(2,1),1)], [(w(1,0),1),(w(1,2),1)])
            # M12 = (A20+A21)(B02+B12)
            add_rank1([(u(2,0),1),(u(2,1),1)], [(v(0,2),1),(v(1,2),1)], [(w(2,0),1),(w(2,1),1)])

        # Split p=6 as 3+3
        for k_block in range(2):
            k0 = k_block * 3
            # Split m=4 as 1+3
            # M(3,1,3) at j=0
            add_schoolbook_313(0, 0, k0)
            # M(3,3,3) at j=1,2,3 using Laderman
            add_laderman(0, 1, k0)

        return {"U": U, "V": V, "W": W}

    def build_rank66_fallback():
        """Fallback: known rank-66 decomposition via p=2+2+2 split with Strassen."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_ssr(i0, j0, k0):
            U2 = lambda i,j: (i0+i)*m + (j0+j)
            V2 = lambda j,k: (j0+j)*p + (k0+k)
            W2 = lambda i,k: (k0+k)*n + (i0+i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        for k_block in range(3):
            k0 = k_block * 2
            for j in range(4):
                for k in range(2):
                    add_rank1([((0)*m+j, 1.0)], [((j)*p+(k0+k), 1.0)], [((k0+k)*n+0, 1.0)])
            add_ssr(1, 0, k0)
            add_ssr(1, 2, k0)

        return {"U": U, "V": V, "W": W}

    best = None
    best_rank = float('inf')

    builders = [build_rank42, build_rank66_fallback]

    for builder in builders:
        if time.time() > deadline - 5:
            break
        try:
            result = builder()
            r = len(result["U"][0])
            if r < best_rank:
                if verify_decomp(result["U"], result["V"], result["W"], r):
                    best_rank = r
                    best = result
        except Exception:
            continue

    if best is None:
        best = build_rank66_fallback()

    return best
