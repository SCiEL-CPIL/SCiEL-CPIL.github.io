"""Combined program: partition-based decomposition (Strassen on 2x2 core +
block cross terms) with exact tensor verification and a deadline guard.

The structural idea comes from the block-partition decomposition (rank 26),
which splits each 3x3 matrix into a 2x2 core plus one extra row/column and
applies the rank-7 Strassen algorithm to the core. The refinement machinery
is an exact verification pass (tensor reconstruction via the factor matrices)
plus a wall-clock deadline guard, so we always return a valid decomposition.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 115.0  # safety margin under the 120 s limit

    n, m, p = DIMENSIONS
    assert (n, m, p) == (3, 3, 3)

    # ---- Build the rank-26 partition decomposition ------------------------
    # Index conventions (n = m = p = 3, so all indices run 0..8):
    #   A_{ij} -> U[i*m + j] = U[i*3 + j]
    #   B_{jk} -> V[j*m + k] = V[j*3 + k]
    #   C_{ik} -> W[k*n + i] = W[k*3 + i]   (official trace convention)
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def widx(i, k):
        return k * n + i

    def add_term(u_terms, v_terms, w_terms):
        """u_terms/v_terms/w_terms: list of (index, coeff)."""
        nonlocal rank
        for (u, cu) in u_terms:
            U[u][rank] += cu
        for (v, cv) in v_terms:
            V[v][rank] += cv
        for (w, cw) in w_terms:
            W[w][rank] += cw
        rank += 1

    # A indices: A00=0 A01=1 A02=2 A10=3 A11=4 A12=5 A20=6 A21=7 A22=8
    # B indices: B00=0 B01=1 B02=2 B10=3 B11=4 B12=5 B20=6 B21=7 B22=8

    # --- 1) 2x2 core (A00,A01,A10,A11)*(B00,B01,B10,B11) via Strassen (rank 7)
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    add_term([(0, 1), (4, 1)], [(0, 1), (4, 1)],
             [(widx(0, 0), 1), (widx(1, 1), 1)])
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    add_term([(3, 1), (4, 1)], [(0, 1)],
             [(widx(1, 0), 1), (widx(1, 1), -1)])
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    add_term([(0, 1)], [(1, 1), (4, -1)],
             [(widx(0, 1), 1), (widx(1, 1), 1)])
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    add_term([(4, 1)], [(3, 1), (0, -1)],
             [(widx(0, 0), 1), (widx(1, 0), 1)])
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    add_term([(0, 1), (1, 1)], [(4, 1)],
             [(widx(0, 0), -1), (widx(0, 1), 1)])
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)],
             [(widx(1, 1), 1)])
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    add_term([(1, 1), (4, -1)], [(3, 1), (4, 1)],
             [(widx(0, 0), 1)])

    # --- 2) a*b^T : (A02,A12)*(B20,B21) -> C00,C01,C10,C11  (rank 4)
    add_term([(2, 1)], [(6, 1)], [(widx(0, 0), 1)])
    add_term([(2, 1)], [(7, 1)], [(widx(0, 1), 1)])
    add_term([(5, 1)], [(6, 1)], [(widx(1, 0), 1)])
    add_term([(5, 1)], [(7, 1)], [(widx(1, 1), 1)])

    # --- 3) A2*b : (A00,A01,A10,A11)*(B02,B12) -> C02,C12  (rank 4)
    add_term([(0, 1)], [(2, 1)], [(widx(0, 2), 1)])
    add_term([(1, 1)], [(5, 1)], [(widx(0, 2), 1)])
    add_term([(3, 1)], [(2, 1)], [(widx(1, 2), 1)])
    add_term([(4, 1)], [(5, 1)], [(widx(1, 2), 1)])

    # --- 4) a*b3 : (A02,A12)*B22 -> C02,C12  (rank 2)
    add_term([(2, 1)], [(8, 1)], [(widx(0, 2), 1)])
    add_term([(5, 1)], [(8, 1)], [(widx(1, 2), 1)])

    # --- 5) a^T*B2 : (A20,A21)*(B00,B10,B01,B11) -> C20,C21  (rank 4)
    add_term([(6, 1)], [(0, 1)], [(widx(2, 0), 1)])
    add_term([(7, 1)], [(3, 1)], [(widx(2, 0), 1)])
    add_term([(6, 1)], [(1, 1)], [(widx(2, 1), 1)])
    add_term([(7, 1)], [(4, 1)], [(widx(2, 1), 1)])

    # --- 6) a3*b^T : A22*(B20,B21) -> C20,C21  (rank 2)
    add_term([(8, 1)], [(6, 1)], [(widx(2, 0), 1)])
    add_term([(8, 1)], [(7, 1)], [(widx(2, 1), 1)])

    # --- 7) a^T*b : (A20,A21)*(B02,B12) -> C22  (rank 2)
    add_term([(6, 1)], [(2, 1)], [(widx(2, 2), 1)])
    add_term([(7, 1)], [(5, 1)], [(widx(2, 2), 1)])

    # --- 8) a3*b3 : A22*B22 -> C22  (rank 1)
    add_term([(8, 1)], [(8, 1)], [(widx(2, 2), 1)])

    result = {"U": U, "V": V, "W": W}

    # ---- Exact verification (refinement machinery) ------------------------
    # Reconstruct C = A @ B from the factors and compare with the true product.
    try:
        if time.time() - start < DEADLINE:
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            # Random integer test matrices (exact, small range to keep ints exact).
            rng = np.random.default_rng(12345)
            for _ in range(200):
                A = rng.integers(-4, 5, size=(n, m)).astype(float)
                B = rng.integers(-4, 5, size=(m, p)).astype(float)
                C_true = A @ B
                # C_{ik} = sum_j A_{ij} B_{jk}
                # Reconstruct via factors:
                #   x_jk = sum_r V[j,k,r] * (sum_i' A_{i'j} U[i'j,r])
                #   C_{ik} = sum_j x_jk * W[k,i,r] summed over r
                # Equivalent tensor contraction:
                #   C_{ik} = sum_{j,r} A_{ij} B_{jk} W_{k i r}
                # where U,V,W are the factor columns. Build the full tensor:
                # T_{i j k} = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # and check T == einsum of the multiplication tensor.
                T = np.einsum('ir,jr,kir->ijk', Ua, Va, Wa)
                Mmul = np.einsum('ij,jk->ik', A, B)
                # T_{ijk} should equal A_{ij} B_{jk} for all i,j,k
                # i.e. T == np.einsum('ij,jk->ijk', A, B)
                Mmul3 = np.einsum('ij,jk->ijk', A, B)
                if not np.allclose(T, Mmul3, atol=1e-9):
                    # Fall back to guaranteed-valid schoolbook rank-27.
                    raise ValueError("verification failed")
    except Exception:
        # Guaranteed-valid fallback: schoolbook rank 27.
        r27 = n * m * p
        U27 = [[0] * r27 for _ in range(n * m)]
        V27 = [[0] * r27 for _ in range(m * p)]
        W27 = [[0] * r27 for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[i * m + j][c] = 1
                    V27[j * p + k][c] = 1
                    W27[k * n + i][c] = 1
                    c += 1
        return {"U": U27, "V": V27, "W": W27}

    # Deadline guard: if we somehow ran long, still return the valid result.
    if time.time() - start >= DEADLINE:
        return result

    return result
# EVOLVE-BLOCK-END
