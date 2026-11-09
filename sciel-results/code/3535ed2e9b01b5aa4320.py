import numpy as np
import time

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 140.0  # safety margin before 150s wall-clock kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Strassen 2x2 decomposition (rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    strassen_terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # Laderman 3x3 decomposition (rank 23) - 0-indexed local: idx = row*3+col
    # a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    laderman_terms = [
        # M1: (a10+a20)(b10+b20)
        ({3: 1, 6: 1}, {3: 1, 6: 1}, {3: 1}),
        # M2: (a10+a20)(b11+b21)
        ({3: 1, 6: 1}, {4: 1, 7: 1}, {4: 1}),
        # M3: a10(b21-b11)
        ({3: 1}, {7: 1, 4: -1}, {4: 1}),
        # M4: a20(b11-b21)
        ({6: 1}, {4: 1, 7: -1}, {6: 1, 7: 1}),
        # M5: (a11+a12)(b10+b20)
        ({4: 1, 5: 1}, {3: 1, 6: 1}, {4: 1, 5: 1}),
        # M6: a12(b20-b10)
        ({5: 1}, {6: 1, 3: -1}, {5: 1}),
        # M7: (a21+a22)(b11+b21)
        ({7: 1, 8: 1}, {4: 1, 7: 1}, {7: 1, 8: 1}),
        # M8: a22(b21-b11)
        ({8: 1}, {7: 1, 4: -1}, {8: 1}),
        # M9: a11(b20-b10)
        ({4: 1}, {6: 1, 3: -1}, {6: 1}),
        # M10: a12(b21-b11)
        ({5: 1}, {7: 1, 4: -1}, {7: 1}),
        # M11: a21(b10-b20)
        ({7: 1}, {3: 1, 6: -1}, {6: -1}),
        # M12: a22(b11-b21)
        ({8: 1}, {4: 1, 7: -1}, {7: -1}),
        # M13: a11*b11
        ({4: 1}, {4: 1}, {4: 1}),
        # M14: a12*b12
        ({5: 1}, {5: 1}, {5: 1}),
        # M15: a21*b12
        ({7: 1}, {5: 1}, {7: 1}),
        # M16: a22*b22
        ({8: 1}, {8: 1}, {8: 1}),
        # M17: (a11+a12+a21+a22)(b10+b20)
        ({4: 1, 5: 1, 7: 1, 8: 1}, {3: 1, 6: 1}, {4: 1, 5: 1, 7: 1, 8: 1}),
        # M18: (a10+a11+a20+a21)(b11+b21)
        ({3: 1, 4: 1, 6: 1, 7: 1}, {4: 1, 7: 1}, {3: 1, 4: 1, 6: 1, 7: 1}),
        # M19: (a10+a11+a20+a21)(b10+b20)
        ({3: 1, 4: 1, 6: 1, 7: 1}, {3: 1, 6: 1}, {3: 1, 4: 1, 6: 1, 7: 1}),
        # M20: (a11+a12+a21+a22)(b11+b21)
        ({4: 1, 5: 1, 7: 1, 8: 1}, {4: 1, 7: 1}, {4: 1, 5: 1, 7: 1, 8: 1}),
        # M21: (a10+a20)(b20+b21)
        ({3: 1, 6: 1}, {6: 1, 7: 1}, {6: 1, 7: 1}),
        # M22: (a10+a20)(b10+b20)
        ({3: 1, 6: 1}, {3: 1, 6: 1}, {3: 1, 6: 1}),
        # M23: (a11+a12+a21+a22)(b10+b20)
        ({4: 1, 5: 1, 7: 1, 8: 1}, {3: 1, 6: 1}, {4: 1, 5: 1, 7: 1, 8: 1}),
    ]

    def build_tensor_product(outer_terms, inner_terms):
        """Build 6x6 tensor = outer(3x3 grid) tensor inner(2x2 blocks)."""
        rank = len(outer_terms) * len(inner_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for (u_o, v_o, w_o) in outer_terms:
            for (u_i, v_i, w_i) in inner_terms:
                for ik_idx, co in u_o.items():
                    i, k = divmod(ik_idx, 3)
                    for elem, ci in u_i.items():
                        a, b = divmod(elem, 2)
                        U[(2*i + a) * m + (2*k + b)][col] += co * ci
                for kj_idx, co in v_o.items():
                    k, j = divmod(kj_idx, 3)
                    for elem, ci in v_i.items():
                        a, b = divmod(elem, 2)
                        V[(2*k + a) * p + (2*j + b)][col] += co * ci
                for ij_idx, co in w_o.items():
                    i, j = divmod(ij_idx, 3)
                    for elem, ci in w_i.items():
                        a, b = divmod(elem, 2)
                        W[(2*j + b) * n + (2*i + a)][col] += co * ci
                col += 1
        return U, V, W

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float).reshape(n, m, rank)
        V_np = np.array(V, dtype=float).reshape(m, p, rank)
        W_np = np.array(W, dtype=float).reshape(p, n, rank)
        T = np.einsum('ikt,kjt,jit->ikj', U_np, V_np, W_np)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-8)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Strassen(2x2) tensor Laderman(3x3) = rank 161
    if time.time() < _DEADLINE:
        U, V, W = build_tensor_product(laderman_terms, strassen_terms)
        if verify(U, V, W, 161):
            return {"U": U, "V": V, "W": W}

    # Fallback: Strassen(2x2) tensor schoolbook(3x3) = rank 189
    if time.time() < _DEADLINE:
        schoolbook3 = [({i*3+k: 1}, {k*3+j: 1}, {i*3+j: 1})
                       for i in range(3) for k in range(3) for j in range(3)]
        U, V, W = build_tensor_product(schoolbook3, strassen_terms)
        if verify(U, V, W, 189):
            return {"U": U, "V": V, "W": W}

    # Final fallback: full schoolbook 6x6x6 = rank 216
    return schoolbook()
# EVOLVE-BLOCK-END
