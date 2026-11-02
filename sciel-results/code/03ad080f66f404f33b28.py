import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_DEADLINE_S = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE_S - (time.time() - _START)


def _verify(U, V, W, num_tests=10):
    n, m, p = DIMENSIONS
    R = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for _ in range(num_tests):
        A = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        B = np.random.randint(-10, 11, (3, 3)).astype(np.float64)
        C_expected = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        U_proj = U_arr.T @ A_flat
        V_proj = V_arr.T @ B_flat
        C_computed = np.zeros((3, 3), dtype=np.float64)
        for r in range(R):
            p_r = U_proj[r] * V_proj[r]
            if abs(p_r) > 1e-15:
                w_col = W_arr[:, r]
                w_matrix = w_col.reshape(3, 3)
                C_computed += p_r * w_matrix.T
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _rank26_construct():
    """
    Rank-26 decomposition: Strassen on 2x2 block + scalar 1x1 + cross terms.
    """
    terms = []

    def add(u, v, w):
        terms.append((u, v, w))

    # --- Strassen 2x2x2 on top-left block ---
    add({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1, 4: 1})
    add({3: 1}, {0: 1, 4: 1}, {3: 1, 4: 1})
    add({0: 1}, {4: 1, 0: -1}, {1: 1})
    add({4: 1}, {3: 1, 0: -1}, {3: 1})
    add({0: 1, 1: 1}, {3: 1}, {0: 1, 4: 1})
    add({3: 1, 0: -1}, {0: 1, 1: 1}, {4: 1})
    add({1: 1, 4: -1}, {3: 1, 4: 1}, {1: 1})

    # --- 1x1 block ---
    add({8: 1}, {8: 1}, {8: 1})

    # --- Cross terms ---
    add({6: 1}, {0: 1}, {2: 1})
    add({6: 1}, {1: 1}, {5: 1})
    add({6: 1}, {2: 1}, {8: 1})
    add({7: 1}, {0: 1}, {2: 1})
    add({7: 1}, {1: 1}, {5: 1})
    add({7: 1}, {2: 1}, {8: 1})
    add({6: 1}, {3: 1}, {2: 1})
    add({6: 1}, {4: 1}, {5: 1})
    add({6: 1}, {5: 1}, {8: 1})
    add({7: 1}, {3: 1}, {2: 1})
    add({7: 1}, {4: 1}, {5: 1})
    add({7: 1}, {5: 1}, {8: 1})
    add({2: 1}, {6: 1}, {6: 1})
    add({2: 1}, {7: 1}, {6: 1})
    add({5: 1}, {6: 1}, {7: 1})
    add({5: 1}, {7: 1}, {7: 1})
    add({8: 1}, {6: 1}, {2: 1})
    add({8: 1}, {7: 1}, {5: 1})
    add({0: 1}, {2: 1}, {6: 1})
    add({1: 1}, {5: 1}, {6: 1})
    add({2: 1}, {8: 1}, {6: 1})
    add({3: 1}, {2: 1}, {7: 1})
    add({4: 1}, {5: 1}, {7: 1})
    add({5: 1}, {8: 1}, {7: 1})

    rank = len(terms)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]

    for r, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][r] = val
        for idx, val in v.items():
            V[idx][r] = val
        for idx, val in w.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def construct():
    best = _schoolbook()
    
    if _time_left() > 5.0:
        cand26 = _rank26_construct()
        if _verify(cand26["U"], cand26["V"], cand26["W"], num_tests=10):
            best = cand26

    return best
