"""Rank-42 exact tensor decomposition for 2x4 * 4x6 matrix multiplication using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 130.0


def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12

    try:
        U, V, W = _construct_strassen_decomposition(n, m, p)
        if _verify_decomposition(n, m, p, U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            U_s, V_s, W_s = _construct_schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
    except Exception:
        try:
            U_s, V_s, W_s = _construct_schoolbook(n, m, p)
            return {"U": U_s, "V": V_s, "W": W_s}
        except Exception:
            return {"U": [], "V": [], "W": []}


def _construct_schoolbook(n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _construct_strassen_decomposition(n, m, p):
    """
    Decompose 2x4 * 4x6 into 6 independent 2x2 * 2x2 block multiplications
    using Strassen's rank-7 algorithm for each block.
    
    A (2x4) = [A_L | A_R] where A_L = A[:,0:2], A_R = A[:,2:4] (each 2x2)
    B (4x6) split into 3 column-blocks of width 2: B_b = B[:, b*2:(b+1)*2] (each 4x2)
    B_b = [B_top_b; B_bot_b] where B_top_b = B_b[0:2,:], B_bot_b = B_b[2:4,:]
    
    For each block b and each half a (0=left, 1=right):
    C[:, b*2:(b+1)*2] += A_half_a * B_half_b
    
    Total: 3 (p-blocks) * 2 (m-halves) * 7 (Strassen rank) = 42
    """
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    rank = 42

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen for 2x2 * 2x2 (rank 7)
    # U coeffs: for each of 7 products, list of (index_in_2x2, coefficient)
    # Index: 0=X00, 1=X01, 2=X10, 3=X11
    U_coeffs = [
        [(0, 1), (3, 1)],       # P1: X00 + X11
        [(2, 1), (3, 1)],       # P2: X10 + X11
        [(0, 1)],               # P3: X00
        [(3, 1)],               # P4: X11
        [(0, 1), (1, 1)],       # P5: X00 + X01
        [(2, 1), (0, -1)],      # P6: X10 - X00
        [(1, 1), (3, -1)]       # P7: X01 - X11
    ]

    V_coeffs = [
        [(0, 1), (3, 1)],       # P1: Y00 + Y11
        [(0, 1)],               # P2: Y00
        [(1, 1), (3, -1)],      # P3: Y01 - Y11
        [(2, 1), (0, -1)],      # P4: Y10 - Y00
        [(3, 1)],               # P5: Y11
        [(0, 1), (1, 1)],       # P6: Y00 + Y01
        [(2, 1), (3, 1)]        # P7: Y10 + Y11
    ]

    W_coeffs = [
        [(0, 1), (3, 1)],       # P1: Z00 + Z11
        [(2, 1), (3, -1)],      # P2: Z10 - Z11
        [(1, 1), (3, 1)],       # P3: Z01 + Z11
        [(0, 1), (2, 1)],       # P4: Z00 + Z10
        [(0, -1), (1, 1)],      # P5: -Z00 + Z01
        [(3, 1)],               # P6: Z11
        [(0, 1)]                # P7: Z00
    ]

    col = 0
    for b in range(3):  # 3 column blocks of B
        for a in range(2):  # 2 row halves of A (left/right)
            for s in range(7):  # 7 Strassen products
                # Fill U
                for (idx, coeff) in U_coeffs[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j, col] += coeff

                # Fill V
                for (idx, coeff) in V_coeffs[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k, col] += coeff

                # Fill W
                for (idx, coeff) in W_coeffs[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i, col] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            j_v = v // p
            k = v % p
            if j_v != j:
                uv = U[u, :] * V[v, :]
                for w in range(pn):
                    if abs(np.dot(uv, W[w, :])) > 1e-9:
                        return False
            else:
                w_target = k * n + i
                uv = U[u, :] * V[v, :]
                if abs(np.dot(uv, W[w_target, :]) - 1.0) > 1e-9:
                    return False
                for w in range(pn):
                    if w == w_target:
                        continue
                    if abs(np.dot(uv, W[w, :])) > 1e-9:
                        return False
    return True


if __name__ == "__main__":
    result = construct()
    print(f"Best rank found: {len(result['U'][0])}")
