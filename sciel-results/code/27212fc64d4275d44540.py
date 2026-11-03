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

        # Quick verification
        if _verify_decomposition(n, m, p, U, V, W):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
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
    Decompose 2x4 * 4x6 into 6 blocks of 2x2 * 2x2, each using Strassen (rank 7).
    Total rank = 6 * 7 = 42.
    
    A (2x4) split into A_L (cols 0-1) and A_R (cols 2-3), each 2x2.
    B (4x6) split by rows: top 2 rows, bottom 2 rows.
    For each of 3 column-pairs of B, we do:
      A_L * B_top_block  +  A_R * B_bot_block
    Each is a 2x2 * 2x2 multiplication.
    """
    rank = 42
    nm = n * m
    mp = m * p
    pn = p * n

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    # Strassen for 2x2 * 2x2:
    # X (first matrix), Y (second matrix), Z (result)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Products:
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Reconstruction:
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6

    # Sparse coefficient lists: (local_index, coefficient)
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
        [(0, -1), (1, 1)],      # P5: -Z00 + Z01  (FIXED: was (0,1) instead of (0,-1))
        [(3, 1)],               # P6: Z11
        [(0, 1)]                # P7: Z00
    ]

    col = 0
    # 3 column-blocks of B (each 2 columns), 2 row-blocks of A/B (top/bottom)
    for b in range(3):
        for a in range(2):
            for s in range(7):
                # Update U (A matrix, 2x4)
                # a=0: A_L (cols 0-1), a=1: A_R (cols 2-3)
                for (idx, coeff) in U_coeffs[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j, col] += coeff

                # Update V (B matrix, 4x6)
                # a=0: top rows (0-1), a=1: bottom rows (2-3)
                # b: column block, cols b*2, b*2+1
                for (idx, coeff) in V_coeffs[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k, col] += coeff

                # Update W (C matrix, 2x6, indexed as k*n + i)
                # k is in column block b, i is row
                for (idx, coeff) in W_coeffs[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i, col] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(n, m, p, U, V, W):
    """Verify exact decomposition by checking tensor equality on all entries."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]

    # Precompute U*V element-wise products for speed
    # For each (u,v) pair, compute the vector U[u,:]*V[v,:]
    # Then check dot with W[w,:] for all w

    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            j_v = v // p
            k = v % p
            if j_v != j:
                # Should be 0 for all w
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
