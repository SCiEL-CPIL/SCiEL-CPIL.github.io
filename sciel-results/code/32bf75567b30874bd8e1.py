import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0


def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product of Strassen's 2x2x2 decomposition to create a
    rank-49 decomposition for 4x4x4, then duplicates it for the two 4x4 blocks
    of the 4x8 output.
    """
    U2, V2, W2 = _strassen_2x2x2()

    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i * 4 + j, c] = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]

            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j * 4 + k, c] = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]

            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = k // 2, k % 2
                    W4[k * 4 + i, c] = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]

    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49

        U[:, col_start:col_end] = U4

        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]

        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]

    # Verification
    n, m, p = 4, 4, 8
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return None

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _build_rank96():
    """
    Attempts to construct a rank-96 decomposition for 4x4 * 4x8.
    
    Strategy: Use Strassen-Kronecker for the first 4x4x4 block (rank 49),
    then try to share some tensor factors with the second 4x4x4 block.
    
    The key insight: if we can find a shared sub-decomposition that covers
    parts of both output blocks, we can reduce the total rank.
    
    For this attempt, we use:
    - Block 0 (output cols 0-3): full Strassen-Kronecker (rank 49)
    - Block 1 (output cols 4-7): try to use a reduced set of factors
    
    Specifically, we note that the matrix multiplication tensor for 4x4x8
    can be seen as two copies of 4x4x4. If we can find a decomposition
    where some factors are shared between the two blocks, we save rank.
    
    One approach: use the same U factors for both blocks (which is already
    done in rank-98), but try to compress the V and W factors.
    
    For a concrete rank-96 attempt, we can try to eliminate 2 rank-1 terms
    by finding linear dependencies. However, this requires solving a system
    of equations, which may not yield integer/half-integer solutions.
    
    As a simpler approach: we can try a hybrid where one block uses Strassen
    and the other uses a different decomposition that shares some structure.
    """
    # This is a placeholder for a more sophisticated rank-reduction attempt.
    # For now, we return None to indicate this specific rank-96 construction
    # was not successfully implemented in this iteration.
    return None


def _schoolbook(n=4, m=4, p=8):
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


def construct():
    # Try rank-96 first (if implemented)
    try:
        if time.time() < DEADLINE:
            res = _build_rank96()
            if res is not None:
                return res
    except Exception:
        pass

    # Try the rank-98 block-Strassen decomposition
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass

    # Fallback to schoolbook
    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
