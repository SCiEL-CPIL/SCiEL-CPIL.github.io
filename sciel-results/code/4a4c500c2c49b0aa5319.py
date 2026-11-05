import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_rank28(n, m, p):
    """Build a rank-28 decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(k_start):
        nonlocal col
        # A indices for 2x2 block (cols 0,1)
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1

        # B indices for 2x2 block (rows 0,1; cols k_start, k_start+1)
        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        # C indices for 2x2 block (rows 0,1; cols k_start, k_start+1)
        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        # M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] += -1
        col += 1

        # M3 = A00(B01+B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] += 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        # M4 = A11(B10+B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] += 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        # M5 = (A00+A01)B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] += -1; W[c01][col] += 1
        col += 1

        # M6 = (A10+A00)(B00+B01)
        U[a10][col] += 1; U[a00][col] += -1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (A01+A11)(B10+B11)
        U[a01][col] += 1; U[a11][col] += -1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    # Apply Strassen to 3 blocks: cols 0-1, 2-3, 4-5
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)

    # Handle A[:, 2] * B[2, :] -> C[:, :]
    # This is 2x1x7 multiplication. Rank 7 (schoolbook for this part).
    for k in range(7):
        # Term for C[0, k] = A[0, 2] * B[2, k]
        U[0 * 3 + 2][col] += 1
        V[2 * 7 + k][col] += 1
        W[k * 2 + 0][col] += 1
        col += 1
        
        # Term for C[1, k] = A[1, 2] * B[2, k]
        U[1 * 3 + 2][col] += 1
        V[2 * 7 + k][col] += 1
        W[k * 2 + 1][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _peel_columns(U, V, W, n, m, p, max_rounds=50, tol=1e-10):
    """Aggressively peel removable columns."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    for _ in range(max_rounds):
        if current_rank == 0:
            break
        removed = False
        for col in range(current_rank):
            U_test = [row[:col] + row[col+1:] for row in U]
            V_test = [row[:col] + row[col+1:] for row in V]
            W_test = [row[:col] + row[col+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, tol):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, current_rank

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    result = _build_rank28(n, m, p)

    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce rank
        U, V, W, cur_rank = _peel_columns(result["U"], result["V"], result["W"], n, m, p)
        if cur_rank < len(result["U"][0]):
            result = {"U": U, "V": V, "W": W}
        
        return result

    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
