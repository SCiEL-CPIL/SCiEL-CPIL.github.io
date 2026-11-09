import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
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


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _construct_rank17(n, m, p):
    """
    Constructive rank 17 decomposition for 2x3x3 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 core (indices i,j,k in {0,1})
    and schoolbook terms for the remaining boundary indices (j=2 or k=2).
    """
    if n != 2 or m != 3 or p != 3:
        return None

    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen's 7 terms for i,j,k in {0,1}
    # r=0 (M1): (A00+A11)(B00+B11)
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1

    # r=1 (M2): (A10+A11)B00
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1

    # r=2 (M3): A00(B01-B11)
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1

    # r=3 (M4): A11(B10-B00)
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1

    # r=4 (M5): (A00+A01)B11
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1

    # r=5 (M6): (A10-A00)(B00+B01)
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1

    # r=6 (M7): (A01-A11)(B10+B11)
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1

    # Boundary terms (r=7..16)
    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]

    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}


def _gauss_solve(A, b, tol=Fraction(0)):
    """
    Solve A x = b using Gaussian elimination with partial pivoting.
    A is list of lists of Fractions, b is list of Fractions.
    Returns x (list of Fractions) or None if no solution.
    """
    n_rows = len(A)
    n_cols = len(A[0])
    if n_rows == 0:
        return [Fraction(0)] * n_cols

    # Augmented matrix
    aug = [A[i] + [b[i]] for i in range(n_rows)]

    rank = 0
    for col in range(n_cols):
        if rank >= n_rows:
            break
        # Find pivot
        pivot = None
        for row in range(rank, n_rows):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        # Swap
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        # Eliminate
        pivot_val = aug[rank][col]
        for row in range(n_rows):
            if row != rank and aug[row][col] != 0:
                factor = aug[row][col] / pivot_val
                for c in range(col, n_cols + 1):
                    aug[row][c] -= factor * aug[rank][c]
        rank += 1

    # Check for consistency
    for row in range(rank, n_rows):
        if aug[row][n_cols] != 0:
            return None

    # Extract solution
    x = [Fraction(0)] * n_cols
    for row in range(rank):
        # Find leading column
        lead_col = None
        for c in range(n_cols):
            if aug[row][c] != 0:
                lead_col = c
                break
        if lead_col is not None:
            x[lead_col] = aug[row][n_cols] / aug[row][lead_col]

    return x


def _try_eliminate_tensor(U, V, W, n, m, p, target_idx, max_time=30.0):
    """
    Try to eliminate tensor at target_idx by expressing it as a linear combination
    of the other tensors. Returns (new_U, new_V, new_W) or None.
    """
    start_time = time.time()
    current_rank = len(U[0])
    if current_rank <= 1:
        return None

    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]

    # Build system: for each (i,j,k),
    # sum_{r != target} c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = U[i,j,target] * V[j,k,target] * W[k,i,target]
    num_eqs = n * m * p
    num_unknowns = current_rank - 1

    A = []
    b = []
    unknown_indices = [r for r in range(current_rank) if r != target_idx]

    for (i, j, k) in product(range(n), range(m), range(p)):
        row = []
        for r in unknown_indices:
            row.append(Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r])
        A.append(row)
        b.append(Uf[i*m+j][target_idx] * Vf[j*p+k][target_idx] * Wf[k*n+i][target_idx])

    coeffs = _gauss_solve(A, b)
    if coeffs is None:
        return None

    # Verify solution
    for (i, j, k) in product(range(n), range(m), range(p)):
        lhs = Fraction(0)
        for idx, r in enumerate(unknown_indices):
            lhs += coeffs[idx] * Uf[i*m+j][r] * Vf[j*p+k][r] * Wf[k*n+i][r]
        rhs = Uf[i*m+j][target_idx] * Vf[j*p+k][target_idx] * Wf[k*n+i][target_idx]
        if lhs != rhs:
            return None

    # Eliminate: add -coeffs * target_tensor to other tensors
    for idx, r in enumerate(unknown_indices):
        if coeffs[idx] != 0:
            for i in range(n * m):
                Uf[i][r] -= coeffs[idx] * Uf[i][target_idx]
            for j in range(m * p):
                Vf[j][r] -= coeffs[idx] * Vf[j][target_idx]
            for k in range(p * n):
                Wf[k][r] -= coeffs[idx] * Wf[k][target_idx]

    # Remove target tensor
    for i in range(n * m):
        Uf[i].pop(target_idx)
    for j in range(m * p):
        Vf[j].pop(target_idx)
    for k in range(p * n):
        Wf[k].pop(target_idx)

    # Convert back
    U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Uf]
    V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Vf]
    W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in Wf]

    return {"U": U_out, "V": V_out, "W": W_out}


def _search_rank_reduction(U, V, W, n, m, p, target_rank, max_time=60.0):
    """
    Try to reduce rank to target_rank by iteratively eliminating tensors.
    """
    start_time = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = len(current_U[0])

    if current_rank <= target_rank:
        return None

    for attempt in range(current_rank - target_rank):
        if time.time() - start_time > max_time:
            break
        if deadline_exceeded():
            break
        if current_rank <= target_rank:
            break

        # Try each tensor as target for elimination
        eliminated = False
        for target_idx in range(current_rank):
            result = _try_eliminate_tensor(
                current_U, current_V, current_W, n, m, p, target_idx,
                max_time=max(5.0, max_time - (time.time() - start_time))
            )
            if result is not None:
                current_U = result["U"]
                current_V = result["V"]
                current_W = result["W"]
                current_rank -= 1
                eliminated = True
                break

        if not eliminated:
            break

    if current_rank <= target_rank:
        return {"U": current_U, "V": current_V, "W": current_W}

    return None


def construct():
    n, m, p = DIMENSIONS

    # Start with rank 17
    result = _construct_rank17(n, m, p)
    if result is None:
        result = _schoolbook(n, m, p)
        return result

    if not _verify_exact(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        return result

    # Try to reduce rank
    for target in range(16, 10, -1):
        if deadline_exceeded():
            break
        reduced = _search_rank_reduction(
            result["U"], result["V"], result["W"], n, m, p, target,
            max_time=max(10.0, TIME_LIMIT - elapsed() - 5.0)
        )
        if reduced is not None:
            if _verify_exact(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                result = reduced
                break

    return result


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
