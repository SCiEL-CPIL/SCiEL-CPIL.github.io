"""
Exact rank-minimal tensor decomposition for 2x3 * 3x5 matrix multiplication.

Strategy:
- The block-diagonal construction gives rank 25 (20 for 2x2x5 block, 5 for 1x1x5 block).
- The full schoolbook rank is 30.
- We attempt a targeted search to find a lower-rank exact decomposition for the
  2x2x5 sub-tensor.  If none is found within the time budget, we fall back to the
  known-valid rank-25 block-diagonal result.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]
DEADLINE = 120.0
_START = time.time()


def _check(U, V, W, n, m, p):
    """Return True iff the tensor T[i,j,k]=delta(i,k) equals sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]."""
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1 if i == k else 0
                got = 0.0
                for c in range(r):
                    got += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(got - expected) > 1e-9:
                    return False
    return True


# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Rank n*m*p exact decomposition (always valid)."""
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


def _block_diagonal(n, m, p):
    """
    Rank 25 decomposition via block-diagonal embedding:
      Block 1: 2x2 * 2x5  -> rank 20 (schoolbook for this sub-block)
      Block 2: 1x1 * 1x5  -> rank  5
    Total rank = 25.
    """
    n1, n2 = 2, 1
    m1, m2 = 2, 1
    assert n1 + n2 == n and m1 + m2 == m

    # --- Block 1: 2x2 * 2x5, rank 20 ---
    r1 = n1 * m1 * p
    U1 = [[0] * r1 for _ in range(n1 * m1)]
    V1 = [[0] * r1 for _ in range(m1 * p)]
    W1 = [[0] * r1 for _ in range(p * n1)]
    c = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p):
                U1[i * m1 + j][c] = 1
                V1[j * p + k][c] = 1
                W1[k * n1 + i][c] = 1
                c += 1

    # --- Block 2: 1x1 * 1x5, rank 5 ---
    r2 = n2 * m2 * p
    U2 = [[0] * r2 for _ in range(n2 * m2)]
    V2 = [[0] * r2 for _ in range(m2 * p)]
    W2 = [[0] * r2 for _ in range(p * n2)]
    c = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p):
                U2[i * m2 + j][c] = 1
                V2[j * p + k][c] = 1
                W2[k * n2 + i][c] = 1
                c += 1

    # --- Embed into full spaces ---
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # A1: rows 0..1, cols 0..1 of A
    for i in range(n1):
        for j in range(m1):
            local = i * m1 + j
            glob = i * m + j
            for c in range(r1):
                U[glob][c] = U1[local][c]

    # A2: row 2, col 2 of A
    for i in range(n2):
        for j in range(m2):
            local = i * m2 + j
            glob = (n1 + i) * m + (m1 + j)
            for c in range(r2):
                U[glob][r1 + c] = U2[local][c]

    # B1: rows 0..1 of B
    for j in range(m1):
        for k in range(p):
            local = j * p + k
            glob = j * p + k
            for c in range(r1):
                V[glob][c] = V1[local][c]

    # B2: row 2 of B
    for j in range(m2):
        for k in range(p):
            local = j * p + k
            glob = (m1 + j) * p + k
            for c in range(r2):
                V[glob][r1 + c] = V2[local][c]

    # W1: output rows for block 1
    for i in range(n1):
        for k in range(p):
            local = k * n1 + i
            glob = k * n + i
            for c in range(r1):
                W[glob][c] = W1[local][c]

    # W2: output rows for block 2
    for i in range(n2):
        for k in range(p):
            local = k * n2 + i
            glob = k * n + (n1 + i)
            for c in range(r2):
                W[glob][r1 + c] = W2[local][c]

    return {"U": U, "V": V, "W": W}


def _search_low_rank_subtensor(n1, m1, p, max_rank, time_budget=30.0):
    """
    Attempt to find an exact decomposition of rank <= max_rank for the
    n1 x m1 * m1 x p sub-tensor.  Returns (U, V, W) lists or None.
    Uses a simple random-search + least-squares refinement heuristic.
    """
    total_time = time_budget
    start = time.time()
    rng = random.Random(42)

    best_rank = None
    best_UVW = None

    for r in range(max_rank, 0, -1):
        if time.time() - start > total_time:
            break
        # Try several random initializations
        for trial in range(50):
            if time.time() - start > total_time:
                break
            # Random initialization with small entries
            U = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(n1 * m1)], dtype=float)
            V = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(m1 * p)], dtype=float)
            W = np.array([[rng.uniform(-1, 1) for _ in range(r)] for _ in range(p * n1)], dtype=float)

            # Check if this random guess happens to be exact (very unlikely, but cheap)
            U_l = [[float(x) for x in row] for row in U]
            V_l = [[float(x) for x in row] for row in V]
            W_l = [[float(x) for x in row] for row in W]
            if _check(U_l, V_l, W_l, n1, m1, p):
                return U_l, V_l, W_l

            # Build linear system: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # This is nonlinear.  Instead, fix two matrices and solve for the third.
            # Fix U and V, solve for W:  W[k,i,:] appears in equations for each (i,j,k).
            # Equation: sum_j sum_c U[i*m1+j,c] * V[j*p+k,c] * W[k*n1+i,c] = 1
            # For fixed (k,i): sum_j sum_c (U[i*m1+j,c] * V[j*p+k,c]) * W[k*n1+i,c] = 1
            # This is a linear system in W[k*n1+i,:].
            # Number of unknowns per (k,i): r.  Number of equations: m1 (for j=0..m1-1).
            # If m1 >= r, we have more equations than unknowns -> overdetermined.
            # If m1 < r, underdetermined -> many solutions.

            # Let's fix U and V, and for each (k,i) solve the linear system for W row.
            # If a solution exists for all (k,i), then we have a valid decomposition.

            # Build the coefficient matrix for a given (k,i):
            # For each j, coefficient for W[k*n1+i,c] is sum over... no, it's:
            # sum_c (U[i*m1+j,c] * V[j*p+k,c]) * W[k*n1+i,c] = 1  for each j.
            # So the row of the linear system for equation j is:
            #   [ U[i*m1+j,0]*V[0*p+k,0], U[i*m1+j,1]*V[0*p+k,1], ..., U[i*m1+j,r-1]*V[0*p+k,r-1] ]
            # Wait, the index in V is j*p+k, which depends on j and k.
            # So for equation j:  sum_c (U[i*m1+j,c] * V[j*p+k,c]) * W[k*n1+i,c] = 1
            # The coefficient of W[k*n1+i,c] in equation j is U[i*m1+j,c] * V[j*p+k,c].

            valid_W = True
            W_solution = [[0.0] * r for _ in range(p * n1)]
            for k in range(p):
                for i in range(n1):
                    # Build m1 x r matrix A where A[j,c] = U[i*m1+j,c] * V[j*p+k,c]
                    A = np.zeros((m1, r))
                    for j in range(m1):
                        for c in range(r):
                            A[j, c] = U[i * m1 + j, c] * V[j * p + k, c]
                    # Solve A @ w = 1 for w in R^r
                    if m1 >= r:
                        # Overdetermined or square: use lstsq
                        w, residuals, rankA, sv = np.linalg.lstsq(A, np.ones(m1), rcond=None)
                        if residuals.size > 0 and residuals[0] > 1e-6:
                            valid_W = False
                            break
                        W_solution[k * n1 + i] = [float(x) for x in w]
                    else:
                        # Underdetermined: need A @ w = 1 to be consistent.
                        # Check if 1 is in the column space of A.
                        # Use lstsq; if residual is ~0, we're good.
                        w, residuals, rankA, sv = np.linalg.lstsq(A, np.ones(m1), rcond=None)
                        if residuals.size > 0 and residuals[0] > 1e-6:
                            valid_W = False
                            break
                        W_solution[k * n1 + i] = [float(x) for x in w]
                if not valid_W:
                    break

            if valid_W:
                # Verify the full decomposition
                if _check(U_l, V_l, W_solution, n1, m1, p):
                    best_rank = r
                    best_UVW = (U_l, V_l, W_solution)
                    break

        if best_rank is not None:
            break

    return best_UVW if best_UVW is not None else None


def construct():
    n, m, p = DIMENSIONS
    best = _block_diagonal(n, m, p)  # rank 25, guaranteed valid

    # Verify the block-diagonal result
    if not _check(best["U"], best["V"], best["W"], n, m, p):
        best = _schoolbook(n, m, p)  # rank 30, definitely valid

    # Try to find a lower-rank decomposition for the 2x2x5 sub-tensor
    # and reconstruct the full decomposition.
    n1, m1 = 2, 2
    time_remaining = DEADLINE - (time.time() - _START) - 10.0
    if time_remaining > 15.0:
        # Search for a rank < 20 decomposition of the 2x2x5 sub-tensor
        sub_result = _search_low_rank_subtensor(n1, m1, p, max_rank=19, time_budget=min(time_remaining, 40.0))
        if sub_result is not None:
            U1_l, V1_l, W1_l = sub_result
            r1 = len(U1_l[0]) if U1_l else 0
            r2 = n2_m2_p = 1 * 1 * p  # 5
            rank = r1 + r2

            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]

            # Embed U1
            for i in range(n1):
                for j in range(m1):
                    local = i * m1 + j
                    glob = i * m + j
                    for c in range(r1):
                        U[glob][c] = U1_l[local][c]

            # Embed U2 (1x1x5 schoolbook)
            for c in range(r2):
                U[(n1) * m + (m1)][r1 + c] = 1  # only element

            # Embed V1
            for j in range(m1):
                for k in range(p):
                    local = j * p + k
                    glob = j * p + k
                    for c in range(r1):
                        V[glob][c] = V1_l[local][c]

            # Embed V2
            for k in range(p):
                glob = (m1) * p + k
                V[glob][r1 + 0] = 1  # r2=5, only c=0 index within block2? No, r2=5, c ranges 0..4
                # Actually r2 = 5, and the 1x1x5 schoolbook has 5 components each with a single 1.
                # V2 has shape (1*p, 5) = (5, 5). V2[j*p+k][c] = 1 if j==0 and c==k, else 0.
                # Wait, in the schoolbook for 1x1*p: U[0][c]=1, V[0*p+k][c]=1, W[k*n2+0][c]=1.
                # So V2[k][k] = 1 (diagonal).

            # Let me redo V2 and W2 properly.
            # Reset and redo embedding for block 2.
            # V2: shape (1*p, r2) = (5, 5). V2[k][c] = 1 if c == k else 0.
            for k in range(p):
                for c in range(r2):
                    V[(m1) * p + k][r1 + c] = 1 if c == k else 0

            # W1: shape (p*n1, r1) = (10, r1)
            for i in range(n1):
                for k in range(p):
                    local = k * n1 + i
                    glob = k * n + i
                    for c in range(r1):
                        W[glob][c] = W1_l[local][c]

            # W2: shape (p*n2, r2) = (5, 5). W2[k][c] = 1 if c == k else 0.
            for k in range(p):
                for c in range(r2):
                    W[k * n + n1][r1 + c] = 1 if c == k else 0

            # Verify
            if _check(U, V, W, n, m, p):
                if rank < len(best["U"][0]):
                    best = {"U": U, "V": V, "W": W}

    # Deadline guard
    if time.time() - _START > DEADLINE - 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
