"""Matrix multiplication 3x5x6 tensor rank minimization via peeling + exact search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90

    def verify_exact(U_mat, V_mat, W_mat, n, m, p):
        """U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank) as lists of lists of Fraction."""
        if not U_mat:
            return n * m * p == 0
        rank = len(U_mat[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ur = i * m + j
                    vr = j * p + k
                    wr = k * n + i
                    for r in range(rank):
                        total += U_mat[ur][r] * V_mat[vr][r] * W_mat[wr][r]
                    if total != 1:
                        return False
        return True

    def from_frac_matrix(fmat):
        result = []
        for row in fmat:
            out_row = []
            for x in row:
                if isinstance(x, Fraction):
                    if x.denominator == 1:
                        out_row.append(x.numerator)
                    else:
                        out_row.append(x)
                else:
                    out_row.append(x)
            result.append(out_row)
        return result

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W

    # Peeling construction: peel first row (i=0) and first column (k=0) of C.
    # This exploits rectangular asymmetry: the (0,0) entry is shared by both.
    # Sub-problem: (n-1) x (m-1) x (p-1) with indices i'=1..n-1, j'=1..m-1, k'=1..p-1.
    def peeling_decomposition(n, m, p):
        """
        Build a decomposition by peeling the first row and first column of C.
        Rank = (m-1) + (p-1) + (n-1)*(m-1)*(p-1)
        """
        U = []
        V = []
        W = []
        col = 0

        # Part 1: peel first row of C (i=0)
        # For each j in 0..m-1, k in 1..p-1:
        # U[i*m+j, r] = 1 if i==0, else 0
        # V[j*p+k, r] = 1
        # W[k*n+i, r] = 1 if i==0, else 0
        for j in range(m):
            for k in range(1, p):
                for i in range(n):
                    U.append([Fraction(1) if i == 0 else Fraction(0)] * 1)
                for jj in range(m):
                    for kk in range(p):
                        V.append([Fraction(1) if (jj == j and kk == k) else Fraction(0)])
                for kk in range(p):
                    for ii in range(n):
                        W.append([Fraction(1) if (kk == k and ii == 0) else Fraction(0)])
                col += 1

        # Part 2: peel first column of C (k=0), but NOT the (0,0) entry (already covered)
        # For each i in 1..n-1, j in 0..m-1:
        # U[i*m+j, r] = 1
        # V[j*p+0, r] = 1
        # W[0*n+i, r] = 1 if i>0, else 0
        for i in range(1, n):
            for j in range(m):
                for ii in range(n):
                    for jj in range(m):
                        U.append([Fraction(1) if (ii == i and jj == j) else Fraction(0)])
                for jj in range(m):
                    for kk in range(p):
                        V.append([Fraction(1) if (jj == j and kk == 0) else Fraction(0)])
                for kk in range(p):
                    for ii in range(n):
                        W.append([Fraction(1) if (kk == 0 and ii == i) else Fraction(0)])
                col += 1

        # Part 3: sub-problem for i>=1, j>=1, k>=1 (i.e., i'=i-1, j'=j-1, k'=k-1)
        for i in range(1, n):
            for j in range(1, m):
                for k in range(1, p):
                    for ii in range(n):
                        for jj in range(m):
                            U.append([Fraction(1) if (ii == i and jj == j) else Fraction(0)])
                    for jj in range(m):
                        for kk in range(p):
                            V.append([Fraction(1) if (jj == j and kk == k) else Fraction(0)])
                    for kk in range(p):
                        for ii in range(n):
                            W.append([Fraction(1) if (kk == k and ii == i) else Fraction(0)])
                    col += 1

        return U, V, W

    # Build the peeling decomposition (rank 80 for 3x5x6)
    U_frac, V_frac, W_frac = peeling_decomposition(n, m, p)
    best_rank = len(U_frac[0]) if U_frac else 0

    # Verify the peeling decomposition
    if not verify_exact(U_frac, V_frac, W_frac, n, m, p):
        # Fallback to schoolbook
        U_frac, V_frac, W_frac = schoolbook_decomposition(n, m, p)
        best_rank = schoolbook_rank

    # Try to reduce rank further via exact search
    # For small ranks, we can try random integer/half-integer initializations
    def try_random_rank(rank, max_restarts=20, seed=42):
        """Try to find a decomposition of given rank using random restarts with small rationals."""
        if rank < 1:
            return None

        rng = np.random.RandomState(seed)

        for restart in range(max_restarts):
            # Initialize with small random values from {-1, 0, 1} and {-1/2, 1/2}
            U_r = rng.randint(-1, 2, (n * m, rank)).astype(np.float64)
            V_r = rng.randint(-1, 2, (m * p, rank)).astype(np.float64)
            W_r = rng.randint(-1, 2, (p * n, rank)).astype(np.float64)

            # Quick numerical check
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U_r[i * m + j, r] * V_r[j * p + k, r] * W_r[k * n + i, r]
                        T_check[i, j, k] = total

            if np.allclose(T_check, 1.0, atol=1e-10):
                # Convert to fractions
                U_f = [[Fraction(x).limit_denominator(4) for x in row] for row in U_r.tolist()]
                V_f = [[Fraction(x).limit_denominator(4) for x in row] for row in V_r.tolist()]
                W_f = [[Fraction(x).limit_denominator(4) for x in row] for row in W_r.tolist()]
                if verify_exact(U_f, V_f, W_f, n, m, p):
                    return U_f, V_f, W_f

        return None

    # Try ranks below the peeling rank
    for rank in range(best_rank - 1, max(best_rank - 10, 0), -1):
        if _time_remaining() < 10:
            break

        result = try_random_rank(rank, max_restarts=5, seed=rank)
        if result is not None:
            U_frac, V_frac, W_frac = result
            best_rank = rank
            break

    # Return the best result found
    result_U = from_frac_matrix(U_frac)
    result_V = from_frac_matrix(V_frac)
    result_W = from_frac_matrix(W_frac)

    return {"U": result_U, "V": result_V, "W": result_W}
