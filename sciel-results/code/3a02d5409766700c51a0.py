"""Matrix multiplication 3x5x6 tensor rank minimization via structured search."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFE_MARGIN = 5.0

def _time_remaining():
    return _DEADLINE - _SAFE_MARGIN - (time.time() - _START_TIME)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x6 matrix. Output factor W follows W[k*n+i, r].
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 90

    # Verification function using exact arithmetic
    def verify_exact(U_mat, V_mat, W_mat, n, m, p):
        """U_mat: (n*m, rank), V_mat: (m*p, rank), W_mat: (p*n, rank) as lists of lists of Fraction"""
        rank = len(U_mat[0]) if U_mat else 0
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
        """Convert list of lists of Fraction to list of lists of int or Fraction"""
        result = []
        for row in fmat:
            out_row = []
            for x in row:
                if x.denominator == 1:
                    out_row.append(x.numerator)
                else:
                    out_row.append(x)
            result.append(out_row)
        return result

    # Initialize with schoolbook
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

    U_frac, V_frac, W_frac = schoolbook_decomposition(n, m, p)
    best_rank = schoolbook_rank

    # Try to find a better decomposition using random search
    def try_random_rank(rank, max_restarts=10, seed=42):
        """Try to find a decomposition of given rank using random restarts."""
        if rank < 1:
            return None

        rng = np.random.RandomState(seed)

        for restart in range(max_restarts):
            # Initialize with small random integers
            U_r = rng.randint(-1, 2, (n*m, rank))
            V_r = rng.randint(-1, 2, (m*p, rank))
            W_r = rng.randint(-1, 2, (p*n, rank))

            # Quick numerical check
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        T_check[i,j,k] = total

            if np.allclose(T_check, 1.0, atol=1e-10):
                # Convert to fractions
                U_f = [[Fraction(x) for x in row] for row in U_r.tolist()]
                V_f = [[Fraction(x) for x in row] for row in V_r.tolist()]
                W_f = [[Fraction(x) for x in row] for row in W_r.tolist()]
                if verify_exact(U_f, V_f, W_f, n, m, p):
                    return U_f, V_f, W_f

        return None

    # Try some lower ranks
    for rank in range(80, 89):
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
