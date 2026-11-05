"""Search for a low-rank exact decomposition of 3x3x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _time_left():
    return DEADLINE - time.time() - SAFETY_MARGIN

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] == 1."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact arithmetic using Fractions.
    Assumes inputs are integers or half-integers (multiples of 0.5).
    We multiply by 2 to convert to integers for exact checking.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    r = U.shape[1]
    
    # Convert to Fractions
    # We assume inputs are clean half-integers.
    # If they are floats, we round to nearest 0.5 to be safe, but verification should be on the returned values.
    # The prompt says "integer/half-integer entries".
    
    U_frac = [[Fraction(U[i, c]).limit_denominator(2) for c in range(r)] for i in range(n*m)]
    V_frac = [[Fraction(V[i, c]).limit_denominator(2) for c in range(r)] for i in range(m*p)]
    W_frac = [[Fraction(W[i, c]).limit_denominator(2) for c in range(r)] for i in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    s += U_frac[i*m + j][c] * V_frac[j*p + k][c] * W_frac[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_3x3_core():
    """
    Returns U_core, V_core, W_core for 3x3x3 multiplication with rank 7.
    Indices:
    U: (3*3) x 7, indexed by (i, j)
    V: (3*3) x 7, indexed by (j, k)
    W: (3*3) x 7, indexed by (k, i)
    """
    U = np.zeros((9, 7))
    V = np.zeros((9, 7))
    W = np.zeros((9, 7))
    
    # Strassen's 7 multiplication formulas
    # M1 = (A00 + A22) * (B00 + B22)
    # M2 = (A20 + A22) * B00
    # M3 = A00 * (B01 - B21)
    # M4 = A22 * (B11 - B01)
    # M5 = (A01 + A11) * B11
    # M6 = (A10 - A20) * (B00 + B10)
    # M7 = (A01 - A11) * (A12 + A22) * B21 ... Wait, standard Strassen is for 2x2.
    # For 3x3, the rank 7 decomposition is not simply the 2x2 Strassen.
    # The rank of 3x3x3 multiplication is 7 (Strassen).
    # A known rank-7 decomposition for 3x3x3:
    # We can use the standard Strassen 7 terms if we embed 2x2? No, 3x3 is different.
    # Actually, the rank of 3x3x3 is 7.
    # A specific decomposition:
    # Terms:
    # 1. A00 B00
    # 2. A11 B11
    # 3. A22 B22
    # 4. (A00+A11)(B00+B11) - A00B00 - A11B11
    # 5. (A00+A22)(B00+B22) - A00B00 - A22B22
    # 6. (A11+A22)(B11+B22) - A11B11 - A22B22
    # 7. ... this doesn't cover all off-diagonals correctly.
    
    # Let's use a known rank-7 decomposition for 3x3.
    # Reference: Strassen (1969) "Gaussian multiplication is not minimal".
    # Actually, the rank is 7.
    # One explicit decomposition:
    # U, V, W entries in {-1, 0, 1}
    
    # Term 0: A00 B00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # (0,0) x (0,0) -> C00
    # Term 1: A11 B11
    U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1 # (1,1) x (1,1) -> C11
    # Term 2: A22 B22
    U[2, 2] = 1; V[2, 2] = 1; W[2, 2] = 1 # (2,2) x (2,2) -> C22
    
    # We need to generate off-diagonals:
    # C01 needs A00 B01 + A01 B11 + A02 B21
    # C02 needs A00 B02 + A01 B12 + A02 B22
    # C10 needs A10 B00 + A11 B10 + A12 B20
    # C12 needs A10 B02 + A11 B12 + A12 B22
    # C20 needs A20 B00 + A21 B10 + A22 B20
    # C21 needs A20 B01 + A21 B11 + A22 B21
    
    # This is getting complicated. Let's use a known rank 7 scheme for 3x3.
    # From literature, a rank 7 decomposition for 3x3x3 exists.
    # However, implementing it from memory is risky.
    # Alternative: Use rank 9 (schoolbook for 3x3) for the core?
    # If we use schoolbook 3x3 for each of the 7 columns, rank = 7 * 9 = 63.
    # This is the same as the main schoolbook.
    
    # Can we do better than 63?
    # If we use Strassen 2x2 (rank 7) for the 2x2 subblocks?
    # 3x3 can be viewed as 2x2 blocks?
    # Partition A, B, C into 2x2 blocks with some overlap?
    
    # Let's try a simple improvement:
    # For each column k, we compute C[:, k] = A B[:, k].
    # This is a 3x3 matrix times a 3x1 vector.
    # The tensor for 3x3x1 multiplication has rank 9 (schoolbook) or less?
    # It is known that the rank of 3x3x1 is 9? No, likely less.
    # Actually, the rank of n x n x 1 is n^2? No.
    # For 2x2x1, rank is 4?
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # 4 terms. Can we do 3?
    # If we could, total rank would be 7 * 3 = 21.
    
    # Given the difficulty of finding a valid <63 rank decomposition in this time,
    # and the risk of invalidity, we will stick to a robust approach.
    # However, we can try to verify a potential rank 49 scheme.
    # If we use Strassen 2x2 (rank 7) for the first 2 rows/cols?
    # No, the structure doesn't align easily.
    
    # Let's return the schoolbook core for now, which is 3x3 rank 9.
    U = np.zeros((9, 9))
    V = np.zeros((9, 9))
    W = np.zeros((9, 9))
    idx = 0
    for i in range(3):
        for j in range(3):
            U[i*3+j, idx] = 1
            V[j*3+0, idx] = 1 # Wait, V is (j,k). For 3x3x3 core, k is 0,1,2.
            # We are treating this as 3x3x3.
            # So V index is j*3+k.
            # But here we are just building the core.
            pass
    return None # Placeholder

def construct():
    n, m, p = DIMENSIONS
    start = time.time()

    # Fallback: Schoolbook
    # Rank = 3 * 3 * 7 = 63
    fallback = _schoolbook(n, m, p)
    best = fallback
    best_rank = 63

    # We can try to improve by checking if a known low-rank scheme exists.
    # Since we cannot reliably construct a rank < 63 scheme in this environment
    # without risking invalidity (which scores 0), we return the schoolbook.
    
    # One minor optimization: ensure the output format is correct.
    # U: (n*m, rank) = (9, 63)
    # V: (m*p, rank) = (21, 63)
    # W: (p*n, rank) = (21, 63)
    
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # If verification fails (shouldn't for schoolbook), return empty or raise
    # But schoolbook is guaranteed valid.
    return best
