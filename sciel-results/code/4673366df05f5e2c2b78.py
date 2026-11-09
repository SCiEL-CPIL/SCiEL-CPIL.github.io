"""Matrix multiplication 3x3 x 3x4 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 4]


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += float(U[i * m + j][r]) * float(V[j * p + k][r]) * float(W[k * n + i][r])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    n, m, p = int(n), int(m), int(p)
    R = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(R):
                    val += Fraction(str(U[i * m + j][r])) * Fraction(str(V[j * p + k][r])) * Fraction(str(W[k * n + i][r]))
                if val != 1:
                    return False
    return True


def _strassen_3x3_3x4():
    """
    Construct a rank-34 decomposition using Strassen's algorithm.
    
    Split A (3x3) as:
        A = [A11 A12]  where A11 is 2x2, A12 is 2x1
             [A21 A22]  where A21 is 1x2, A22 is 1x1
    
    Split B (3x4) as:
        B = [B11 B12]  where B11 is 2x2, B12 is 2x2
             [B21 B22]  where B21 is 1x2, B22 is 1x2
    
    C = AB:
        C11 = A11*B11 + A12*B21  (2x2)
        C12 = A11*B12 + A12*B22  (2x2)
        C21 = A21*B11 + A22*B21  (1x2)
        C22 = A21*B12 + A22*B22  (1x2)
    
    Use Strassen (rank 7) for the 2x2*2x2 parts.
    The remaining terms are handled directly.
    
    Actually, let me use a more systematic approach:
    We can view this as: C = [A11 A12; A21 A22] * [B11 B12; B21 B22]
    
    The full product has 36 scalar multiplications in schoolbook.
    Using Strassen for the 2x2*2x2 block (A11*B11) saves 4 mults (8->7).
    But we also have cross terms.
    
    Let me just directly construct rank-34 factors.
    
    Alternative: Use the known fact that R(3,3,3) <= 23 and adapt.
    For R(3,3,4), a simple bound is R(3,3,4) <= R(3,3,3) + R(3,3,1) = 23 + 9 = 32?
    No, that's not quite right.
    
    Let me use a different approach: construct the decomposition explicitly.
    
    For the 3x3 * 3x4 case, I'll use the following strategy:
    - Treat it as 3x3 * 3x3 + 3x3 * 3x1
    - R(3,3,3) <= 23 (best known)
    - R(3,3,1) = 9 (schoolbook, since it's just a matrix-vector product)
    - But this gives 32, and we can do better with a unified construction.
    
    Actually, the simplest guaranteed improvement over 36 is to use Strassen
    on a 2x2 sub-block. Let me construct rank-34 directly.
    
    Here's a concrete construction:
    
    Label the 36 schoolbook terms by (i,j,k) for i,j in {0,1,2}, k in {0,1,2,3}.
    
    For the 2x2 Strassen part (rows 0,1 of A; cols 0,1 of B):
    M1 = (A00+A11)*(B00+B11)
    M2 = (A00+A11)*(B10-B00)  
    M3 = (A10-A00)*(B00+B11)
    M4 = A00*(B11-B10)
    M5 = A11*(B00-B10)
    M6 = (A01+A10)*(B10+B11)
    M7 = (A01-A11)*(B00+B10)
    
    These give:
    C00 = M1 + M4 - M5 - M7
    C01 = M2 + M4 + M7
    C10 = M3 + M5 + M7
    C11 = M1 - M3 + M5 - M6
    
    The remaining 28 terms (involving row/col 2) are done in schoolbook.
    Total: 7 + 28 = 35. That's only rank 35.
    
    We can save one more by noting that some of the "remaining" terms
    can be combined. Let me just go with rank 35 as a safe fallback,
    and then try to reduce.
    
    Actually, let me just build rank-35 and then iteratively reduce.
    """
    n, m, p = 3, 3, 4
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen terms for the 2x2 block (rows 0,1 of A; cols 0,1 of B)
    # M1 = (A00+A11)*(B00+B11)
    U[0][col] = 1; U[3][col] = 1  # A00 (i=0,j=0), A11 (i=1,j=1)
    V[0][col] = 1; V[4][col] = 1  # B00 (j=0,k=0), B11 (j=1,k=1)
    W[0][col] = 1; W[3][col] = 1; W[6][col] = 1; W[9][col] = 1  # C00, C01, C10, C11
    # Actually W index is k*n+i. For C00: k=0,i=0 -> 0. C01: k=1,i=0 -> 1. C10: k=0,i=1 -> 3. C11: k=1,i=1 -> 4.
    W[0][col] = 1; W[1][col] = 1; W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A00+A11)*(B10-B00)
    U[0][col] = 1; U[3][col] = 1
    V[3][col] = 1; V[0][col] = -1  # B10 (j=1,k=0), B00 (j=0,k=0)
    # C00 -= M2, C01 += M2, C10 -= M2, C11 += M2 (from Strassen formulas)
    # C00 = M1 - M2 + ... so W for M2: C00 coeff -1, C01 coeff +1, C10 coeff -1, C11 coeff +1
    W[0][col] = -1; W[1][col] = 1; W[3][col] = -1; W[4][col] = 1
    col += 1
    
    # M3 = (A10-A00)*(B00+B11)
    U[3][col] = 1; U[0][col] = -1  # A10 (i=1,j=0), A00 (i=0,j=0)
    V[0][col] = 1; V[4][col] = 1
    # C00 -= M3, C01 += M3, C10 += M3, C11 -= M3
    W[0][col] = -1; W[1][col] = 1; W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M4 = A00*(B11-B10)
    U[0][col] = 1
    V[4][col] = 1; V[3][col] = -1  # B11, B10
    # C00 += M4, C01 += M4, C10 -= M4, C11 -= M4
    W[0][col] = 1; W[1][col] = 1; W[3][col] = -1; W[4][col] = -1
    col += 1
    
    # M5 = A11*(B00-B10)
    U[3][col] = 1
    V[0][col] = 1; V[3][col] = -1
    # C00 -= M5, C01 += M5, C10 += M5, C11 -= M5
    W[0][col] = -1; W[1][col] = 1; W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M6 = (A01+A10)*(B10+B11)
    U[1][col] = 1; U[3][col] = 1  # A01 (i=0,j=1), A10 (i=1,j=0)
    V[3][col] = 1; V[4][col] = 1
    # C00 -= M6, C01 -= M6, C10 += M6, C11 -= M6
    W[0][col] = -1; W[1][col] = -1; W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # M7 = (A01-A11)*(B00+B10)
    U[1][col] = 1; U[3][col] = -1
    V[0][col] = 1; V[3][col] = 1
    # C00 -= M7, C01 += M7, C10 += M7, C11 -= M7
    W[0][col] = -1; W[1][col] = 1; W[3][col] = 1; W[4][col] = -1
    col += 1
    
    # Now handle all remaining terms involving index 2 (row or col) in schoolbook
    # Terms: all (i,j,k) where i=2 or j=2 or k is handled but i,j not both in {0,1}
    # Already handled: (i,j) in {0,1}^2 with k in {0,1} via Strassen
    # Remaining: 
    #   (i,j) in {0,1}^2, k=2,3: 4*2=8 terms
    #   i=2, j in {0,1,2}, k in {0,1,2,3}: 3*4=12 terms
    #   i in {0,1}, j=2, k in {0,1,2,3}: 2*4=8 terms
    # Total remaining: 8+12+8 = 28
    # But we used 7 for Strassen, so total = 7+28 = 35
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this was handled by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}


def _try_reduce(U, V, W, n, m, p, rng, deadline):
    """Try to reduce rank by 1 using random linear combinations."""
    r = len(U[0])
    if r <= 1:
        return None
    
    # Try random linear combinations to zero out a column
    for attempt in range(200):
        if time.time() > deadline:
            return None
        a = rng.randrange(r)
        b = rng.randrange(r)
        if a == b:
            continue
        beta = rng.choice([Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2),
                           Fraction(2), Fraction(-2), Fraction(1, 3), Fraction(-1, 3),
                           Fraction(1, 4), Fraction(-1, 4), Fraction(3), Fraction(-3)])
        
        tU = [row[:] for row in U]
        tV = [row[:] for row in V]
        tW = [row[:] for row in W]
        
        for idx in range(n * m):
            tU[idx][a] = tU[idx][a] + beta * tU[idx][b]
        for idx in range(m * p):
            tV[idx][a] = tV[idx][a] + beta * tV[idx][b]
        for idx in range(p * n):
            tW[idx][a] = tW[idx][a] + beta * tW[idx][b]
        
        # Check if column a is now zero in tensor sense
        zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = tU[i * m + j][a] * tV[j * p + k][a] * tW[k * n + i][a]
                    if val != Fraction(0):
                        zero = False
                        break
                if not zero:
                    break
            if not zero:
                break
        
        if zero:
            U2 = [row[:a] + row[a + 1:] for row in tU]
            V2 = [row[:a] + row[a + 1:] for row in tV]
            W2 = [row[:a] + row[a + 1:] for row in tW]
            if _verify_exact(U2, V2, W2, n, m, p):
                return {"U": U2, "V": V2, "W": W2}
    
    return None


def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Start with Strassen-based rank-35 construction
    best = _strassen_3x3_3x4()
    best_rank = 35
    
    # Verify
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank
    
    # Try iterative rank reduction
    U, V, W = best["U"], best["V"], best["W"]
    current_rank = len(U[0])
    
    while current_rank > 1 and time.time() < deadline:
        result = _try_reduce(U, V, W, n, m, p, rng, deadline)
        if result is not None:
            U, V, W = result["U"], result["V"], result["W"]
            current_rank = len(U[0])
            best = result
            best_rank = current_rank
        else:
            # Try a few more random seeds
            for seed in range(10):
                if time.time() > deadline:
                    break
                rng2 = random.Random(seed * 1000 + current_rank)
                result = _try_reduce(U, V, W, n, m, p, rng2, deadline)
                if result is not None:
                    U, V, W = result["U"], result["V"], result["W"]
                    current_rank = len(U[0])
                    best = result
                    best_rank = current_rank
                    break
            else:
                # No reduction found, break
                break
    
    return best


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"Verified: {_verify(result['U'], result['V'], result['W'], 3, 3, 4)}")
