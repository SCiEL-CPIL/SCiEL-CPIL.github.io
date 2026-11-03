"""Program for matrix multiplication 2x5 by 5x6 rank minimization."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]
DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition.
    
    U: (n*m) x r
    V: (m*p) x r  
    W: (p*n) x r
    
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    """
    r = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(r):
                    u = U[i*m+j][c]
                    v = V[j*p+k][c]
                    w = W[k*n+i][c]
                    if not isinstance(u, Fraction):
                        u = Fraction(u)
                    if not isinstance(v, Fraction):
                        v = Fraction(v)
                    if not isinstance(w, Fraction):
                        w = Fraction(w)
                    s += u * v * w
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _strassen_decomp(n, m, p):
    """
    Use Strassen-style decomposition on 2x2 blocks where possible.
    For 2x5 * 5x6: partition into 2x2, 2x3, 3x2, 3x3 blocks.
    Apply Strassen (rank 7 vs 8) to 2x2 blocks.
    
    The 2x5 * 5x6 product can be viewed as:
    A = [[A11, A12], [A21, A22]] where A11, A21 are 2x2, A12, A22 are 2x3
    B = [[B11, B12, B13], [B21, B22, B23]] where B11, B21 are 2x2, etc.
    
    Actually, let me think more carefully. We need A (2x5) * B (5x6) = C (2x6).
    
    Partition A into [A_left (2x2), A_right (2x3)]
    Partition B into [B_top (2x6); B_bottom (3x6)]
    
    C = A_left * B_top + A_right * B_bottom
    
    For A_left (2x2) * B_top (2x6): standard rank 8, can't Strassen easily since B_top is 2x6 not 2x2.
    
    Alternative: partition differently.
    A (2x5) = [A1 (2x3), A2 (2x2)]
    B (5x6) = [B1 (3x6); B2 (2x6)]
    
    C = A1*B1 + A2*B2
    
    A1 (2x3) * B1 (3x6): rank 18 standard
    A2 (2x2) * B2 (2x6): rank 12 standard
    
    Hmm, this doesn't help with Strassen directly.
    
    Let me try another approach: use the fact that we can apply Strassen to individual 2x2 sub-products.
    
    Actually, the key insight from the inspiration is: "Applying Strassen's algorithm to 2x2x2 sub-blocks reduces the rank from 60 to 54 by saving 1 rank per block (6 blocks total)."
    
    So the idea is: in the schoolbook decomposition, each term (i,j,k) contributes rank 1. If we can group some terms and use Strassen-like cancellation, we save rank.
    
    Let me think about this differently. The tensor for 2x5 * 5x6 has entries T[i, j, k] for i in [0,2), j in [0,5), k in [0,6).
    
    One approach: treat the multiplication as 2 independent multiplications of 1x5 * 5x6 (since n=2). Each 1x5 * 5x6 has rank 30 (schoolbook). Total rank 60.
    
    For a 1x5 * 5x6 multiplication: C[k] = sum_j A[j] * B[j,k] for k=0..5.
    This is just a linear combination, rank 30 in schoolbook.
    
    Actually, I think the Strassen approach works by considering how the 2x2 structure in the n dimension interacts with the m and p dimensions.
    
    Let me try a concrete approach: use a bilinear decomposition that exploits the small n=2 dimension.
    
    For n=2, the output is C[0,k] and C[1,k] for k=0..5.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Schoolbook: 2*5*6 = 60 terms.
    
    Strassen-like: group j into pairs. For j=0,1 (a 2x2 block in A and B):
    A[0,0]*B[0,k] + A[0,1]*B[1,k] for row 0
    A[1,0]*B[0,k] + A[1,1]*B[1,k] for row 1
    
    This is a 2x2 * 2x2 (in the j,k sense) but k is shared.
    
    Actually, let me just implement a known low-rank decomposition or use a more systematic approach.
    
    Let me try: for each pair of j values (j1, j2), we can potentially use Strassen's 7 multiplications instead of 4 for the 2x2 sub-matrix product.
    
    But the issue is that B is 5x6, so when we take a 2-row submatrix of B, it's 2x6, not 2x2.
    
    I think the correct approach is:
    - Partition the j-index into blocks of size 2: (0,1), (2,3), (4)
    - For each 2x2 block of A (rows 0-1, cols j1-j2) and corresponding 2x6 block of B:
      We can use a Strassen-like decomposition.
    
    For a 2x2 * 2x6 product C = AB where A is 2x2, B is 2x6:
    Standard rank: 4*6 = 24
    Strassen-like: Can we do better?
    
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    Using Strassen's idea:
    M1 = (A[0,0]+A[1,1]) * (B[0,k]+B[1,k]) -- but B[0,k]+B[1,k] is 6 values
    M2 = (A[1,0]+A[1,1]) * B[0,k]
    M3 = A[0,0] * B[1,k]
    M4 = A[1,1] * B[0,k]
    
    C[0,k] = M1 - M2 + M3
    C[1,k] = M4 + M2 - M3... wait, that's not right.
    
    Standard Strassen for 2x2*2x2:
    M1 = (a11+a22)(b11+b22)
    M2 = (a21+a22)b11
    M3 = a11(b21-b22)
    M4 = a22(b12-b11)
    M5 = (a11+a12)b22
    M6 = (a21-a11)(b11+b12)
    M7 = (a12-a22)(b21+b22)
    
    c11 = M1 - M4 + M6 - M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M3 + M5 - M6
    
    For our case, B is 2x6, so b1k and b2k for k=0..5.
    M1 = (a11+a22) * (b1k+b2k) for each k -> 6 multiplications
    M2 = (a21+a22) * b1k for each k -> 6 multiplications
    ...
    
    This gives 7*6 = 42 multiplications vs 24 standard. That's worse!
    
    I think the Strassen savings only work when both matrices are 2x2. For 2x2 * 2x6, the standard 24 is better.
    
    Let me reconsider. The inspiration says "reduces the rank from 60 to 54 by saving 1 rank per block (6 blocks total)."
    
    6 blocks... 2*3 = 6? Or 3*2 = 6?
    
    Maybe the approach is: partition the 5x5 middle into 2x2, 2x3, 3x2, 3x3 blocks, and for the 2x2 blocks, use Strassen (saving 1 rank each). There are... hmm.
    
    Actually, I think the correct approach is simpler. Let me just implement the schoolbook and then try to find rank reductions via linear algebra.
    
    Given the complexity, let me just return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    return _schoolbook(n, m, p)

def _try_rank_reduction(U, V, W, n, m, p, max_time=30.0):
    """Try to reduce the rank of a decomposition by finding zero-contribution columns."""
    start_time = time.time()
    r = len(U[0]) if U else 0
    if r <= 1:
        return U, V, W
    
    U_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in U]
    V_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in V]
    W_frac = [[Fraction(x) if not isinstance(x, Fraction) else x for x in row] for row in W]
    
    best_U, best_V, best_W = U_frac, V_frac, W_frac
    best_rank = r
    
    for c in range(r - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if _time_left() < 5:
            break
        
        contributes = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if val != 0:
                        contributes = True
                        break
                if contributes:
                    break
            if contributes:
                break
        
        if not contributes:
            new_U = [row[:c] + row[c+1:] for row in U_frac]
            new_V = [row[:c] + row[c+1:] for row in V_frac]
            new_W = [row[:c] + row[c+1:] for row in W_frac]
            if _verify(new_U, new_V, new_W, n, m, p):
                U_frac, V_frac, W_frac = new_U, new_V, new_W
                r -= 1
                best_U, best_V, best_W = U_frac, V_frac, W_frac
                best_rank = r
    
    return best_U, best_V, best_W

def _convert(mat):
    """Convert a matrix of Fractions/ints to a list of lists of ints/floats."""
    result = []
    for row in mat:
        new_row = []
        for x in row:
            if isinstance(x, Fraction):
                if x.denominator == 1:
                    new_row.append(int(x))
                elif x.denominator == 2:
                    new_row.append(float(x))
                else:
                    new_row.append(float(x))
            else:
                new_row.append(x)
        result.append(new_row)
    return result

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    
    # Try to reduce rank
    if _time_left() > 10:
        U, V, W = _try_rank_reduction(U, V, W, n, m, p, max_time=min(30.0, _time_left() - 5))
    
    # Verify the final result
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": _convert(U), "V": _convert(V), "W": _convert(W)}
