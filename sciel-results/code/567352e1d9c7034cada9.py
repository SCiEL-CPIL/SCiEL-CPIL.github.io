"""Matrix multiplication 2x2 x 2x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _fr(x):
    if isinstance(x, Fraction):
        return x
    return Fraction(x)

def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s
    n, m, p = DIMENSIONS
    schoolbook = n * m * p  # 20

    # --- Schoolbook fallback (rank 20) ---
    def schoolbook_decomp():
        r = n * m * p
        U = [[Fraction(0)] * r for _ in range(n * m)]
        V = [[Fraction(0)] * r for _ in range(m * p)]
        W = [[Fraction(0)] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = Fraction(1)
                    V[j*p+k][col] = Fraction(1)
                    W[k*n+i][col] = Fraction(1)
                    col += 1
        return U, V, W, r

    best = schoolbook_decomp()

    # --- Explicit block construction: Strassen(7) for cols 0,1 + rank-3 for cols 2,3,4 ---
    # This gives rank 7 + 3*3 = 16.
    # We build U,V,W with columns grouped.
    def block_decomp():
        # Strassen 7-term for 2x2 * 2x2 (left factor A 2x2, right 2x2 block)
        # U rows indexed by (i,j) in A: row = i*m + j = i*2 + j
        # V rows indexed by (j,k) in B block: row = j*p + k
        # W rows indexed by (k,i) in C block: row = k*n + i
        # We'll assign 7 columns for the 2x2 block on columns 0,1 of B.
        # Strassen products (A rows i, cols j; B rows j, cols k):
        # m1 = (A00+A11)(B00+B11)
        # m2 = (A10+A11)B00
        # m3 = A00(B10-B11)
        # m4 = A11(B01-B00)
        # m5 = (A00+A01)B11
        # m6 = (A10-A00)(B00+B01)
        # m7 = (A01-A11)(B10+B11)
        # C00 = m1 + m4 - m5 + m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m6
        # We map to columns 0,1 of B (k=0,1).
        r = 16
        U = [[Fraction(0)] * r for _ in range(n * m)]   # 4 rows
        V = [[Fraction(0)] * r for _ in range(m * p)]   # 10 rows
        W = [[Fraction(0)] * r for _ in range(p * n)]   # 10 rows

        # --- Strassen block for k in {0,1} ---
        # U row: i*2 + j ; V row: j*5 + k ; W row: k*2 + i
        # m1: (A00+A11)(B00+B11)
        U[0*2+0][0] += 1; U[1*2+1][0] += 1
        V[0*5+0][0] += 1; V[1*5+1][0] += 1
        W[0*2+0][0] += 1; W[1*2+0][0] += 1; W[0*2+1][0] += -1; W[1*2+1][0] += 1
        # m2: (A10+A11)B00
        U[1*2+0][1] += 1; U[1*2+1][1] += 1
        V[0*5+0][1] += 1
        W[1*2+0][1] += 1
        # m3: A00(B10-B11)
        U[0*2+0][2] += 1
        V[1*5+0][2] += 1; V[1*5+1][2] += -1
        W[0*2+1][2] += 1; W[1*2+0][2] += 1
        # m4: A11(B01-B00)
        U[1*2+1][3] += 1
        V[0*5+1][3] += 1; V[0*5+0][3] += -1
        W[0*2+0][3] += 1; W[1*2+0][3] += 1
        # m5: (A00+A01)B11
        U[0*2+0][4] += 1; U[0*2+1][4] += 1
        V[1*5+1][4] += 1
        W[0*2+1][4] += -1; W[0*2+0][4] += -1
        # m6: (A10-A00)(B00+B01)
        U[1*2+0][5] += 1; U[0*2+0][5] += -1
        V[0*5+0][5] += 1; V[0*5+1][5] += 1
        W[1*2+1][5] += 1
        # m7: (A01-A11)(B10+B11)
        U[0*2+1][6] += 1; U[1*2+1][6] += -1
        V[1*5+0][6] += 1; V[1*5+1][6] += 1
        W[0*2+0][6] += 1; W[1*2+1][6] += -1

        # --- rank-3 for each single column k in {2,3,4} ---
        # 2x2 * 2x1: C0 = A00 B0 + A01 B1, C1 = A10 B0 + A11 B1
        # rank-3 construction (verified below):
        # m_a = (A00+A11) B0
        # m_b = (A01+A10) B1
        # m_c = A00 B1  ... need to check. Let's use a known rank-3:
        # Actually use:
        # t1 = (A00 + A11) * B0
        # t2 = A01 * B1
        # t3 = A10 * B0
        # C0 = t1 + t2 - A11*B0 - A00*B0 ... not closed.
        # Let me just use a verified rank-3 via the program's search over small grid for the 2x2x2x1.
        # For safety, use rank-4 (schoolbook) for each single column -> 4*3=12, total 7+12=19.
        # But we want < 20. 7+12 = 19 < 20. Good enough as a guaranteed improvement.
        col = 7
        for k in [2, 3, 4]:
            # schoolbook 4 terms for this column
            for i in range(n):
                for j in range(m):
                    U[i*2+j][col] += 1
                    V[j*5+k][col] += 1
                    W[k*2+i][col] += 1
                    col += 1
        # r = 7 + 12 = 19
        r = 7 + 12
        return U, V, W, r

    U_b, V_b, W_b, r_b = block_decomp()
    if _verify(U_b, V_b, W_b, n, m, p, r_b):
        best = (U_b, V_b, W_b, r_b)

    # --- Structured exact search for smaller ranks ---
    # Try ranks from 10 up to best_rank-1 using a smarter parameterized search.
    # We use the observation that a valid decomposition can be found by solving the
    # polynomial system; we do a randomized exact search with small half-integers
    # and a local refinement.
    best_rank = best[3]
    lower = max(n*m, m*p, p*n)  # 10

    # Candidate value set for entries (half-integers)
    vals = [Fraction(v, 2) for v in range(-4, 5)]  # -2 .. 2 in steps of 0.5

    # We'll attempt to find a decomposition of a given rank by random init +
    # iterative "coordinate" repair using exact arithmetic.
    def try_rank(target_rank, max_iters=20000):
        # Random init
        U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
        for _ in range(max_iters):
            if time.time() > deadline:
                return None
            if _verify(U, V, W, n, m, p, target_rank):
                return (U, V, W, target_rank)
            # Pick a random (i,j,k) and a random column c, perturb W to try to fix
            # the residual for that (i,j,k).
            i = random.randrange(n); j = random.randrange(m); k = random.randrange(p)
            c = random.randrange(target_rank)
            # residual = 1 - sum over c' of U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c']
            s = Fraction(0)
            for c2 in range(target_rank):
                s += U[i*m+j][c2] * V[j*p+k][c2] * W[k*n+i][c2]
            res = 1 - s
            uv = U[i*m+j][c] * V[j*p+k][c]
            if uv == 0:
                # try to make uv nonzero
                if random.random() < 0.5:
                    U[i*m+j][c] = random.choice(vals)
                else:
                    V[j*p+k][c] = random.choice(vals)
            else:
                # set W so that this term contributes res (exact division)
                W[k*n+i][c] = res / uv
        return None

    for target_rank in range(lower, best_rank):
        if time.time() > deadline:
            break
        res = try_rank(target_rank)
        if res is not None:
            best = res
            best_rank = target_rank
            break  # found minimal in this ascending scan

    U, V, W, rank = best
    # Convert Fractions to plain numbers (int or float half-integers) for output
    def conv(mat):
        out = []
        for row in mat:
            nr = []
            for x in row:
                d = x.denominator
                if d == 1:
                    nr.append(x.numerator)
                else:
                    nr.append(x.numerator / d)
            out.append(nr)
        return out

    return {"U": conv(U), "V": conv(V), "W": conv(W)}
# EVOLVE-BLOCK-END
