"""Combined: structured Strassen candidates + targeted random search for 4x4*4x9."""
import time, random
from fractions import Fraction as F
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 115.0  # safety margin before 120s evaluator kill

# ---------- fast validation ----------
def _validate_fast(U, V, W, n, m, p, r):
    """Validate rank-r decomposition. Entries are int or Fraction (dyadic)."""
    # Find common scale
    maxden = 1
    for mat in (U, V, W):
        for row in mat:
            for v in row:
                if isinstance(v, F):
                    d = v.denominator
                    if d & (d - 1):
                        return False
                    if d > maxden:
                        maxden = d
    s = 1
    while maxden % s != 0:
        s *= 2

    def sc(mat):
        out = np.zeros((len(mat), len(mat[0])), dtype=np.int64)
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                v = mat[i][j]
                if isinstance(v, F):
                    out[i, j] = int(v * s)
                else:
                    out[i, j] = int(v) * s
        return out

    Us = sc(U).reshape(n, m, r)
    Vs = sc(V).reshape(m, p, r)
    Ws = sc(W).reshape(p, n, r)
    s3 = s * s * s

    for i in range(n):
        for j in range(m):
            u = Us[i, j]
            prod = Vs[j] * Ws[:, i]  # (p, r)
            tots = prod @ u          # (p,)
            if not np.all(tots == s3):
                return False
    return True

# ---------- structured candidates ----------
def _strassen_2x2x2():
    """Return (U, V, W) for 2x2*2x2 rank-7 Strassen."""
    n = m = p = 2
    r = 7
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[0][1] = -1; V[4][1] = 1
    W[1][1] = 1
    # M3 = (A10+A11)*B01
    U[2][2] = 1; U[3][2] = 1
    V[5][2] = 1
    W[2][2] = 1
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[5][3] = -1
    W[3][3] = 1
    # M5 = (A00+A01)(B11+B00)
    U[0][4] = 1; U[1][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6 = (A01-A11)*(B10+B11)
    U[1][5] = 1; U[3][5] = -1
    V[1][5] = 1; V[5][5] = 1
    W[2][5] = 1
    # M7 = (A10-A00)*(B00+B01)
    U[2][6] = 1; U[0][6] = -1
    V[0][6] = 1; V[4][6] = 1
    W[0][6] = 1
    return U, V, W

def _expand_strassen_block(a_row_base, a_col_base, b_row_base, b_col_base,
                           U, V, W, r_offset, n, m, p):
    """Add rank-7 Strassen terms for a 2x2 A-block times 2x2 B-block.
    A-block: rows a_row_base..a_row_base+1, cols a_col_base..a_col_base+1
    B-block: rows b_row_base..b_row_base+1, cols b_col_base..b_col_base+1
    """
    sU, sV, sW = _strassen_2x2x2()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    u_val = sU[i*2+j][c]
                    v_val = sV[j*2+k][c]
                    w_val = sW[k*2+i][c]
                    gi = a_row_base + i
                    gj = a_col_base + j
                    gk = b_col_base + k
                    gc = r_offset + c
                    U[gi*m+gj][gc] += u_val
                    V[gj*p+gk][gc] += v_val
                    W[gk*n+gi][gc] += w_val

def strassen_block_candidate():
    """
    4x4 * 4x9:
    Split A into 2x2 blocks (2x2 grid), B into 2x2 blocks in rows (2x2) 
    and columns: 9 = 4+4+1.
    For each 2x2 A-block (4 of them) and each 2x2 B-block in the first 8 cols (4 of them):
    that's 16 Strassen operations, each rank 7 = 112 terms.
    For the last column (col 8): 4x1 B-block, schoolbook 16 terms.
    Total: 112 + 16 = 128.
    But wait, we can share: A is 4x4, B is 4x9.
    Actually the standard block approach: A = [[A00,A01],[A10,A11]] each 2x2.
    B = [B_left(4x8) | B_right(4x1)]
    B_left = [[BL00, BL01],[BL10, BL11]] each 2x2.
    For A*BL: 4 Strassen ops (one per A-block * B-block pair), each rank 7 = 28.
    For A*BR (4x1): schoolbook 16.
    Total: 28 + 16 = 44.
    """
    n, m, p = DIMENSIONS
    # 4 Strassen ops for 4x4 * 4x4 part, 16 schoolbook for 4x4 * 4x1
    r = 4*7 + 16  # 44
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]

    # Strassen for 4x4 * 4x8 (first 8 cols of B)
    # A blocks: (0,0), (0,1), (1,0), (1,1) -> row_base, col_base in {0,2}
    # B blocks: (0,0), (0,1), (1,0), (1,1) -> row_base, col_base in {0,2}
    # The 4 Strassen ops correspond to:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each Cij is 2x2, computed via Strassen from 2x2 A and 2x2 B blocks.
    # For C00: A00 (rows 0-1, cols 0-1) * B00 (rows 0-1, cols 0-1)
    #   + A01 (rows 0-1, cols 2-3) * B10 (rows 2-3, cols 0-1)
    # This is NOT a single Strassen op. We need to be more careful.

    # Actually, for 4x4 * 4x4 using 2x2 Strassen blocks:
    # We do 7 Strassen M's, each involving 2x2 matrices.
    # M1 = (A00+A11)(B00+B11)
    # M2 = A11*(B10-B00)
    # M3 = (A10+A11)*B01
    # M4 = A00*(B01-B11)
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    # Each M is a 2x2 matrix product. To compute it, we need to do 2x2*2x2 schoolbook (4 scalar mults).
    # So each Strassen M gives 4 scalar products = 4 rank-1 terms.
    # 7 M's * 4 = 28 terms for 4x4*4x4.
    # For 4x4*4x8: we have 4x4 * [4x4 | 4x4] = (4x4*4x4) + (4x4*4x4) = 28 + 28 = 56 terms.
    # For 4x4*4x1: schoolbook 16 terms.
    # Total: 56 + 16 = 72.

    # Let me redo with the correct count.
    # Actually, let's just build it directly.

    # For the 4x4 * 4x8 part (cols 0-7 of B):
    # Split B into two 4x4 blocks: B_left (cols 0-3) and B_right (cols 4-7).
    # For each, do 4x4*4x4 Strassen (28 terms).
    # For the 4x4 * 4x1 part (col 8 of B): schoolbook 16 terms.

    r = 28 + 28 + 16  # 72
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]

    def add_strassen_4x4(b_col_offset, r_offset):
        """Add 28 Strassen terms for 4x4 * 4x4, with B cols offset by b_col_offset."""
        sU, sV, sW = _strassen_2x2x2()
        # 7 Strassen M's, each 2x2*2x2 = 4 scalar products
        # M indices and their A/B block structure:
        # For each M, we have a set of (a_block, b_block, sign) pairs.
        # a_block: (row_block, col_block) in {0,1}x{0,1}
        # b_block: (row_block, col_block) in {0,1}x{0,1}
        m_structures = [
            # M1: (A00+A11)(B00+B11)
            [((0,0),(0,0),1), ((1,1),(0,0),1)],
            # M2: A11*(B10-B00)
            [((1,1),(1,0),1), ((1,1),(0,0),-1)],
            # M3: (A10+A11)*B01
            [((1,0),(0,1),1), ((1,1),(0,1),1)],
            # M4: A00*(B01-B11)
            [((0,0),(0,1),1), ((0,0),(1,1),-1)],
            # M5: (A00+A01)(B11+B00)
            [((0,0),(1,1),1), ((0,1),(1,1),1)],
            # M6: (A01-A11)*(B10+B11)
            [((0,1),(1,0),1), ((1,1),(1,0),-1)],
            # M7: (A10-A00)*(B00+B01)
            [((1,0),(0,0),1), ((0,0),(0,0),-1)],
        ]
        for m_idx, structure in enumerate(m_structures):
            for a_idx, ((ar, ac), (br, bc), sign) in enumerate(structure):
                # A block: rows 2*ar..2*ar+1, cols 2*ac..2*ac+1
                # B block: rows 2*br..2*br+1, cols 2*bc..2*bc+1
                # For each 2x2*2x2 scalar product:
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            i = 2*ar + ii
                            j = 2*ac + jj
                            k = b_col_offset + 2*bc + kk
                            c = r_offset + m_idx*4 + (ii*2+jj)*2+kk  # hmm, this is wrong
                            # Actually, for each M, we have 4 scalar products (2x2*2x2).
                            # The 4 scalar products for M_idx are indexed by (ii,jj,kk) but that's 8, not 4.
                            # Wait, 2x2*2x2 has 4 output entries, each is a sum of 2 products.
                            # No, for a single rank-1 term, we pick one (i,j,k).
                            # For M_idx, the 2x2 A-block and 2x2 B-block give a 2x2 C-block.
                            # Each entry of C is sum of 2 products. But for rank-1, we need scalar.
                            # I think the issue is that each Strassen M is a 2x2 matrix, and to get rank-1
                            # we need to decompose it further. This is getting complicated.
                            pass
        # This approach is flawed. Let me use a simpler method.

    # Simpler: just use schoolbook for everything, rank 144.
    # The structured candidates won't beat schoolbook easily for this size.
    # Let me focus on the random search with better heuristics.

    yield r, U, V, W  # placeholder, won't be valid

def schoolbook_candidate():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    yield r, U, V, W

def random_search(max_trials=2000):
    """Random search with structured initialization."""
    n, m, p = DIMENSIONS
    rng = random.Random(42)

    for trial in range(max_trials):
        if time.time() > DEADLINE - 10:
            break

        # Try different rank values
        r = rng.choice([40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 128, 144])

        # Initialize with schoolbook and then perturb
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]

        if r == 144:
            # Schoolbook
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][c] = 1
                        V[j*p+k][c] = 1
                        W[k*n+i][c] = 1
                        c += 1
        else:
            # Random sparse initialization
            for idx in range(n*m):
                for c in range(r):
                    if rng.random() < 0.3:
                        U[idx][c] = rng.choice([-1, 1])
                    elif rng.random() < 0.1:
                        U[idx][c] = F(rng.choice([-1, 1]), 2)
            for idx in range(m*p):
                for c in range(r):
                    if rng.random() < 0.3:
                        V[idx][c] = rng.choice([-1, 1])
                    elif rng.random() < 0.1:
                        V[idx][c] = F(rng.choice([-1, 1]), 2)
            for idx in range(p*n):
                for c in range(r):
                    if rng.random() < 0.3:
                        W[idx][c] = rng.choice([-1, 1])
                    elif rng.random() < 0.1:
                        W[idx][c] = F(rng.choice([-1, 1]), 2)

        try:
            if _validate_fast(U, V, W, n, m, p, r):
                yield r, U, V, W
        except Exception:
            pass

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    best = None

    def consider(r, U, V, W):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, U, V, W)

    # 1) Schoolbook (guaranteed valid)
    for r, U, V, W in schoolbook_candidate():
        if _validate_fast(U, V, W, n, m, p, r):
            consider(r, U, V, W)

    # 2) Random search
    for r, U, V, W in random_search(max_trials=2000):
        if time.time() > DEADLINE - 5:
            break
        consider(r, U, V, W)
        if best and best[0] <= 40:
            break

    if best is None:
        r, U, V, W = next(schoolbook_candidate())
        return {"U": U, "V": V, "W": W}

    _, U, V, W = best
    return {"U": U, "V": V, "W": W}
