import time
import random
from fractions import Fraction

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i

def make_strassen_2x2x2():
    """Return 7 rank-1 terms for 2x2x2 matmul. Each term is (u, v, w) dicts."""
    terms = []
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    terms.append({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
    # M2: (A10+A11)(B00) -> C10 - C11
    terms.append({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1})
    # M3: (A00)(B01-B11) -> C01 + C11
    terms.append({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1})
    # M4: (A11)(B10-B00) -> C00 + C10
    terms.append({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1})
    # M5: (A00+A01)(B11) -> -C00 + C01
    terms.append({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1})
    # M6: (A10-A00)(B00+B01) -> C11
    terms.append({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1})
    # M7: (A01-A11)(B10+B11) -> C00
    terms.append({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1})
    return terms

def verify_decomposition(U, V, W, r):
    """Check exact decomposition. U,V,W are lists of lists (rows x rank)."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for l in range(r):
                    s += U[Aidx(i,j)][l] * V[Bidx(j,k)][l] * W[Cidx(i,k)][l]
                if s != 1:
                    return False
    return True

def build_from_terms(terms, a_off, b_off, c_off):
    """
    Build U, V, W contributions from a list of (u, v, w) term dicts.
    u: {(ai,aj): val}, v: {(bj,bk): val}, w: {(ci,ck): val}
    a_off, b_off, c_off are offsets for the block position.
    Returns (U_contrib, V_contrib, W_contrib, rank_used)
    """
    r = len(terms)
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    for l, (u, v, w) in enumerate(terms):
        for (ai, aj), val in u.items():
            if val != 0:
                U[Aidx(ai + a_off, aj)][l] += val
        for (bj, bk), val in v.items():
            if val != 0:
                V[Bidx(bj + b_off, bk)][l] += val
        for (ci, ck), val in w.items():
            if val != 0:
                W[Cidx(ci + c_off, ck)][l] += val
    return U, V, W, r

def combine_blocks(block_list):
    """
    block_list: list of (U, V, W, r) for non-overlapping blocks.
    Returns combined (U, V, W, total_r)
    """
    total_r = sum(b[3] for b in block_list)
    U = [[0]*total_r for _ in range(n*m)]
    V = [[0]*total_r for _ in range(m*p)]
    W = [[0]*total_r for _ in range(p*n)]
    
    offset = 0
    for (Ub, Vb, Wb, rb) in block_list:
        for row in range(n*m):
            for l in range(rb):
                if Ub[row][l] != 0:
                    U[row][offset + l] = Ub[row][l]
        for row in range(m*p):
            for l in range(rb):
                if Vb[row][l] != 0:
                    V[row][offset + l] = Vb[row][l]
        for row in range(p*n):
            for l in range(rb):
                if Wb[row][l] != 0:
                    W[row][offset + l] = Wb[row][l]
        offset += rb
    return U, V, W, total_r

def schoolbook_block(rows_a, rows_b, cols_b, rows_c, cols_c):
    """
    Schoolbook decomposition for a block: A is rows_a x cols_a, B is cols_a x cols_b, C is rows_c x cols_c.
    Actually for matmul: A[i,j] * B[j,k] = C[i,k].
    For a block where A covers rows [ar, ar+ra), cols [ac, ac+ca),
    B covers rows [br, br+cb), cols [bc, bc+cc),
    C covers rows [cr, cr+rc), cols [ck, ck+cc2).
    For schoolbook: ra*cb*cc2 terms.
    """
    # This is the simple term-by-term: each (i,j,k) in the block gets one term
    terms = []
    for i in range(rows_a):
        for j in range(rows_b):
            for k in range(cols_c):
                u = {(0, j): 1}  # local coords
                v = {(j, k): 1}
                w = {(i, k): 1}
                terms.append((u, v, w))
    return terms

def strassen_2x2_2x2_offset(a_off, b_off, c_off):
    """Get strassen terms with global offsets."""
    base = make_strassen_2x2x2()
    terms = []
    for (u, v, w) in base:
        u_new = {(ai + a_off, aj): val for (ai, aj), val in u.items()}
        v_new = {(bj + b_off, bk): val for (bj, bk), val in v.items()}
        w_new = {(ci + c_off, ck): val for (ci, ck), val in w.items()}
        terms.append((u_new, v_new, w_new))
    return terms

def build_partition_a3_b4_c6():
    """
    Partition: A rows into [0], [1], [2]; B cols into [0,1,2,3]; C cols into [0,1,2,3,4,5].
    For each row of A and all of B,C:
    - If we use 1x4x4 blocks: 4*4 = 16 terms each, 3*16 = 48 total.
    That's worse. Let's use a smarter approach.
    
    Alternative: Split A into 3 rows, B into 2+2, C into 3+3.
    - A[0] x B[0:2] x C[0:3]: 1x2x3 schoolbook = 2*3 = 6 terms
    - A[0] x B[2:4] x C[3:6]: 1x2x3 = 6 terms
    - A[1] x B[0:2] x C[0:3]: 6 terms
    - A[1] x B[2:4] x C[3:6]: 6 terms
    - A[2] x B[0:2] x C[0:3]: 6 terms
    - A[2] x B[2:4] x C[3:6]: 6 terms
    Total: 36. Much better!
    """
    blocks = []
    # Block config: (a_rows, b_cols, c_cols) with offsets
    configs = [
        # (a_start, b_start, c_start, a_size, b_size, c_size)
        (0, 0, 0, 1, 2, 3),
        (0, 2, 3, 1, 2, 3),
        (1, 0, 0, 1, 2, 3),
        (1, 2, 3, 1, 2, 3),
        (2, 0, 0, 1, 2, 3),
        (2, 2, 3, 1, 2, 3),
    ]
    
    for (ar, br, cr, asz, bsz, csz) in configs:
        terms = []
        for i in range(asz):
            for j in range(bsz):
                for k in range(csz):
                    u = {((ar + i), (br + j)): 1}
                    v = {((br + j), (cr + k)): 1}
                    w = {((ar + i), (cr + k)): 1}
                    terms.append((u, v, w))
        U, V, W, r = build_from_terms(terms, 0, 0, 0)
        blocks.append((U, V, W, r))
    
    return combine_blocks(blocks)

def build_partition_strassen():
    """
    Original approach: A split as [0], [1,2]; B split as [0,1,2,3] into 2+2; C split as [0,1,2,3,4,5] into 3+3.
    - A[0] x B[0:2] x C[0:3]: 1x2x3 schoolbook = 6
    - A[0] x B[2:4] x C[3:6]: 1x2x3 schoolbook = 6
    - A[1:3] x B[0:2] x C[0:3]: 2x2x3. Use Strassen-like?
    
    Actually let's do:
    A: 3 rows -> split as [0] and [1,2]
    B: 4 cols -> split as [0,1] and [2,3]  
    C: 6 cols -> split as [0,1,2] and [3,4,5]
    
    For A[1:3] x B[0:2] x C[0:3]: 2x2x3
    Use Strassen for 2x2 part, then handle the 3rd col of C separately?
    Or use a 2x2x3 decomposition.
    
    Let me use: for 2x2x3, split C into 2+1.
    2x2x2 part: Strassen (7 terms)
    2x2x1 part: 2*2*1 = 4 terms (schoolbook)
    Total: 11 per 2x2x3 block.
    
    Blocks:
    - A[0] x B[0:2] x C[0:3]: 1x2x3 = 6
    - A[0] x B[2:4] x C[3:6]: 1x2x3 = 6
    - A[1:3] x B[0:2] x C[0:3]: 2x2x3 = 11
    - A[1:3] x B[2:4] x C[3:6]: 2x2x3 = 11
    Total: 6+6+11+11 = 34
    """
    blocks = []
    
    # Helper: build a 2x2x3 block using Strassen(2x2x2) + schoolbook(2x2x1)
    def build_2x2x3(a_off, b_off, c_off):
        terms = []
        # Strassen part for first 2 cols of C
        s_terms = strassen_2x2_2x2_offset(a_off, b_off, c_off)
        terms.extend(s_terms)
        # Schoolbook for 3rd col of C (index 2)
        for i in range(2):
            for j in range(2):
                u = {((a_off+i), (b_off+j)): 1}
                v = {((b_off+j), (c_off+2)): 1}
                w = {((a_off+i), (c_off+2)): 1}
                terms.append((u, v, w))
        U, V, W, r = build_from_terms(terms, 0, 0, 0)
        return (U, V, W, r)
    
    # Helper: build 1x2x3 block (schoolbook)
    def build_1x2x3(a_off, b_off, c_off):
        terms = []
        for j in range(2):
            for k in range(3):
                u = {((a_off), (b_off+j)): 1}
                v = {((b_off+j), (c_off+k)): 1}
                w = {((a_off), (c_off+k)): 1}
                terms.append((u, v, w))
        U, V, W, r = build_from_terms(terms, 0, 0, 0)
        return (U, V, W, r)
    
    blocks.append(build_1x2x3(0, 0, 0))    # A[0] B[0:2] C[0:3]
    blocks.append(build_1x2x3(0, 2, 3))    # A[0] B[2:4] C[3:6]
    blocks.append(build_2x2x3(1, 0, 0))    # A[1:3] B[0:2] C[0:3]
    blocks.append(build_2x2x3(1, 2, 3))    # A[1:3] B[2:4] C[3:6]
    
    return combine_blocks(blocks)

def build_partition_strassen_full():
    """
    Use Strassen for ALL 2x2x2 subblocks.
    A: 3 rows -> [0], [1,2]
    B: 4 cols -> [0,1], [2,3]
    C: 6 cols -> [0,1,2,3,4,5] -> [0,1], [2,3], [4,5]
    
    Blocks:
    - A[0] x B[0:2] x C[0:2]: 1x2x2 = 4
    - A[0] x B[0:2] x C[2:4]: 1x2x2 = 4
    - A[0] x B[0:2] x C[4:6]: 1x2x2 = 4
    - A[0] x B[2:4] x C[0:2]: 4
    - A[0] x B[2:4] x C[2:4]: 4
    - A[0] x B[2:4] x C[4:6]: 4
    - A[1:3] x B[0:2] x C[0:2]: 2x2x2 = 7
    - A[1:3] x B[0:2] x C[2:4]: 7
    - A[1:3] x B[0:2] x C[4:6]: 7
    - A[1:3] x B[2:4] x C[0:2]: 7
    - A[1:3] x B[2:4] x C[2:4]: 7
    - A[1:3] x B[2:4] x C[4:6]: 7
    Total: 6*4 + 6*7 = 24 + 42 = 66. Same as original!
    
    Let me try a different split for C: [0,1,2], [3,4,5]
    - A[0] x B[0:2] x C[0:3]: 1x2x3 = 6
    - A[0] x B[0:2] x C[3:6]: 6
    - A[0] x B[2:4] x C[0:3]: 6
    - A[0] x B[2:4] x C[3:6]: 6
    - A[1:3] x B[0:2] x C[0:3]: 2x2x3 = 11
    - A[1:3] x B[0:2] x C[3:6]: 11
    - A[1:3] x B[2:4] x C[0:3]: 11
    - A[1:3] x B[2:4] x C[3:6]: 11
    Total: 4*6 + 4*11 = 24 + 44 = 68. Worse.
    """
    blocks = []
    
    def build_1x2x2(a_off, b_off, c_off):
        terms = []
        for j in range(2):
            for k in range(2):
                u = {((a_off), (b_off+j)): 1}
                v = {((b_off+j), (c_off+k)): 1}
                w = {((a_off), (c_off+k)): 1}
                terms.append((u, v, w))
        U, V, W, r = build_from_terms(terms, 0, 0, 0)
        return (U, V, W, r)
    
    def build_2x2x2(a_off, b_off, c_off):
        terms = strassen_2x2_2x2_offset(a_off, b_off, c_off)
        U, V, W, r = build_from_terms(terms, 0, 0, 0)
        return (U, V, W, r)
    
    # 6 blocks of 1x2x2
    for bi in range(2):
        for ci in range(3):
            blocks.append(build_1x2x2(0, bi*2, ci*2))
    # 6 blocks of 2x2x2
    for bi in range(2):
        for ci in range(3):
            blocks.append(build_2x2x2(1, bi*2, ci*2))
    
    return combine_blocks(blocks)

def local_search(U, V, W, r, deadline, rng):
    """
    Local search: try to drop a term, or replace 2 terms with 1.
    """
    best_r = r
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    def can_drop(arr, indices_to_keep, l):
        """Check if removing column l still gives exact decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for ll in range(arr[0].__len__()):
                        if ll == l:
                            continue
                        s += U[Aidx(i,j)][ll] * V[Bidx(j,k)][ll] * W[Cidx(i,k)][ll]
                    if s != 1:
                        return False
        return True
    
    def remove_column(U, V, W, l):
        newU = [row[:l] + row[l+1:] for row in U]
        newV = [row[:l] + row[l+1:] for row in V]
        newW = [row[:l] + row[l+1:] for row in W]
        return newU, newV, newW
    
    # First pass: try dropping individual terms
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for l in range(best_r):
            if can_drop(best_U, best_V, best_W, l):
                best_U, best_V, best_W = remove_column(best_U, best_V, best_W, l)
                best_r -= 1
                changed = True
                break
    
    # Try 2-to-1 replacement: check if columns l1, l2 can be merged
    # This is hard to check exactly. Skip for now.
    
    return best_r, best_U, best_V, best_W

def construct():
    deadline = time.time() + 140
    rng = random.Random(42)
    
    # Try multiple partition strategies
    candidates = []
    
    # Strategy 1: Pure schoolbook with fine partition (36 terms)
    U1, V1, W1, r1 = build_partition_a3_b4_c6()
    if verify_decomposition(U1, V1, W1, r1):
        candidates.append((r1, U1, V1, W1, "fine_schoolbook"))
    
    # Strategy 2: Strassen-based (34 terms)
    U2, V2, W2, r2 = build_partition_strassen()
    if verify_decomposition(U2, V2, W2, r2):
        candidates.append((r2, U2, V2, W2, "strassen_mixed"))
    
    # Strategy 3: Full Strassen (66 terms)
    U3, V3, W3, r3 = build_partition_strassen_full()
    if verify_decomposition(U3, V3, W3, r3):
        candidates.append((r3, U3, V3, W3, "full_strassen"))
    
    # Pick the best
    if not candidates:
        # Fallback to original 66
        U, V, W = baseline_66()
        r = 66
    else:
        candidates.sort(key=lambda x: x[0])
        r, U, V, W, name = candidates[0]
    
    # Local search to reduce further
    r, U, V, W = local_search(U, V, W, r, deadline, rng)
    
    assert verify_decomposition(U, V, W, r), "Final decomposition is not exact!"
    
    return {"U": U, "V": V, "W": W}

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return U, V, W

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
