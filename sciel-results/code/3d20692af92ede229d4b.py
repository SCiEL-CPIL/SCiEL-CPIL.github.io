"""Combined: Strassen-block decomposition for 4x4 x 4x7 matrix multiplication."""
import time

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = _START + 140.0


def _strassen_2x2_terms():
    """7 Strassen terms for 2x2x2. Each term: (U4, V4, W4) for flattened 2x2."""
    # A=[a,b;c,d] -> idx 0,1,2,3; B=[e,f;g,h] -> idx 0,1,2,3; C=[i,j;k,l] -> idx 0,1,2,3
    # C0=M1-M4+M5-M7, C1=M3+M5, C2=M2+M4, C3=M1-M2+M6-M7
    terms = []
    # M1=(a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    # M2=(c+d)(e): U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    # M3=a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    # M4=d(g-e): U=[0,0,0,1], V=[0,0,1,-1], W=[-1,0,1,0]
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    # M5=(a+b)(h): U=[1,1,0,0], V=[0,0,0,1], W=[1,1,0,0]
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    # M6=(c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # M7=(b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,-1]
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    return terms


def _embed_2x2(vec, row_off, col_off, n):
    """Embed 2x2 flat vec into n x n flat vector at (row_off, col_off)."""
    res = [0] * (n * n)
    for idx, val in enumerate(vec):
        r = idx // 2
        c = idx % 2
        res[(row_off + r) * n + (col_off + c)] = val
    return res


def _embed_2x3(vec, row_off, col_off, n, p):
    """Embed 2x3 flat vec into n x p flat vector at (row_off, col_off)."""
    res = [0] * (n * p)
    for idx, val in enumerate(vec):
        r = idx // 3
        c = idx % 3
        res[(row_off + r) * p + (col_off + c)] = val
    return res


def _add_to(vec, target):
    for i in range(len(vec)):
        target[i] += vec[i]


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    def _verify(res):
        U, V, W = res["U"], res["V"], res["W"]
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    # Part 1: 4x4 x 4x4 -> 4x4 (k=0..3), using Strassen on 2x2 blocks
    # Part 2: 4x4 x 4x3 -> 4x3 (k=4..6), schoolbook on 2x2 x 2x3 blocks
    
    U = []  # will be list of lists, each of length rank
    V = []
    W = []
    
    # We'll build terms as (U_vec, V_vec, W_vec) and then transpose to matrix form
    terms = []
    
    strassen = _strassen_2x2_terms()
    
    # Part 1: 4x4 x 4x4 x 4 (indices k=0..3)
    # Block positions: (0,0),(0,1),(1,0),(1,1) for 2x2 blocks in 4x4
    block_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each block multiplication, apply Strassen
    # A block (ar, ac) * B block (br, bc) contributes to C block (ar, bc)
    
    for (ar, ac) in block_pos:
        for (br, bc) in block_pos:
            # This product contributes to C block (ar, bc)
            for (su, sv, sw) in strassen:
                U_vec = _embed_2x2(su, ar * 2, ac * 2, 4)
                V_vec = _embed_2x2(sv, br * 2, bc * 2, 4)
                W_vec = _embed_2x2(sw, ar * 2, bc * 2, 4)
                terms.append((U_vec, V_vec, W_vec))
    
    # Part 2: 4x4 x 4x3 x 4 (indices k=4..6)
    # A is 4x4, B is 4x3, C is 4x3
    # Split A into 2x2 blocks: A00(0,0), A01(0,1), A10(1,0), A11(1,1)
    # Split B into 2x3 blocks: B0(rows 0-1, cols 0-2), B1(rows 2-3, cols 0-2)
    # C: C0(rows 0-1, cols 0-2), C1(rows 2-3, cols 0-2)
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Each 2x2 x 2x3: schoolbook, 2*2*3 = 12 terms? No, 2*3=6 output entries, each 2 products.
    # Schoolbook: for each (i,j,k) in 2x2x3: one term. That's 12 terms per 2x2x2x3.
    # Wait: 2x2 matrix times 2x3 matrix. Output is 2x3. C[i,k] = sum_j A[i,j]*B[j,k].
    # Schoolbook: for each i in 0..1, j in 0..1, k in 0..2: one term. 2*2*3=6 terms.
    
    for (ar, ac) in block_pos:
        # A block at (ar, ac), which is 2x2 at row ar*2, col ac*2
        for br in [0, 1]:  # B block row: 0 for top 2x3, 1 for bottom 2x3
            # B block is 2x3 at row br*2, col 0
            for i in range(2):  # row within 2x2
                for j in range(2):  # col within 2x2 (also row in B)
                    for k in range(3):  # col in 2x3
                        U_vec = [0] * 16
                        U_vec[(ar * 2 + i) * 4 + (ac * 2 + j)] = 1
                        V_vec = [0] * 28
                        V_vec[(br * 2 + j) * 7 + k] = 1
                        W_vec = [0] * 28
                        W_vec[(ar * 2 + i) * 7 + k] = 1
                        terms.append((U_vec, V_vec, W_vec))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(28)]
    W = [[0] * rank for _ in range(28)]
    
    for c, (U_vec, V_vec, W_vec) in enumerate(terms):
        for i in range(16):
            if U_vec[i] != 0:
                U[i][c] = U_vec[i]
        for i in range(28):
            if V_vec[i] != 0:
                V[i][c] = V_vec[i]
        for i in range(28):
            if W_vec[i] != 0:
                W[i][c] = W_vec[i]
    
    res = {"U": U, "V": V, "W": W}
    
    if _verify(res):
        return res
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
