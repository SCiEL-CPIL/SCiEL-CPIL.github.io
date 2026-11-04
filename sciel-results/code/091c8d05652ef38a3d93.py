"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    nm, mp, pn = n*m, m*p, p*n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _strassen_2x2(A, B):
    # A, B are 2x2; returns 2x2 = A*B using 7 products
    M1 = (A[1,1] - A[0,1]) * (B[0,0] + B[0,1])
    M2 = (A[0,0] + A[1,0]) * B[0,0]
    M3 = A[0,1] * (B[1,0] + B[1,1])
    M4 = A[1,1] * (B[1,1] - B[0,1])
    M5 = (A[0,0] + A[0,1]) * B[1,1]
    M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    return np.array([
        [M1 + M4 + M5 - M7, M5 + M6],
        [M3 + M7, M1 - M3 + M4 + M6]
    ])

def construct():
    n, m, p = DIMENSIONS  # 4,4,7
    nm, mp, pn = n*m, m*p, p*n
    T = _build_tensor(n, m, p)
    deadline = time.time() + 140  # safety margin before 150s kill

    # ---- Build a structured baseline: Strassen on 4x4*4x4, schoolbook on 4x4*4x3 ----
    # We will build U,V,W as lists of (row, col, val) then assemble.
    def add(U, V, W, i, j, k, u, v, w):
        U.append((i, u)); V.append((j, v)); W.append((k, w))

    # We'll accumulate factor matrices as dicts: factor -> list of (index, value) per rank-col
    U_rows = {}  # row -> {col: val}
    V_rows = {}
    W_rows = {}
    col = 0
    def new_col(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        U_rows.setdefault(u_idx, {})[col] = U_rows.get(u_idx, {}).get(col, 0) + u_val
        V_rows.setdefault(v_idx, {})[col] = V_rows.get(v_idx, {}).get(col, 0) + v_val
        W_rows.setdefault(w_idx, {})[col] = W_rows.get(w_idx, {}).get(col, 0) + w_val
        col += 1

    # ---- 4x4 * 4x4 part: Strassen over 2x2 blocks ----
    # A blocks: Aab with a,b in {0,1}; A row index = 2*a + i, col index = 2*b + j
    # B (4x4) blocks: Bcd with c,d in {0,1}; B row = 2*c + j, col = 2*d + k
    # C = A*B, C row = 2*a+i, col = 2*d+k
    # Strassen products P1..P7 are linear combos of A blocks times B blocks.
    # We express each product as: (linear combo of A entries) * (linear combo of B entries)
    # giving C contributions.
    # For each of the 7 products, the A-side is a linear combination of 4 A-blocks (each 2x2 = 4 entries)
    # and B-side is a linear combination of 4 B-blocks.
    # To keep rank low we treat each 2x2*2x2 Strassen product as a single "product" but that product
    # itself is a 2x2*2x2 multiply needing 8 scalar products (schoolbook) OR we recursively use Strassen.
    # Simpler: for each 2x2*2x2 block multiply, use schoolbook (8) -> 4 blocks * 8 = 32 for 4x4*4x4.
    # But Strassen gives 7 * (cost of product). Each "product" is (sum of A entries)*(sum of B entries)
    # which is rank-1 in the (A-entries x B-entries) space but produces a 2x2 C. To realize it we need
    # to multiply a scalar (combo of A) by each B entry then sum -> that's 4 scalar products per product
    # (since B combo has 4 entries). 7 products * 4 = 28. Good.
    #
    # Strassen for 2x2*2x2:
    # P1 = (A22 - A12)(B11 + B12)
    # P2 = (A11 + A21) B11
    # P3 = A12 (B21 + B22)
    # P4 = A22 (B22 - B12)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 + P4 + P5 - P7 ; C12 = P5 + P6
    # C21 = P3 + P7 ; C22 = P1 - P3 + P4 + P6
    #
    # Each Pi = (sum over A-block entries with coeff) * (sum over B-block entries with coeff)
    # We realize each Pi as: for each B-entry in the B-combo, one scalar product
    # (A-combo) * (that B-entry), contributing to all 4 C positions with the Pi's C-coeffs.

    # Define A-block entry mapping: block (a,b), entry (i,j) -> A row 2a+i, col 2b+j
    # B-block entry: block (c,d), entry (j,k) -> B row 2c+j, col 2d+k
    # C entry: block (a,d), entry (i,k) -> C row 2a+i, col 2d+k

    # Strassen coefficients: each product is (Acoeffs dict block->(i,j)->coef) * (Bcoeffs)
    # We'll define products as list of (A_terms, B_terms, C_terms)
    # A_terms: list of (block (a,b), entry (i,j), coef)
    # B_terms: list of (block (c,d), entry (j,k), coef)
    # C_terms: list of (block (a,d), entry (i,k), coef)

    def aentry(a,b,i,j): return (2*a+i, 2*b+j)
    def bentry(c,d,j,k): return (2*c+j, 2*d+k)
    def centry(a,d,i,k): return (2*a+i, 2*d+k)

    # Build the 7 Strassen products for a 2x2*2x2 with block labels
    # A blocks: A11=(0,0) A12=(0,1) A21=(1,0) A22=(1,1)
    # B blocks: B11=(0,0) B12=(0,1) B21=(1,0) B22=(1,1)
    # We'll enumerate products generically.
    products = []
    # P1 = (A22 - A12)(B11 + B12)
    products.append((
        [((1,1),0,0,1),((1,1),1,0,1),((1,1),0,1,1),((1,1),1,1,1),
         ((0,1),0,0,-1),((0,1),1,0,-1),((0,1),0,1,-1),((0,1),1,1,-1)],
        [((0,0),0,0,1),((0,0),1,0,1),((0,0),0,1,1),((0,0),1,1,1),
         ((0,1),0,0,1),((0,1),1,0,1),((0,1),0,1,1),((0,1),1,1,1)],
        [((0,0),0,0,1),((0,1),0,0,1),((1,0),0,0,1),((1,1),0,0,1),
         ((0,0),0,1,1),((0,1),0,1,1),((1,0),0,1,1),((1,1),0,1,1),
         ((0,0),1,0,1),((0,1),1,0,1),((1,0),1,0,1),((1,1),1,0,1),
         ((0,0),1,1,1),((0,1),1,1,1),((1,0),1,1,1),((1,1),1,1,1)]
    ))
    # This is getting very intricate; instead use a cleaner recursive/numeric approach:
    # Build the full 4x4*4x4 Strassen decomposition numerically by treating each product
    # as a rank-1 in (A x B) producing C via known C-coeffs. We'll just verify numerically.

    # Given complexity, fall back to a clean schoolbook for 4x4*4x4 (rank 16) + 4x4*4x3 (rank 12) = 28,
    # which is already better than 112, then let ALS try to reduce.
    # Actually let's just do full schoolbook 4x4*4x7 = 112 baseline guaranteed, then ALS reduce.
    # But the lesson says Strassen 4x4*4x4 (28) + schoolbook 4x4*4x3 (12) = 40 worked (score 1.077).
    # Let me build that 40-rank baseline properly using the product representation.

    # Reset
    U_rows = {}; V_rows = {}; W_rows = {}; col = 0

    # Strassen 4x4*4x4: 7 products, each product = (A-combo) * (B-combo) producing C via C-coeffs.
    # Represent each product by: A_terms list of (A_row, A_col, coef), B_terms list of (B_row, B_col, coef),
    # C_terms list of (C_row, C_col, coef). Realize product as: for each B_term, one scalar product
    # (sum A_terms) * (that B entry) -> but that changes rank. To keep 7 products we need the product
    # to be a SINGLE scalar. So the C must be a scalar times (A-combo)*(B-combo). But C is 2x2 per block.
    # Correct realization: each Strassen product Pi is a scalar = (A-combo)*(B-combo). Then C = sum of
    # Pi * (C-coeff matrix). But C-coeff matrix is 2x2, meaning Pi contributes to 4 C entries.
    # To realize "Pi * c" at 4 C entries with Pi being one scalar product, we need 4 factor-columns? No:
    # Pi is one scalar = sum_u A_u * sum_v B_v. We can realize Pi as rank-1: U has A-coeffs, V has B-coeffs,
    # W has... but W is indexed by C entries. Pi contributes c_ij to C_ij. We need W[C_ij] = c_ij for the same
    # rank column. So ONE rank column: U=A-coeffs, V=B-coeffs, W = C-coeffs (4 entries). 
    # That's valid! One product = one rank column where W holds the C-coefficients.
    # So 7 products = 7 rank columns for the 4x4*4x4 part. 

    # Define Strassen products with A-coeffs (over 16 A entries), B-coeffs (over 16 B entries), C-coeffs (over 16 C entries)
    # A entry (i,j) for i,j in 0..3 ; B entry (j,k) ; C entry (i,k)
    def A(i,j): return (i,j)
    def B(j,k): return (j,k)
    def C(i,k): return (i,k)

    strassen = []
    # P1 = (A22 - A12)(B11 + B12)
    A1 = {A(2,0):1,A(2,1):1,A(3,0):1,A(3,1):1,A(1,0):-1,A(1,1):-1,A(2,0):1}
    # careful: A22 = rows2,3 cols2,3 ; A12 = rows0,1 cols2,3
    A1 = {}
    for r in [2,3]:
        for c in [2,3]: A1[A(r,c)] = 1
    for r in [0,1]:
        for c in [2,3]: A1[A(r,c)] = -1
    B1 = {}
    for r in [0,1]:
        for c in [0,1]: B1[B(r,c)] = 1
    C1 = {}
    for r in [0,1]:
        for c in [0,1]: C1[C(r,c)] = 1
    for r in [2,3]:
        for c in [0,1]: C1[C(r,c)] = 1
    strassen.append((A1, B1, C1))
    # P2 = (A11 + A21) B11
    A2 = {}
    for r in [0,1]:
        for c in [0,1]: A2[A(r,c)] = 1
    for r in [2,3]:
        for c in [0,1]: A2[A(r,c)] = 1
    B2 = {}
    for r in [0,1]:
        for c in [0,1]: B2[B(r,c)] = 1
    C2 = {}
    for r in [0,1]: C2[C(r,0)] = 1
    strassen.append((A2, B2, C2))
    # P3 = A12 (B21 + B22)
    A3 = {}
    for r in [0,1]:
        for c in [2,3]: A3[A(r,c)] = 1
    B3 = {}
    for r in [2,3]:
        for c in [0,3]: B3[B(r,c)] = 1
    C3 = {}
    for r in [2,3]: C3[C(r,0)] = 1
    strassen.append((A3, B3, C3))
    # P4 = A22 (B22 - B12)
    A4 = {}
    for r in [2,3]:
        for c in [2,3]: A4[A(r,c)] = 1
    B4 = {}
    for r in [2,3]:
        for c in [2,3]: B4[B(r,c)] = 1
    for r in [0,1]:
        for c in [2,3]: B4[B(r,c)] = -1
    C4 = {}
    for r in [0,1]: C4[C(r,0)] = 1
    for r in [2,3]: C4[C(r,1)] = 1
    strassen.append((A4, B4, C4))
    # P5 = (A11 + A12) B22
    A5 = {}
    for r in [0,1]:
        for c in [0,3]: A5[A(r,c)] = 1
    B5 = {}
    for r in [2,3]:
        for c in [2,3]: B5[B(r,c)] = 1
    C5 = {}
    for r in [0,1]:
        for c in [0,1]: C5[C(r,c)] = 1
    strassen.append((A5, B5, C5))
    # P6 = (A21 - A11)(B11 + B12)
    A6 = {}
    for r in [2,3]:
        for c in [0,1]: A6[A(r,c)] = 1
    for r in [0,1]:
        for c in [0,1]: A6[A(r,c)] = -1
    B6 = {}
    for r in [0,1]:
        for c in [0,1]: B6[B(r,c)] = 1
    C6 = {}
    for r in [0,1]: C6[C(r,1)] = 1
    for r in [2,3]: C6[C(r,0)] = 1
    strassen.append((A6, B6, C6))
    # P7 = (A12 - A22)(B21 + B22)
    A7 = {}
    for r in [0,1]:
        for c in [2,3]: A7[A(r,c)] = 1
    for r in [2,3]:
        for c in [2,3]: A7[A(r,c)] = -1
    B7 = {}
    for r in [2,3]:
        for c in [0,3]: B7[B(r,c)] = 1
    C7 = {}
    for r in [0,1]: C7[C(r,0)] = -1
    for r in [2,3]:
        for c in [0,1]: C7[C(r,c)] = 1
    strassen.append((A7, B7, C7))

    for (Ac, Bc, Cc) in strassen:
        new_col_idx = col
        for (i,j), v in Ac.items():
            U_rows.setdefault(i*m+j, {})[new_col_idx] = U_rows.get(i*m+j, {}).get(new_col_idx,0) + v
        for (j,k), v in Bc.items():
            V_rows.setdefault(j*p+k, {})[new_col_idx] = V_rows.get(j*p+k, {}).get(new_col_idx,0) + v
        for (i,k), v in Cc.items():
            W_rows.setdefault(k*n+i, {})[new_col_idx] = W_rows.get(k*n+i, {}).get(new_col_idx,0) + v
        col += 1

    # ---- 4x4 * 4x3 part (columns 4,5,6 of B): schoolbook, rank 12 ----
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                new_col_idx = col
                U_rows.setdefault(i*m+j, {})[new_col_idx] = 1
                V_rows.setdefault(j*p+k, {})[new_col_idx] = 1
                W_rows.setdefault(k*n+i, {})[new_col_idx] = 1
                col += 1

    base_rank = col
    # Assemble baseline arrays
    def assemble(rows, nrows):
        arr = np.zeros((nrows, base_rank), dtype=np.float64)
        for r, d in rows.items():
            for c, v in d.items():
                arr[r, c] = v
        return arr
    U = assemble(U_rows, nm)
    V = assemble(V_rows, mp)
    W = assemble(W_rows, pn)

    def check(U, V, W, tol=1e-8):
        Ta = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - Ta)) < tol

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = base_rank
    if not check(U, V, W):
        # Fallback to full schoolbook if baseline wrong
        U = np.zeros((nm, n*m*p)); V = np.zeros((mp, n*m*p)); W = np.zeros((pn, n*m*p))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = n*m*p

    # ---- Try ALS to reduce rank, starting near baseline ----
    T_flat = T.reshape(nm, mp*pn)
    def try_rank(target, restarts=2, iters=40):
        nonlocal best_U, best_V, best_W, best_rank
        for rs in range(restarts):
            if time.time() > deadline - 8: return False
            # Initialize with baseline truncated + noise
            U = best_U[:, :min(target, best_rank)].copy()
            V = best_V[:, :min(target, best_rank)].copy()
            W = best_W[:, :min(target, best_rank)].copy()
            if target > min(target, best_rank):
                U = np.hstack([U, np.random.randn(nm, target-best_rank)*0.1])
                V = np.hstack([V, np.random.randn(mp, target-best_rank)*0.1])
                W = np.hstack([W, np.random.randn(pn, target-best_rank)*0.1])
            else:
                U = U[:, :target]; V = V[:, :target]; W = W[:, :target]
            for it in range(iters):
                if time.time() > deadline - 8: return False
                MU = np.einsum('br,cr->rbc', V, W).reshape(target, mp*pn)
                try:
                    U = T_flat @ np.linalg.pinv(MU)
                except: break
                MV = np.einsum('ar,cr->rac', U, W).reshape(target, nm*pn)
                try:
                    V = T.reshape(mp, nm*pn) @ np.linalg.pinv(MV)
                except: break
                MW = np.einsum('ar,br->rab', U, V).reshape(target, nm*mp)
                try:
                    W = T.reshape(pn, nm*mp) @ np.linalg.pinv(MW)
                except: break
                if check(U, V, W, tol=1e-8):
                    if target < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target
                    return True
        return False

    for target in range(best_rank - 1, 27, -1):
        if time.time() > deadline - 10: break
        if try_rank(target, restarts=2, iters=35):
            break

    # Snap near-integers to reduce representation error
    for arr in (best_U, best_V, best_W):
        arr[np.abs(arr) < 1e-10] = 0.0
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
