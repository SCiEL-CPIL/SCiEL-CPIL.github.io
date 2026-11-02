import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2 x 2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-vectors (row-major 2x2)."""
    # Strassen's 7 products for [[a,b],[c,d]] * [[e,f],[g,h]]
    # M1 = (a+b)(e+f)
    # M2 = (c+d)e
    # M3 = a(g+h)
    # M4 = d(f-g)
    # M5 = (a+d)(e+h)
    # M6 = (b-c)(g+h)
    # M7 = (a-c)(e+f)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M5, M1-M2+M4+M6]]
    
    # U vectors (for A), V vectors (for B), W vectors (for C)
    # A = [[a,b],[c,d]] indexed as [0,1,2,3]
    # B = [[e,f],[g,h]] indexed as [0,1,2,3]
    # C = [[c00,c01],[c10,c11]] indexed as [0,1,2,3]
    
    terms = []
    # M1 = (a+b)(e+f) -> contributes to c00 (+1), c10 (-1), c11 (+1)
    u = np.array([1, 1, 0, 0])
    v = np.array([1, 0, 1, 0])
    w = np.array([1, 0, -1, 1])
    terms.append((u, v, w))
    
    # M2 = (c+d)e -> contributes to c10 (+1), c11 (-1)
    u = np.array([0, 0, 1, 1])
    v = np.array([1, 0, 0, 0])
    w = np.array([0, 0, 1, -1])
    terms.append((u, v, w))
    
    # M3 = a(g+h) -> contributes to c01 (+1)
    u = np.array([1, 0, 0, 0])
    v = np.array([0, 0, 1, 1])
    w = np.array([0, 1, 0, 0])
    terms.append((u, v, w))
    
    # M4 = d(f-g) -> contributes to c00 (+1), c10 (-1)
    u = np.array([0, 0, 0, 1])
    v = np.array([0, 1, -1, 0])
    w = np.array([1, 0, -1, 0])
    terms.append((u, v, w))
    
    # M5 = (a+d)(e+h) -> contributes to c00 (-1), c10 (+1), c01 (+1)
    u = np.array([1, 0, 0, 1])
    v = np.array([1, 0, 0, 1])
    w = np.array([-1, 1, 1, 0])
    terms.append((u, v, w))
    
    # M6 = (b-c)(g+h) -> contributes to c11 (+1)
    u = np.array([0, 1, -1, 0])
    v = np.array([0, 0, 1, 1])
    w = np.array([0, 0, 0, 1])
    terms.append((u, v, w))
    
    # M7 = (a-c)(e+f) -> contributes to c00 (+1)
    u = np.array([1, 0, -1, 0])
    v = np.array([1, 1, 0, 0])
    w = np.array([1, 0, 0, 0])
    terms.append((u, v, w))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p, verbose=False):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    # Build the multiplication tensor from U, V, W
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if i,j,k form a valid triple (always for matrix mult)
    # Actually, T[i,j,k] should be 1 for all i,j,k since the multiplication tensor
    # has T[i,j,k] = 1 for all valid indices.
    
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-6:
                    errors += 1
                    if verbose and errors <= 10:
                        print(f"Error at ({i},{j},{k}): got {val}, expected 1.0")
    if verbose:
        print(f"Verification: {errors} errors out of {n*m*p} entries")
    return errors == 0

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    # Strategy: Block decomposition
    # 4x5 x 5x6: split m=5 into 2+3
    # A (4x5) = [A1(4x2); A2(4x3)]
    # B (5x6) = [B1(2x6) | B2(3x6)]
    # C = A1*B1 + A2*B2
    
    # For A1*B1: 4x2 x 2x6
    #   Split 4=2+2, 6=2+2+2
    #   A1 = [A11(2x2); A12(2x2)]
    #   B1 = [B11(2x2) | B12(2x2) | B13(2x2)]
    #   A1*B1 = A11*B11 + A11*B12 + A11*B13 + A12*B11 + A12*B12 + A12*B13
    #   Each 2x2x2x2 uses Strassen rank 7
    #   Total: 6*7 = 42
    
    # For A2*B2: 4x3 x 3x6
    #   Split 4=2+2, 6=2+2+2
    #   A2 = [A21(2x3); A22(2x3)]
    #   B2 = [B21(3x2) | B22(3x2) | B23(3x2)]
    #   A2*B2 = A21*B21 + A21*B22 + A21*B23 + A22*B21 + A22*B22 + A22*B23
    #   Each 2x3x3x2: need rank for this
    #   2x3 x 3x2: trivial rank = 2*3*2 = 12
    #   Can we do better? 2x3x3x2 tensor...
    #   For now use trivial: 6*12 = 72
    #   Total: 42 + 72 = 114. Not great.
    
    # Better: For 2x3 x 3x2, note it's equivalent to computing 6 scalar products
    # with sharing. Actually 2x3 * 3x2 is just 2x2 matrix mult where each "entry"
    # is a dot product of length 3. So it's like 4 dot products of length 3 = 12 mults.
    # But we can use Strassen on the 2x2 structure: 7 dot products of length 3 = 21? No.
    # Each Strassen product is a sum of products, so 7 * 3 = 21 scalar mults. Worse.
    # So trivial 12 is better for 2x3 x 3x2.
    
    # Let me try a different split.
    
    # Alternative: 4x5 x 5x6
    # Pad 5 to 8? No, that increases rank.
    
    # Let me try: split n=4 into 2+2, m=5 into 2+3, p=6 into 2+2+2
    # This gives 2*2*3 blocks for A, 2*3*3 for B... complicated.
    
    # Simpler approach: Just use the trivial decomposition but try to find
    # a slightly better one through block Strassen on sub-problems.
    
    # Let me try: 4x5 x 5x6
    # Split m=5 as 2+3
    # Part 1: 4x2 x 2x6 (rank ~42 via Strassen)
    # Part 2: 4x3 x 3x6 (rank ?)
    #   4x3 x 3x6: split 4=2+2, 6=2+2+2
    #   = 2x3x3x2 + 2x3x3x2 + 2x3x3x2 + 2x3x3x2 + 2x3x3x2 + 2x3x3x2
    #   Each 2x3x3x2: trivial rank 12, total 72
    #   Grand total: 42 + 72 = 114
    
    # Can we do 2x3x3x2 better? 
    # 2x3 * 3x2: Let A = [[a1,a2,a3],[b1,b2,b3]], B = [[x1,x2],[y1,y2],[z1,z2]]
    # C = [[a1x1+a2y1+a3z1, a1x2+a2y2+a3z2],
    #      [b1x1+b2y1+b3z1, b1x2+b2y2+b3z2]]
    # This is 4 dot products of length 3. Trivial: 12 mults.
    # Using Strassen on the 2x2 "structure": 7 "products" where each product
    # is a bilinear form. But each "product" in Strassen is a single scalar mult.
    # Here each "entry" is a dot product, so Strassen would give 7 dot products = 21 mults. Worse.
    # So 12 is optimal for 2x3x3x2? Probably.
    
    # Total rank: 42 + 72 = 114. Still better than 120!
    
    # Let me implement this.
    
    rank = 114
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    strassen_terms = _strassen_2x2()
    
    idx = 0
    
    # Part 1: 4x2 x 2x6
    # A1 is rows 0-3, cols 0-1 of A (indices 0-1 in m dimension)
    # B1 is rows 0-1, cols 0-5 of B (indices 0-1 in m dimension)
    # Split A1 into A11 (rows 0-1, cols 0-1) and A12 (rows 2-3, cols 0-1)
    # Split B1 into B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3), B13 (rows 0-1, cols 4-5)
    
    # A11: n-indices 0,1; m-indices 0,1
    # A12: n-indices 2,3; m-indices 0,1
    # B11: m-indices 0,1; p-indices 0,1
    # B12: m-indices 0,1; p-indices 2,3
    # B13: m-indices 0,1; p-indices 4,5
    
    for ai in range(2):  # A11 or A12
        n_off = ai * 2  # 0 or 2
        for bi in range(3):  # B11, B12, B13
            p_off = bi * 2  # 0, 2, 4
            for (su, sv, sw) in strassen_terms:
                # su is 4-vector for 2x2 A block, indexed [n_off+0, n_off+1] x [0, 1]
                # sv is 4-vector for 2x2 B block, indexed [0, 1] x [p_off+0, p_off+1]
                # sw is 4-vector for 2x2 C block, indexed [n_off+0, n_off+1] x [p_off+0, p_off+1]
                
                # U: n*m indices, m=5. A block is at rows n_off..n_off+1, cols 0..1
                for a_idx in range(4):
                    ni = n_off + a_idx // 2
                    mj = a_idx % 2
                    U[ni * m + mj, idx] = su[a_idx]
                
                # V: m*p indices, p=6. B block is at rows 0..1, cols p_off..p_off+1
                for b_idx in range(4):
                    mj = b_idx // 2
                    pk = b_idx % 2
                    V[mj * p + p_off + pk, idx] = sv[b_idx]
                
                # W: p*n indices, n=4. C block is at rows p_off..p_off+1, cols n_off..n_off+1
                for c_idx in range(4):
                    pk = c_idx // 2
                    ni = c_idx % 2
                    W[(p_off + pk) * n + n_off + ni, idx] = sw[c_idx]
                
                idx += 1
    
    # Part 2: 4x3 x 3x6
    # A2 is rows 0-3, cols 2-4 of A (m-indices 2,3,4)
    # B2 is rows 2-4, cols 0-5 of B (m-indices 2,3,4)
    # Split A2 into A21 (rows 0-1, cols 2-4) and A22 (rows 2-3, cols 2-4)
    # Split B2 into B21 (rows 2-4, cols 0-1), B22 (rows 2-4, cols 2-3), B23 (rows 2-4, cols 4-5)
    
    for ai in range(2):  # A21 or A22
        n_off = ai * 2  # 0 or 2
        for bi in range(3):  # B21, B22, B23
            p_off = bi * 2  # 0, 2, 4
            # 2x3 x 3x2: trivial decomposition, 12 terms
            for i_local in range(2):
                for j_local in range(3):
                    for k_local in range(2):
                        ni = n_off + i_local
                        mj = 2 + j_local  # m-indices 2,3,4
                        pk = p_off + k_local
                        
                        U[ni * m + mj, idx] = 1.0
                        V[mj * p + pk, idx] = 1.0
                        W[pk * n + ni, idx] = 1.0
                        idx += 1
    
    assert idx == rank, f"Expected {rank} terms, got {idx}"
    
    # Verify
    if time.time() < deadline - 10:
        valid = _verify_decomposition(U, V, W, n, m, p, verbose=False)
        if not valid:
            # Fallback to trivial
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, idx] = 1.0
                        V[j * p + k, idx] = 1.0
                        W[k * n + i, idx] = 1.0
                        idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
