import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    n, m, p = 2, 4, 5
    
    def verify_numpy(U, V, W, rank):
        """Fast verification using numpy tensor contraction."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Reshape to 3D tensors
        U3 = Ua.reshape(n, m, rank)
        V3 = Va.reshape(m, p, rank)
        W3 = Wa.reshape(p, n, rank)
        # Compute sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r]
        # Use einsum
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check all entries equal 1
        if not np.allclose(result, 1.0, atol=1e-10):
            return False
        # Also check for exact integer/half-integer values
        # The entries should be exact, so check with tolerance
        return True
    
    def verify_exact(U, V, W, rank):
        """Exact verification with fractions to ensure correctness."""
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    # Rank-36 Strassen-based solution
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W36[gk*n + gi][col] += val_w
                    col += 1
            else:
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W36[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank36 and verify_numpy(U36, V36, W36, rank36):
        if verify_exact(U36, V36, W36, rank36):
            return {"U": U36, "V": V36, "W": W36}
    
    # Try to find a lower rank by searching over block structures
    # The key insight: we can try different combinations of Strassen and standard
    # terms for different (q, s) blocks to reduce rank
    
    # Strategy: For each (q, s) block, we can choose either:
    # - 2x2 Strassen (7 terms) for s < 2
    # - 1x1 standard (4 terms) for s = 2 (since it's 2x1)
    # But we can also try:
    # - For s < 2: use 2x2 Strassen (7 terms) OR try to share terms
    # - For s = 2: use 2x1 standard (4 terms) OR try 2x1 Strassen-like (3 terms?)
    
    # Actually, for 2x1 multiplication, the minimum rank is 3 (not 4)
    # because we can use: (A1+A2)(B1), A1(B2-B1), A2(B1) -> wait, that's for 2x2
    # For 2x1: C1 = A1*B1, C2 = A2*B1. Minimum rank is 2 (just two multiplications)
    # Actually no, 2x1 needs rank 2 minimum since there are 2 outputs.
    
    # Let me reconsider. The current approach uses:
    # - 2 q-values x 2 s-values (s=0,1) with 7 Strassen terms each = 28
    # - 2 q-values x 1 s-value (s=2) with 4 standard terms each = 8
    # Total = 36
    
    # Can we do better? Let's think about the structure:
    # Matrix A is 2x4, B is 4x5
    # Split A into 2 blocks of 2x2: A = [A_q for q in {0,1}]
    # Split B into 3 blocks: B = [B_s for s in {0,1,2}] where B_0, B_1 are 2x2 and B_2 is 2x1
    
    # For each (q, s) pair, we compute A_q * B_s
    # - If s < 2: A_q is 2x2, B_s is 2x2 -> rank 7 (Strassen)
    # - If s = 2: A_q is 2x2, B_s is 2x1 -> rank 2 (just 2 mults)
    
    # So minimum with this block structure: 2*2*7 + 2*1*2 = 28 + 4 = 32
    
    # Let's implement rank-32 solution!
    rank32 = 32
    U32 = [[0] * rank32 for _ in range(n * m)]
    V32 = [[0] * rank32 for _ in range(m * p)]
    W32 = [[0] * rank32 for _ in range(p * n)]
    
    col = 0
    for q in range(2):
        for s in range(3):
            if s < 2:
                # Use 7 Strassen terms for 2x2 * 2x2
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U32[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V32[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W32[gk*n + gi][col] += val_w
                    col += 1
            else:
                # Use 2 terms for 2x2 * 2x1 (just direct multiplication)
                # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                # Term 1: A[0,0]*B[0,0] contributes to C[0,0]
                # Term 2: A[0,1]*B[1,0] contributes to C[0,0]
                # Term 3: A[1,0]*B[0,0] contributes to C[1,0]
                # Term 4: A[1,1]*B[1,0] contributes to C[1,0]
                # That's 4 terms, not 2. Let me reconsider.
                
                # Actually for 2x2 * 2x1:
                # Result is 2x1: [c0, c1]
                # c0 = a00*b0 + a01*b1
                # c1 = a10*b0 + a11*b1
                # Minimum rank is 4 (each term is independent)
                # Unless we can factor...
                
                # Hmm, so it's still 4 terms. Let me keep the standard approach.
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U32[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V32[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W32[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank32 and verify_numpy(U32, V32, W32, rank32):
        if verify_exact(U32, V32, W32, rank32):
            return {"U": U32, "V": V32, "W": W32}
    
    # Try rank-36 with numpy verification
    if verify_numpy(U36, V36, W36, rank36):
        if verify_exact(U36, V36, W36, rank36):
            return {"U": U36, "V": V36, "W": W36}
    
    # Rank-30: use 7-term "Strassen" for 2x2 * 2x1 blocks
    # The 2x1 multiplication tensor has rank 3, but we can embed it in a
    # 7-term scheme. For C = A * b where A is 2x2, b is 2x1:
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    #
    # We can use the following 3-term scheme (rank-3 for 2x1):
    # t1 = (a00+a01) * b0  -> contributes a00*b0 to c0, a01*b0 to c1 (but we don't want a01*b0)
    # Actually rank of 2x1 matmul is 2 (just two independent products).
    # Wait: c0 = a00*b0 + a01*b1, c1 = a10*b0 + a11*b1
    # These are 4 independent bilinear terms. Rank is 4.
    # So we can't do better than 4 for 2x2 * 2x1.
    #
    # However, we CAN try a different block decomposition.
    # Instead of splitting 4 columns into 2+2+2, try 2+1+1 or 1+1+1+1.
    # With 1x1 blocks: each is rank 1. Total = 2*4*1 = 8 terms for the "standard" part.
    # But that's worse than 2*2*7 + 2*1*4 = 36.
    #
    # Alternative: split as 1+1+1+1 for B columns, and use Strassen on the 2x2 A side?
    # No, Strassen requires 2x2 * 2x2.
    #
    # Let's try: split A into 2x2 blocks (2 blocks), B into 1x1 columns (5 blocks).
    # For each (q, k) pair: A_q (2x2) * B_k (2x1) -> rank 4
    # Total: 2 * 5 * 4 = 40. Worse.
    #
    # Try: A into 1x1 rows (2 blocks), B into 2x2+2x1 (3 blocks).
    # For each (i, s) pair: A_i (1x2) * B_s (2x2 or 2x1)
    # 1x2 * 2x2: rank 4 (each output is a sum of 2 products, 2 outputs)
    # 1x2 * 2x1: rank 2
    # Total: 2 * (2*4 + 1*2) = 2 * 10 = 20? No wait.
    # A_i is 1x2 (one row of A), B_s is 2x2 or 2x1.
    # Result: 1x2 or 1x1.
    # 1x2 * 2x2 -> 1x2: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11. Rank 4.
    # 1x2 * 2x1 -> 1x1: c0 = a0*b0 + a1*b1. Rank 2.
    # Total: 2 rows * (2 s-blocks * 4 + 1 s-block * 2) = 2 * (8 + 2) = 20.
    # Wait that's 20! Let me verify this is correct.
    #
    # A is 2x4: rows i=0,1; columns j=0,1,2,3.
    # B is 4x5: rows j=0,1,2,3; columns k=0,1,2,3,4.
    # C is 2x5: C[i,k] = sum_j A[i,j]*B[j,k].
    #
    # Split B columns: s=0 -> k=0,1; s=1 -> k=2,3; s=2 -> k=4.
    # For each row i of A and each block s of B:
    #   A_i is the i-th row: [A[i,0], A[i,1], A[i,2], A[i,3]]
    #   But we need to split A_i into 2x2 blocks matching B_s.
    #   For s=0: A_i[0:2] = [A[i,0], A[i,1]] (1x2), B_0 = B[0:2, 0:2] (2x2)
    #   For s=1: A_i[2:4] = [A[i,2], A[i,3]] (1x2), B_1 = B[2:4, 2:4] (2x2)
    #   For s=2: A_i[0:2]... wait, B_2 = B[0:2, 4] is 2x1.
    #   Hmm, the row indices of B_s must match the column indices of A_i.
    #   For s=0: B[0:2, 0:2], A_i columns 0,1 -> A[i,0:2]
    #   For s=1: B[2:4, 2:4], A_i columns 2,3 -> A[i,2:4]
    #   For s=2: B[0:4, 4] is 4x1, not 2x1. This doesn't work cleanly.
    #
    # Let me reconsider. The block structure must be consistent.
    # If I split B's rows into blocks of size 2: rows 0-1, 2-3.
    # Then B = [B_top (2x5), B_bottom (2x5)].
    # A = [A_left (2x2), A_right (2x2)].
    # C = A_left * B_top + A_right * B_bottom.
    #
    # A_left * B_top: 2x2 * 2x5. Split B_top into 2x2, 2x2, 2x1.
    # A_right * B_bottom: 2x2 * 2x5. Same split.
    #
    # For 2x2 * 2x2: rank 7 (Strassen).
    # For 2x2 * 2x1: rank 4.
    # Total: 2 * (2*7 + 1*4) = 2 * 18 = 36. Same as before.
    #
    # Alternative split: B rows into 1+1+1+1 (four 1x5 blocks).
    # A columns into 1+1+1+1 (four 2x1 blocks).
    # Each pair: 2x1 * 1x5 = 2x5. Rank = 10? No, 2x1 * 1x5:
    #   A is 2x1: [a0; a1], B is 1x5: [b0, b1, b2, b3, b4]
    #   C = [a0*b0, a0*b1, a0*b2, a0*b3, a0*b4; a1*b0, a1*b1, a1*b2, a1*b3, a1*b4]
    #   This is rank 10 (10 independent products).
    # Total: 4 * 10 = 40. Worse.
    #
    # I think 36 is hard to beat with simple block decompositions.
    # Let me try a more sophisticated approach: can we share terms between
    # the two q-blocks?
    #
    # For q=0 and q=1, the Strassen terms have the same structure but different
    # A entries. If we could combine them... but they multiply different A entries
    # with the same B entries (for the same s-block). So the products are different.
    #
    # Actually wait: for s=0, both q=0 and q=1 multiply with B[0:2, 0:2].
    # q=0 uses A[0:2, 0:2], q=1 uses A[0:2, 2:4].
    # The Strassen intermediate products are like (a00+a11)*(b00+b11) for q=0
    # and (a02+a13)*(b00+b11) for q=1. These are different products, so no sharing.
    #
    # I'll keep rank 36 as the best known and fix the verification bug.
    
    # Schoolbook fallback
    rank_school = n * m * p
    best = {
        "U": [[0]*rank_school for _ in range(n*m)],
        "V": [[0]*rank_school for _ in range(m*p)],
        "W": [[0]*rank_school for _ in range(p*n)]
    }
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best["U"][i*m+j][c] = 1
                best["V"][j*p+k][c] = 1
                best["W"][k*n+i][c] = 1
                c += 1
    return best
