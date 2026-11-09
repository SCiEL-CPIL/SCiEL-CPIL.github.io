"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 16, 20, 20
    
    # Strategy: Decompose 4x4x5 into:
    #   Block 1: 2x4x2 (rows 0-1 of C, cols 0-1) -> rank 14
    #   Block 2: 2x4x2 (rows 2-3 of C, cols 0-1) -> rank 14
    #   Block 3: 2x4x2 (rows 0-1 of C, cols 2-3) -> rank 14
    #   Block 4: 2x4x2 (rows 2-3 of C, cols 2-3) -> rank 14
    #   Block 5: 4x4x1 (all rows, col 4) -> rank 16
    # Total rank: 14*4 + 16 = 72... that's worse.
    # 
    # Better: Use the rectangular structure directly.
    # 4x4x5 = (2x4x5) + (2x4x5) where first covers rows 0-1, second covers rows 2-3.
    # Each 2x4x5 can be decomposed as 2x4x2 + 2x4x2 + 2x4x1
    # 2x4x2: rank 14 (using a custom algorithm)
    # 2x4x1: rank 8
    # So each 2x4x5 = 14+14+8 = 36
    # Total = 36+36 = 72. Still worse.
    #
    # Let me try: 4x4x5 = 4x4x4 + 4x4x1 = 49 + 16 = 65 (current)
    # Or: 4x4x5 = (4x4x2) + (4x4x2) + (4x4x1)
    # 4x4x2: can use 2x4x2 + 2x4x2 = 14+14 = 28? Or use a direct algorithm.
    # Actually 4x4x2 = 4x(4x2). The 4x2 part: each row of C (4 entries) times 2 cols.
    # 4x4x2: A(4x4)*B(4x2) = C(4x2). Rank of this tensor?
    # Schoolbook: 4*4*2 = 32.
    # We can do: C[:,0] = A*B[:,0] (rank 16) and C[:,1] = A*B[:,1] (rank 16), total 32.
    # Or: C = A * B where B is 4x2. We can write B = [b0, b1] and
    # C[:,0] = A*b0, C[:,1] = A*b1.
    # Using Strassen-like: Let b+ = b0+b1, b- = b0-b1.
    # C[:,+] = A*b+ (rank 16), C[:,-] = A*b- (rank 16)
    # C[:,0] = (C[:,+] + C[:,-])/2, C[:,1] = (C[:,+] - C[:,-])/2
    # Total rank: 32. Same as schoolbook.
    #
    # So 4x4x5 via 4x4x4+4x4x1 = 49+16 = 65 is good.
    # Can we do better on the 4x4x1 part? 4x4x1 is just matrix-vector multiply, rank = 16 (schoolbook is optimal for this since each output element needs all 4 inputs).
    #
    # What about a different decomposition? 
    # 4x4x5: We need C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..3, k in 0..4.
    # 
    # Alternative: Use the fact that 5 = 4+1. The current approach is reasonable.
    # Let's try to optimize the 4x4x4 part.
    # Current: recursive Strassen gives 49.
    # The best known rank for 4x4x4 is 49 (from recursive Strassen).
    # So 49+16=65 might be near-optimal for this decomposition.
    #
    # Let me try a different split: 4x4x5 = (2x4x5) + (2x4x5)
    # 2x4x5: A(2x4)*B(4x5) = C(2x5). 
    # Schoolbook: 2*4*5=40.
    # Can we do better? 
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    # 
    # This is like computing two rows of C. We can use:
    # For each k, C[:,k] = A * B[:,k] (2x4 times 4x1 = 2x1).
    # 2x4x1: rank 8 (schoolbook).
    # So 2x4x5 = 5 * 8 = 40 (schoolbook).
    # Or: C = A*B. We can group columns.
    # Let's use Strassen-like on the 2x4 matrix A.
    # A = [A0 | A1] where A0, A1 are 2x2.
    # B = [B0; B1] where B0, B1 are 2x5.
    # C = A0*B0 + A1*B1 (2x5).
    # A0*B0: 2x2 * 2x5 = 2x5. Rank: 2*2*5=20 schoolbook, or use Strassen on the 2x2 part?
    # Actually 2x2x5: A(2x2)*B(2x5)=C(2x5). 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # For each k: C[:,k] = A * B[:,k]. 2x2x1 rank 4.
    # So 2x2x5 = 5*4 = 20.
    # Or: use the 2x2 Strassen: 7 products of 1x1x5? No, that doesn't work directly.
    # 
    # Let me try: 2x2x5. A(2x2), B(2x5), C(2x5).
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is 2 outputs per k, 5 k's = 10 outputs.
    # Each output C[i,k] = sum of 2 products.
    # We can write: for each k, [C[0,k], C[1,k]]^T = A * [B[0,k], B[1,k]]^T
    # This is a 2x2 * 2x1 multiply, rank 4 (schoolbook: 4 scalar mults).
    # Total: 5*4 = 20.
    # Can we share across k? 
    # Let B+ = B[0,:]+B[1,:] (1x5), B- = B[0,:]-B[1,:] (1x5)
    # A+ = A[0,:]+A[1,:] (1x2), A- = A[0,:]-A[1,:] (1x2)
    # Hmm, this gets complicated. Let me just use the current 65-rank solution.
    #
    # Actually, let me reconsider. The directive says to try the 2x4x2 rank-14 approach.
    # 4x4x5 = 4x(4x5). 
    # 4x5: For each row i of C, C[i,:] = A[i,:]*B (1x4 times 4x5 = 1x5).
    # 1x4x5: rank 20 (schoolbook).
    # 4 rows: 4*20 = 80. Worse.
    #
    # Let me just implement the current 65-rank solution cleanly and verify it works.
    # The current code has a bug in the recursive Strassen part. Let me fix it.
    
    # Schoolbook fallback (rank 80)
    def schoolbook():
        U2 = np.zeros((nm, 80))
        V2 = np.zeros((mp, 80))
        W2 = np.zeros((pn, 80))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U2[i*m+j, r] = 1.0
                    V2[j*p+k, r] = 1.0
                    W2[k*n+i, r] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Use the known-good 4x4x4 Strassen (rank 49) + 4x4x1 schoolbook (rank 16) = 65
    # Let me build this directly using a verified approach.
    
    # 4x4x4 via recursive Strassen: 7 outer products, each 2x2x2 via Strassen (7 inner) = 49
    # 4x4x1 via schoolbook: 16
    
    total_rank = 65
    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)
    
    # Part 2: Schoolbook for column 4 (k=4): rank 16
    for i in range(n):
        for j in range(m):
            col = 49 + i * m + j
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
    
    # Part 1: 4x4x4 via recursive Strassen: rank 49
    # A(4x4) * B(4x4) = C(4x4), where B uses columns 0-3
    
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Outer Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mk is a 2x2 * 2x2 = 2x2 multiply, done via inner Strassen (7 scalar products)
    
    # Block index mapping: block index bi in {0,1,2,3}
    # bi=0: top-left (i in 0-1, j in 0-1)
    # bi=1: top-right (i in 0-1, j in 2-3)
    # bi=2: bottom-left (i in 2-3, j in 0-1)
    # bi=3: bottom-right (i in 2-3, j in 2-3)
    
    def a_global(bi, u, v):
        """Global A index for block bi, local position (u,v) where u,v in {0,1}"""
        i = (bi // 2) * 2 + u
        j = (bi % 2) * 2 + v
        return i * m + j
    
    def b_global(bi, u, v):
        """Global B index for block bi, local position (u,v) where u,v in {0,1}"""
        j = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return j * p + k
    
    def c_global(bi, u, v):
        """Global C index for block bi, local position (u,v) where u,v in {0,1}"""
        i = (bi // 2) * 2 + u
        k = (bi % 2) * 2 + v
        return k * n + i
    
    # Outer Strassen coefficients
    outer_A = [
        [(0, 1.0), (3, 1.0)],   # M1
        [(2, 1.0), (3, 1.0)],   # M2
        [(0, 1.0)],             # M3
        [(3, 1.0)],             # M4
        [(0, 1.0), (1, 1.0)],   # M5
        [(2, 1.0), (0, -1.0)],  # M6
        [(1, 1.0), (3, -1.0)],  # M7
    ]
    outer_B = [
        [(0, 1.0), (3, 1.0)],   # M1
        [(0, 1.0)],             # M2
        [(1, 1.0), (3, -1.0)],  # M3
        [(2, 1.0), (0, -1.0)],  # M4
        [(3, 1.0)],             # M5
        [(0, 1.0), (1, 1.0)],   # M6
        [(2, 1.0), (3, 1.0)],   # M7
    ]
    outer_C = [
        {0: 1.0, 3: 1.0},       # M1 -> C00(+), C11(+)
        {2: 1.0, 3: -1.0},      # M2 -> C10(+), C11(-)
        {1: 1.0, 3: 1.0},       # M3 -> C01(+), C11(+)
        {0: 1.0, 2: 1.0},       # M4 -> C00(+), C10(+)
        {0: -1.0, 1: 1.0},      # M5 -> C00(-), C01(+)
        {3: 1.0},               # M6 -> C11(+)
        {0: 1.0},               # M7 -> C00(+)
    ]
    
    # Inner Strassen (2x2x2) coefficients
    inner_A = [
        [(0, 1.0), (3, 1.0)],   # N1
        [(2, 1.0), (3, 1.0)],   # N2
        [(0, 1.0)],             # N3
        [(3, 1.0)],             # N4
        [(0, 1.0), (1, 1.0)],   # N5
        [(2, 1.0), (0, -1.0)],  # N6
        [(1, 1.0), (3, -1.0)],  # N7
    ]
    inner_B = [
        [(0, 1.0), (3, 1.0)],   # N1
        [(0, 1.0)],             # N2
        [(1, 1.0), (3, -1.0)],  # N3
        [(2, 1.0), (0, -1.0)],  # N4
        [(3, 1.0)],             # N5
        [(0, 1.0), (1, 1.0)],   # N6
        [(2, 1.0), (3, 1.0)],   # N7
    ]
    inner_C = [
        {0: 1.0, 3: 1.0},       # N1 -> C00(+), C11(+)
        {2: 1.0, 3: -1.0},      # N2 -> C10(+), C11(-)
        {1: 1.0, 3: 1.0},       # N3 -> C01(+), C11(+)
        {0: 1.0, 2: 1.0},       # N4 -> C00(+), C10(+)
        {0: -1.0, 1: 1.0},      # N5 -> C00(-), C01(+)
        {3: 1.0},               # N6 -> C11(+)
        {0: 1.0},               # N7 -> C00(+)
    ]
    
    for mk in range(7):
        for nk in range(7):
            col = mk * 7 + nk
            
            # U contributions: A''_nk = sum over outer A blocks and inner A blocks
            for (bi_outer, c1) in outer_A[mk]:
                for (bi_inner, c2) in inner_A[nk]:
                    u_loc = bi_inner // 2
                    v_loc = bi_inner % 2
                    U[a_global(bi_outer, u_loc, v_loc), col] += c1 * c2
            
            # V contributions
            for (bi_outer, c1) in outer_B[mk]:
                for (bi_inner, c2) in inner_B[nk]:
                    u_loc = bi_inner // 2
                    v_loc = bi_inner % 2
                    V[b_global(bi_outer, u_loc, v_loc), col] += c1 * c2
            
            # W contributions
            for (bi_inner, c_inner) in inner_C[nk].items():
                for (bi_outer, c_outer) in outer_C[mk].items():
                    u_loc = bi_inner // 2
                    v_loc = bi_inner % 2
                    W[c_global(bi_outer, u_loc, v_loc), col] += c_inner * c_outer
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(total_rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
