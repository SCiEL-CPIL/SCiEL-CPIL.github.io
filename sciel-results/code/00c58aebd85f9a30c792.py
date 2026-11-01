"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 7-term decomposition for 2x2x2
    # U indices: (i,j) -> 0..3 (i*m+j)
    # V indices: (j,k) -> 0..3 (j*p+k)
    # W indices: (k,i) -> 0..3 (k*n+i)
    
    # Each term is (u_vector, v_vector, w_vector)
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11]
    # w: [C00, C10, C01, C11] (since W[k*n+i] with n=2: index k*2+i)
    # Wait, W index is k*n+i. n=2.
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1]),
        # M3 = A00(B10+B11)
        ([1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]),
        # M4 = A11(B01-B10)
        ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 0, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, -1, 0, 0]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 0, -1, 1]),
    ]
    
    # Schoolbook 4 terms for 2x2x2
    schoolbook_terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]), # A00*B00 -> C00
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]), # A01*B11 -> C01
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]), # A10*B00 -> C10
        ([0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0]), # Wait, A11*B11 -> C11? No, A11*B11 is not standard schoolbook.
    ]
    
    # Let's re-verify schoolbook for 2x2x2:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    schoolbook_terms = [
        ([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]), # A00*B00 -> C00
        ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]), # A00*B01 -> C01
        ([0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]), # A01*B10 -> C00
        ([0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]), # A01*B11 -> C01
        ([0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]), # A10*B00 -> C10
        ([0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]), # A10*B01 -> C11
        ([0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]), # A11*B10 -> C10
        ([0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]), # A11*B11 -> C11
    ]
    # Wait, schoolbook is 8 terms? No, 2*2*2 = 8.
    # But Strassen is 7.
    
    # Strategy:
    # Use Strassen for the first 12 columns (6 blocks of 2).
    # Use Schoolbook for the last 1 column.
    # Total rank = 6 * 7 + 4? No, schoolbook for 2x2x1 is 4 terms.
    # Wait, if p=13, we have 13 columns in B.
    # B is 2x13.
    # We can group columns of B into blocks of 2.
    # 13 = 6*2 + 1.
    # So 6 blocks of 2 columns, and 1 block of 1 column.
    
    # For a block of 2 columns (j=0,1) and k in {k0, k0+1}:
    # We use Strassen 2x2x2.
    # U is the same for all blocks? No, U depends on A, which is 2x2.
    # So U is the same for all Strassen blocks!
    # V depends on B block.
    # W depends on C block.
    
    # Let's build U, V, W.
    rank = 0
    U_rows = n * m # 4
    V_rows = m * p # 26
    W_rows = p * n # 26
    
    U = [[0.0] * 0 for _ in range(U_rows)] # Placeholder
    V = [[0.0] * 0 for _ in range(V_rows)]
    W = [[0.0] * 0 for _ in range(W_rows)]
    
    # We don't know rank yet, so build lists of terms.
    terms = []
    
    # 6 Strassen blocks
    for block_idx in range(6):
        k_start = block_idx * 2
        for (u, v, w) in strassen_terms:
            # u is for A (same for all blocks)
            # v is for B block (columns k_start, k_start+1)
            # w is for C block (rows k_start, k_start+1)
            
            # Expand u, v, w to full length
            u_full = [0.0] * U_rows
            for idx, val in enumerate(u):
                u_full[idx] = val
                
            v_full = [0.0] * V_rows
            # v indices: [B00, B01, B10, B11] relative to block
            # Global indices:
            # B[j][k] -> index j*p + k
            # j=0: k_start, k_start+1 -> indices k_start, k_start+1
            # j=1: k_start, k_start+1 -> indices p+k_start, p+k_start+1
            v_full[k_start] = v[0]       # B00
            v_full[k_start+1] = v[1]     # B01
            v_full[p + k_start] = v[2]   # B10
            v_full[p + k_start+1] = v[3] # B11
            
            w_full = [0.0] * W_rows
            # w indices: [C00, C10, C01, C11] relative to block
            # Global indices:
            # C[k][i] -> index k*n + i
            # k=k_start: i=0,1 -> indices k_start*n, k_start*n+1
            # k=k_start+1: i=0,1 -> indices (k_start+1)*n, (k_start+1)*n+1
            
            # w[0] is C00 (k=0, i=0) -> global (k_start, 0)
            w_full[k_start * n + 0] = w[0]
            # w[1] is C10 (k=0, i=1) -> global (k_start, 1)
            w_full[k_start * n + 1] = w[1]
            # w[2] is C01 (k=1, i=0) -> global (k_start+1, 0)
            w_full[(k_start+1) * n + 0] = w[2]
            # w[3] is C11 (k=1, i=1) -> global (k_start+1, 1)
            w_full[(k_start+1) * n + 1] = w[3]
            
            terms.append((u_full, v_full, w_full))
            
    # 1 Schoolbook block (last column, k=12)
    k_start = 12
    for (u, v, w) in schoolbook_terms:
        u_full = [0.0] * U_rows
        for idx, val in enumerate(u):
            u_full[idx] = val
            
        v_full = [0.0] * V_rows
        # v indices: [B00, B01, B10, B11] but only k=12 exists?
        # No, schoolbook terms are for specific (i,j,k).
        # My schoolbook_terms above were for 2x2x2.
        # I need schoolbook for 2x2x1.
        # Terms:
        # A00*B00 -> C00 (k=12, i=0)
        # A01*B10 -> C00 (k=12, i=0)
        # A00*B01 -> C01 (k=12, i=1)
        # A01*B11 -> C01 (k=12, i=1)
        # A10*B00 -> C10 (k=12, i=0)
        # A11*B10 -> C10 (k=12, i=0)
        # A10*B01 -> C11 (k=12, i=1)
        # A11*B11 -> C11 (k=12, i=1)
        
        # Let's just use the 4 terms from the 2x2x2 schoolbook that involve only k=0?
        # No, let's define 2x2x1 schoolbook explicitly.
        pass
        
    # Let's redefine the last part properly.
    terms = []
    
    # 6 Strassen blocks
    for block_idx in range(6):
        k_start = block_idx * 2
        for (u, v, w) in strassen_terms:
            u_full = [0.0] * U_rows
            for idx, val in enumerate(u):
                u_full[idx] = val
                
            v_full = [0.0] * V_rows
            v_full[k_start] = v[0]
            v_full[k_start+1] = v[1]
            v_full[p + k_start] = v[2]
            v_full[p + k_start+1] = v[3]
            
            w_full = [0.0] * W_rows
            w_full[k_start * n + 0] = w[0]
            w_full[k_start * n + 1] = w[1]
            w_full[(k_start+1) * n + 0] = w[2]
            w_full[(k_start+1) * n + 1] = w[3]
            
            terms.append((u_full, v_full, w_full))
            
    # 1 Schoolbook block (k=12)
    k_start = 12
    # Terms for 2x2x1:
    # A00 B00 C00
    # A01 B10 C00
    # A00 B01 C01
    # A01 B11 C01
    # A10 B00 C10
    # A11 B10 C10
    # A10 B01 C11
    # A11 B11 C11
    
    # Wait, C index is k*n+i. Here k=12.
    # C00 means i=0, k=12 -> index 12*2+0 = 24
    # C01 means i=1, k=12 -> index 12*2+1 = 25
    # C10 means i=0, k=12? No, C[i][j] is not the convention.
    # Convention: W[k*n+i].
    # C is p x n.
    # C[k][i].
    # So C[12][0] and C[12][1].
    
    # Let's check the indices again.
    # C = A * B.
    # A is 2x2. B is 2x13. C is 2x13.
    # C[i][k] = sum_j A[i][j] B[j][k].
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W maps to C^T? Or C is stored as k*n+i?
    # If C is 2x13 (i=0,1; k=0..12), then C[i][k] is standard.
    # But W index is k*n+i.
    # If n=2, index is 2*k + i.
    # This corresponds to C^T[k][i] = C[i][k].
    # So W[k*n+i] corresponds to C[i][k].
    
    # Let's verify with Strassen 2x2x2.
    # C[i][k].
    # C[0][0] = A[0][0]B[0][0] + A[0][1]B[1][0]
    # C[0][1] = A[0][0]B[0][1] + A[0][1]B[1][1]
    # C[1][0] = A[1][0]B[0][0] + A[1][1]B[1][0]
    # C[1][1] = A[1][0]B[0][1] + A[1][1]B[1][1]
    
    # W indices (k*n+i):
    # k=0, i=0 -> 0 (C[0][0])
    # k=0, i=1 -> 1 (C[1][0])
    # k=1, i=0 -> 2 (C[0][1])
    # k=1, i=1 -> 3 (C[1][1])
    
    # So W[0] = C[0][0]
    # W[1] = C[1][0]
    # W[2] = C[0][1]
    # W[3] = C[1][1]
    
    # My Strassen W vectors were:
    # M1: [1, 0, 0, 1] -> W[0]=1, W[3]=1.
    # C[0][0] += 1, C[1][1] += 1.
    # M1 = (A00+A11)(B00+B11).
    # C[0][0] gets A00B00. C[1][1] gets A11B11.
    # This is correct.
    
    # M2: [0, 0, 0, 1] -> W[3]=1. C[1][1] += 1.
    # M2 = (A10+A11)B00.
    # C[1][0] gets A10B00. C[1][1] gets A11B00? No.
    # C[1][1] = A10B01 + A11B11.
    # M2 contributes to C[1][1]?
    # (A10+A11)B00 = A10B00 + A11B00.
    # A10B00 is in C[1][0]. A11B00 is not in C.
    # Something is wrong with my Strassen terms or W mapping.
    
    # Let's re-derive Strassen for C[i][k].
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10+B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C00:
    # M1 - M4 + M5 + M7
    # (A00+A11)(B00+B11) - A11(B01-B10) + (A00+A01)B11 + (A01-A11)(B10+B11)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B10 + A00B11 + A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B01 + A11B10 + A00B11 + A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    # Group A00: B00 + B11 + B11 = B00 + 2B11? No.
    # M1: A00B00, A00B11
    # M5: A00B11
    # Sum A00: B00 + 2B11.
    # But C00 = A00B00 + A01B10.
    # My Strassen formula might be for different indices.
    
    # Standard Strassen:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C01 = M3 + M5
    # A00(B10+B11) + (A00+A01)B11
    # = A00B10 + A00B11 + A00B11 + A01B11
    # = A00B10 + 2A00B11 + A01B11
    # But C01 = A00B01 + A01B11.
    # Mismatch.
    
    # Ah, the standard Strassen definition usually has:
    # M3 = A00(B10+B11)
    # M5 = (A00+A01)B11
    # This is for C[i][j] where A is i x k, B is k x j.
    # Here A is 2x2, B is 2x2.
    # C[0][1] = A[0][0]B[0][1] + A[0][1]B[1][1].
    # My M3 uses B[1][0], B[1][1].
    # So M3 contributes to C[0][0]?
    # A00B10 is C[0][0] term? No, C[0][0] = A00B00 + A01B10.
    # So A00B10 is not in C[0][0].
    
    # I think I mixed up the B indices in Strassen.
    # Let's look up the correct Strassen terms for C = AB.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21+B22)
    # M4 = A22(B12-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's check C12 = M3 + M5
    # A11(B21+B22) + (A11+A12)B22
    # = A11B21 + A11B22 + A11B22 + A12B22
    # = A11B21 + 2A11B22 + A12B22
    # But C12 = A11B12 + A12B22.
    # Still mismatch.
    
    # Okay, I will just use the 46 rank solution found in literature or derive it carefully.
    # Actually, the rank of 2x2x13 is known to be 46?
    # R(2,2,13) = 46?
    # Let's just output the schoolbook rank 52 to be safe, or try a simple block diagonal.
    # If I use Strassen for 2x2x2, rank is 7.
    # For 2x2x13, I can use 6 blocks of 2x2x2 and 1 block of 2x2x1.
    # Rank = 6*7 + 4 = 46.
    
    # Let's fix the Strassen terms to match C[i][k].
    
    # C[i][k] = sum_j A[i][j] B[j][k]
    
    # Let's use the following terms (verified):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10+B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Wait, I already checked this and it failed.
    # Let's check C00 again.
    # M1 = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11B01 - A11B10
    # M5 = A00B11 + A01B11
    # M7 = A01B10 + A01B11 - A11B10 - A11B11
    
    # M1 - M4 + M5 + M7
    # = A00B00 + A00B11 + A11B00 + A11B11
    #   - A11B01 + A11B10
    #   + A00B11 + A01B11
    #   + A01B10 + A01B11 - A11B10 - A11B11
    
    # A00: B00 + B11 + B11 = B00 + 2B11
    # A01: B11 + B10 + B11 = B10 + 2B11
    # A11: B00 + B11 - B01 + B10 - B10 - B11 = B00 - B01
    
    # This is not C00.
    
    # I will abandon Strassen for now and just return the schoolbook decomposition.
    # It is valid and safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
