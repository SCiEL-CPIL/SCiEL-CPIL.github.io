"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Check that sum_c U[:,c] * V[:,c] * W[:,c] equals the matmul tensor
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The matmul tensor is T[i,j,k] = 1 if this is a "multiplication" tensor, i.e.,
    # T[i,j,k] = delta_{i,j,k} in the sense that it represents A*B=C
    # Actually the tensor is T[i,j,k] = 1 for the standard matmul tensor where
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the standard basis, meaning the tensor
    # is the identity in the appropriate sense.
    # More precisely: the tensor T has T[i,j,k] = 1 if and only if the term
    # A[i,j]*B[j,k] appears in C[i,k], which is always true.
    # So T[i,j,k] = 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank12_3x3x2():
    """
    Construct a rank-12 decomposition for 3x3x2 matrix multiplication.
    A (3x3) * B (3x2) -> C (3x2)
    
    Convention:
    U index: i*m + j = 3i + j  (i=0..2, j=0..2)
    V index: j*p + k = 2j + k  (j=0..2, k=0..1)
    W index: k*n + i = 2i + k  (k=0..1, i=0..2)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Strategy: 
    - Use Strassen (rank 7) for the 2x2x2 part (i,j in {0,1}, k in {0,1})
    - Handle remaining terms for i=2 and j=2
    
    Actually, let me use a different approach. Let me directly construct 
    a valid decomposition.
    
    The 3x3x2 matmul tensor has 9*6*6 = 324 entries, all equal to 1
    (in the sense that T[i,j,k]=1 for the standard matmul).
    
    Let me try a systematic construction:
    
    For each (i,k) pair (6 outputs), C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    
    I'll use the following 12 terms:
    Term 1: (A[0,0]+A[0,1]+A[1,0]+A[1,1]) * (B[0,0]+B[1,0]) -> contributes to C[0,0], C[1,0]
    ... this is getting complex. Let me just use schoolbook for the 3x3x2 block
    which is rank 18, giving total rank 54. Same as before.
    
    Actually, let me try a cleaner rank-12 construction:
    
    Split into:
    1. 2x2x2 Strassen (7 terms) for i,j in {0,1}, k in {0,1}
    2. For i=2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
       Use 3 terms: (A[2,0],B[0,k],C[2,k]) for k=0,1 = 2 terms
                    (A[2,1],B[1,k],C[2,k]) for k=0,1 = 2 terms  
                    (A[2,2],B[2,k],C[2,k]) for k=0,1 = 2 terms
       That's 6 terms. Total so far: 13.
    3. For j=2, i in {0,1}: C[i,k] += A[i,2]*B[2,k]
       (A[0,2],B[2,0],C[0,0]), (A[0,2],B[2,1],C[0,1]) = 2 terms
       (A[1,2],B[2,0],C[1,0]), (A[1,2],B[2,1],C[1,1]) = 2 terms
       That's 4 terms. Total: 17.
    
    So rank 17 is easy. To get to 12, I need to be cleverer.
    
    Let me try: combine the i=2 and j=2 parts.
    
    Alternative: Use a 2x3x2 decomposition (rank 8) + 1x3x2 (rank 3) = 11?
    
    2x3x2: A[0:2,:] (2x3) * B (3x2) -> C[0:2,:] (2x2)
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    Terms:
    t1: (A[0,0]+A[0,1], B[0,0]+B[1,0], C[0,0])
    t2: (A[0,0]+A[0,1], B[0,1]+B[1,1], C[0,1])
    t3: (A[1,0]+A[1,1], B[0,0]+B[1,0], C[1,0])
    t4: (A[1,0]+A[1,1], B[0,1]+B[1,1], C[1,1])
    t5: (A[0,2], B[2,0], C[0,0])
    t6: (A[0,2], B[2,1], C[0,1])
    t7: (A[1,2], B[2,0], C[1,0])
    t8: (A[1,2], B[2,1], C[1,1])
    
    Verify t1: contributes (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) to C[0,0]
    = A[0,0]*B[0,0] + A[0,0]*B[1,0] + A[0,1]*B[0,0] + A[0,1]*B[1,0]
    But we only want A[0,0]*B[0,0] + A[0,1]*B[1,0] in C[0,0].
    The extra terms A[0,0]*B[1,0] + A[0,1]*B[0,0] are wrong!
    
    So this naive grouping doesn't work. I need to use Strassen-like corrections.
    
    Let me just use the verified schoolbook approach. The safest bet is rank 54.
    But let me try one more thing: a rank-14 decomposition for 3x3x2.
    
    Actually, I realize I should just verify my constructions numerically.
    Let me construct candidate decompositions and verify them.
    """
    # For now, return None to indicate we should use schoolbook
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Try to construct a rank-36 decomposition (3 x rank-12 for 3x3x2)
    # If it verifies, use it. Otherwise fall back to schoolbook rank 54.
    
    # Let me try to construct a valid decomposition for 3x3x2 with rank 12.
    # I'll use a systematic approach and verify.
    
    # For 3x3x2: A(3x3) * B(3x2) -> C(3x2)
    # U: 9 x r, V: 6 x r, W: 6 x r
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # m=3, p=2, n=3
    # U index: 3i+j, V index: 2j+k, W index: 2i+k
    
    # Let me try a rank-12 construction:
    # I'll use the following approach:
    # - 2x2x2 Strassen for top-left (7 terms)
    # - Handle the cross terms efficiently
    
    # Strassen for 2x2x2 (i,j in {0,1}, k in {0,1}):
    # M0 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> C[0,0] += M0, C[0,1] += 0, C[1,0] += 0, C[1,1] += M0? No.
    # Standard Strassen:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)b00
    # M2 = a00(b10-b11)
    # M3 = a11(b01-b00)
    # M4 = (a00+a01)b11
    # M5 = (a10-a00)(b00+b01)
    # M6 = (a01-a11)(b10+b11)
    # c00 = M0 - M3 - M5 + M6
    # c01 = M2 + M4
    # c10 = M1 + M5
    # c11 = M0 - M1 - M2 + M3
    
    # Let me encode these 7 terms for the 3x3x2 block (indices for 3x3x2):
    # A indices: 0:a00, 1:a01, 2:a02, 3:a10, 4:a11, 5:a12, 6:a20, 7:a21, 8:a22
    # B indices: 0:b00, 1:b01, 2:b10, 3:b11, 4:b20, 5:b21
    # C indices: 0:c00, 1:c01, 2:c10, 3:c11, 4:c20, 5:c21
    # (C index = 2i+k where i is row, k is col)
    
    # Strassen terms (7):
    strassen_332 = [
        # M0 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1 -> [1,0,0, 0,1,0, 0,0,0]
        # V: b00=1, b11=1 -> [1,0, 0,1, 0,0]
        # W: c00=1, c11=1 -> [1,0, 0,1, 0,0]  (c00=idx0, c11=idx3)
        ([1,0,0, 0,1,0, 0,0,0], [1,0, 0,1, 0,0], [1,0, 0,1, 0,0]),
        # M1 = (a10+a11)b00
        # U: a10=1, a11=1 -> [0,0,0, 1,1,0, 0,0,0]
        # V: b00=1 -> [1,0, 0,0, 0,0]
        # W: c10=1 -> [0,0, 1,0, 0,0]  (c10=idx2)
        ([0,0,0, 1,1,0, 0,0,0], [1,0, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M2 = a00(b10-b11)
        # U: a00=1 -> [1,0,0, 0,0,0, 0,0,0]
        # V: b10=1, b11=-1 -> [0,0, 1,-1, 0,0]
        # W: c01=1 -> [0,1, 0,0, 0,0]  (c01=idx1)
        ([1,0,0, 0,0,0, 0,0,0], [0,0, 1,-1, 0,0], [0,1, 0,0, 0,0]),
        # M3 = a11(b01-b00)
        # U: a11=1 -> [0,0,0, 0,1,0, 0,0,0]
        # V: b01=1, b00=-1 -> [-1,1, 0,0, 0,0]
        # W: c00=-1 -> [-1,0, 0,0, 0,0]
        ([0,0,0, 0,1,0, 0,0,0], [-1,1, 0,0, 0,0], [-1,0, 0,0, 0,0]),
        # M4 = (a00+a01)b11
        # U: a00=1, a01=1 -> [1,1,0, 0,0,0, 0,0,0]
        # V: b11=1 -> [0,0, 0,1, 0,0]
        # W: c01=1 -> [0,1, 0,0, 0,0]
        ([1,1,0, 0,0,0, 0,0,0], [0,0, 0,1, 0,0], [0,1, 0,0, 0,0]),
        # M5 = (a10-a00)(b00+b01)
        # U: a10=1, a00=-1 -> [-1,0,0, 1,0,0, 0,0,0]
        # V: b00=1, b01=1 -> [1,1, 0,0, 0,0]
        # W: c10=1 -> [0,0, 1,0, 0,0]
        ([-1,0,0, 1,0,0, 0,0,0], [1,1, 0,0, 0,0], [0,0, 1,0, 0,0]),
        # M6 = (a01-a11)(b10+b11)
        # U: a01=1, a11=-1 -> [0,1,0, 0,-1,0, 0,0,0]
        # V: b10=1, b11=1 -> [0,0, 1,1, 0,0]
        # W: c00=1 -> [1,0, 0,0, 0,0]
        ([0,1,0, 0,-1,0, 0,0,0], [0,0, 1,1, 0,0], [1,0, 0,0, 0,0]),
    ]
    
    # Now handle the remaining terms:
    # i=2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k]
    # j=2, i in {0,1}: C[i,k] += A[i,2]*B[2,k]
    
    # For i=2, k=0: C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # For i=2, k=1: C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # For i=0, k=0: C[0,0] += A[0,2]*B[2,0]
    # For i=0, k=1: C[0,1] += A[0,2]*B[2,1]
    # For i=1, k=0: C[1,0] += A[1,2]*B[2,0]
    # For i=1, k=1: C[1,1] += A[1,2]*B[2,1]
    
    # That's 6 + 4 = 10 more terms, total 17.
    # To reduce, I can combine:
    # (A[2,0]+A[2,1], B[0,0]+B[1,0], C[2,0]) - but this gives cross terms.
    
    # Let me just use 10 simple terms:
    remaining_332 = [
        # A[2,0]*B[0,0] -> C[2,0]
        ([0,0,0, 0,0,0, 1,0,0], [1,0, 0,0, 0,0], [0,0, 0,0, 1,0]),  # C[2,0]=idx4
        # A[2,0]*B[0,1] -> C[2,1]
        ([0,0,0, 0,0,0, 1,0,0], [0,1, 0,0, 0,0], [0,0, 0,0, 0,1]),  # C[2,1]=idx5
        # A[2,1]*B[1,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 1,0, 0,0], [0,0, 0,0, 1,0]),
        # A[2,1]*B[1,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,1,0], [0,0, 0,1, 0,0], [0,0, 0,0, 0,1]),
        # A[2,2]*B[2,0] -> C[2,0]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 1,0], [0,0, 0,0, 1,0]),
        # A[2,2]*B[2,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,0,1], [0,0, 0,0, 0,1], [0,0, 0,0, 0,1]),
        # A[0,2]*B[2,0] -> C[0,0]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 1,0], [1,0, 0,0, 0,0]),
        # A[0,2]*B[2,1] -> C[0,1]
        ([0,0,1, 0,0,0, 0,0,0], [0,0, 0,0, 0,1], [0,1, 0,0, 0,0]),
        # A[1,2]*B[2,0] -> C[1,0]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 1,0], [0,0, 1,0, 0,0]),
        # A[1,2]*B[2,1] -> C[1,1]
        ([0,0,0, 0,0,1, 0,0,0], [0,0, 0,0, 0,1], [0,0, 0,1, 0,0]),
    ]
    
    # Total rank for 3x3x2: 7 + 10 = 17
    # Total rank for 3x3x6: 3 * 17 = 51
    
    # Let me verify this 3x3x2 decomposition
    terms_332 = strassen_332 + remaining_332
    r = len(terms_332)
    U_block = [[0]*r for _ in range(9)]
    V_block = [[0]*r for _ in range(6)]
    W_block = [[0]*r for _ in range(6)]
    for c, (u, v, w) in enumerate(terms_332):
        for idx in range(9):
            U_block[idx][c] = u[idx]
        for idx in range(6):
            V_block[idx][c] = v[idx]
            W_block[idx][c] = w[idx]
    
    # Verify
    valid_block = _verify_decomposition(U_block, V_block, W_block, 3, 3, 2)
    
    if valid_block:
        # Build the full 3x3x6 decomposition
        # The full problem: A(3x3) * B(3x6) -> C(3x6)
        # B is split into B1(3x2), B2(3x2), B3(3x2)
        # C is split into C1(3x2), C2(3x2), C3(3x2)
        # For each block t=0,1,2:
        #   B_t has indices: B[j, 2t+k] for j=0..2, k=0..1
        #   C_t has indices: C[i, 2t+k] for i=0..2, k=0..1
        #   A is the same
        #
        # Full U: 9 x (3*17) = 9x51
        # Full V: 18 x 51
        # Full W: 18 x 51
        #
        # U index: i*m+j = 3i+j (same for all blocks)
        # V index: j*p+k = j*6+k (for block t, k goes 0..1, so actual k = 2t + local_k)
        # W index: k*n+i = (2t+local_k)*3+i
        
        total_rank = 3 * r
        U_full = [[0]*total_rank for _ in range(9)]
        V_full = [[0]*total_rank for _ in range(18)]
        W_full = [[0]*total_rank for _ in range(18)]
        
        for t in range(3):
            for c in range(r):
                col = t * r + c
                u, v, w = terms_332[c]
                # U: same indices
                for idx in range(9):
                    U_full[idx][col] = u[idx]
                # V: local index j*2+k -> global index j*6 + (2t+k)
                for j in range(3):
                    for k in range(2):
                        local_v_idx = j*2 + k
                        global_v_idx = j*6 + (2*t + k)
                        V_full[global_v_idx][col] = v[local_v_idx]
                # W: local index 2*i+k -> global index (2t+k)*3 + i
                for i in range(3):
                    for k in range(2):
                        local_w_idx = 2*i + k
                        global_w_idx = (2*t + k)*3 + i
                        W_full[global_w_idx][col] = w[local_w_idx]
        
        # Verify full decomposition
        if _verify_decomposition(U_full, V_full, W_full, 3, 3, 6):
            return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback: schoolbook rank 54
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
