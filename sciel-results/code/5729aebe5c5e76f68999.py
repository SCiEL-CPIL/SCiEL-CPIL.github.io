"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]
import time
_start_time = time.time()
TIME_LIMIT = 130  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the 4x4x4 part (rank 49) and schoolbook for the 4x4x3 part (rank 48)
    # Total rank: 97
    # U shape: (n*m, rank) = (16, 97)
    # V shape: (m*p, rank) = (28, 97)
    # W shape: (p*n, rank) = (28, 97)
    
    rank_strassen = 49
    rank_schoolbook = 48
    total_rank = rank_strassen + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen for 4x4x4 (columns 0 to 48) ---
    # We need to construct the 49 terms.
    # Let A be 4x4, B be 4x4.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # C = A * B.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to the scalar level.
    # Each Mk is a 2x2 matrix product.
    # Let's use the rank-7 decomposition of 2x2x2 for each Mk.
    # The 2x2x2 decomposition terms (S1..S7) for inputs (a11,a12,a21,a22) and (b11,b12,b21,b22):
    # S1 = (a11+a22)(b11+b22)
    # S2 = (a21+a22)b11
    # S3 = a11(b12-b22)
    # S4 = a22(b21-b11)
    # S5 = (a11+a12)b22
    # S6 = (a21-a11)(b11+b12)
    # S7 = (a12-a22)(b21+b22)
    # Result:
    # c11 = S1 + S4 - S5 + S7
    # c12 = S3 + S5
    # c21 = S2 + S4
    # c22 = S1 - S2 + S3 + S6
    
    # We will define a helper to add a term to U, V, W.
    # u_idx: row in U (0..15)
    # v_idx: row in V (0..27)
    # w_idx: row in W (0..27)
    # val: coefficient (1, -1, or 0)
    # col: current column index
    
    def add_term(u_idx, v_idx, w_idx, val, col):
        if val != 0:
            U[u_idx][col] += val
            V[v_idx][col] += val
            W[w_idx][col] += val
            
    # Indices for A (4x4):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    # Note: U index is i*m + j.
    # A11: i in {0,1}, j in {0,1} -> 0,1,4,5
    # A12: i in {0,1}, j in {2,3} -> 2,3,6,7
    # A21: i in {2,3}, j in {0,1} -> 8,9,12,13
    # A22: i in {2,3}, j in {2,3} -> 10,11,14,15
    
    # Indices for B (4x4 part, so p=4 effectively for this block):
    # B11: j in {0,1}, k in {0,1} -> 0,1,4,5 (V index j*p+k)
    # B12: j in {0,1}, k in {2,3} -> 2,3,6,7
    # B21: j in {2,3}, k in {0,1} -> 8,9,12,13
    # B22: j in {2,3}, k in {2,3} -> 10,11,14,15
    
    # Indices for C (4x4 part, so n=4 effectively for this block):
    # C11: k in {0,1}, i in {0,1} -> 0,1,4,5 (W index k*n+i)
    # C12: k in {0,1}, i in {2,3} -> 2,3,6,7
    # C21: k in {2,3}, i in {0,1} -> 8,9,12,13
    # C22: k in {2,3}, i in {2,3} -> 10,11,14,15
    
    # Let's define the 2x2 blocks indices
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # We need to compute 7 matrix multiplications (M1..M7) of 2x2 matrices.
    # Each requires 7 scalar multiplications.
    # Total 49 terms.
    
    # For each Mk, we have a set of U indices (with coeffs), V indices (with coeffs), and W indices (with coeffs).
    # Then we apply the 2x2 Strassen decomposition to the resulting 2x2 matrices.
    
    # Let's define the "linear combinations" for U and V for each Mk.
    # U is for A, V is for B.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We will store for each Mk:
    # U_terms: list of (idx, coeff)
    # V_terms: list of (idx, coeff)
    # Then for each of the 7 sub-terms (S1..S7) of the 2x2 multiplication, we combine these.
    
    # Actually, it's easier to just iterate through the 49 combinations directly.
    # Let's define the 7 sub-terms of 2x2 Strassen.
    # S1: U: (a11+a22), V: (b11+b22)
    # S2: U: (a21+a22), V: b11
    # S3: U: a11, V: (b12-b22)
    # S4: U: a22, V: (b21-b11)
    # S5: U: (a11+a12), V: b22
    # S6: U: (a21-a11), V: (b11+b12)
    # S7: U: (a12-a22), V: (b21+b22)
    
    # And the output W terms for the 2x2 result:
    # c11: S1 + S4 - S5 + S7
    # c12: S3 + S5
    # c21: S2 + S4
    # c22: S1 - S2 + S3 + S6
    
    # We will map these to the global W indices (C11, C12, etc.).
    # For M1, the result contributes to C11.
    # So if S1 contributes +1 to c11 of the local 2x2 result, it contributes +1 to global C11 indices.
    
    # Let's define the local W indices for the 2x2 result.
    # Local C11: [0,1,4,5] -> Global C11
    # Local C12: [2,3,6,7] -> Global C12
    # Local C21: [8,9,12,13] -> Global C21
    # Local C22: [10,11,14,15] -> Global C22
    
    # We need to be careful: the "local" 2x2 result is a 2x2 matrix.
    # Its entries are mapped to the global 4x4 C matrix.
    # For M1, the local 2x2 result IS the contribution to C11 (a 2x2 block).
    # So local (0,0) -> global C11[0] (which is C[0,0])
    # local (0,1) -> global C11[1] (which is C[0,1])
    # local (1,0) -> global C11[4] (which is C[1,0])
    # local (1,1) -> global C11[5] (which is C[1,1])
    
    # Wait, W index is k*n + i.
    # C11 corresponds to k in {0,1} and i in {0,1}.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 4
    # k=1, i=1 -> 5
    # This matches [0,1,4,5].
    
    # So for each Mk, we have:
    # - A linear combination of A indices (U)
    # - A linear combination of B indices (V)
    # - A target C block (C11, C12, etc.)
    
    # Let's define the data structure for the 7 Mk's.
    # M1: U: A11+A22, V: B11+B22, Target: C11
    # M2: U: A21+A22, V: B11, Target: C21
    # M3: U: A11, V: B12-B22, Target: C12
    # M4: U: A22, V: B21-B11, Target: C21 (Wait, M4 is in C11 and C21? No.)
    # Let's re-verify Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So:
    # M1 contributes to C11 (+1) and C22 (+1)
    # M2 contributes to C21 (+1) and C22 (-1)
    # M3 contributes to C12 (+1) and C22 (+1)
    # M4 contributes to C11 (+1) and C21 (+1)
    # M5 contributes to C11 (-1) and C12 (+1)
    # M6 contributes to C22 (+1)
    # M7 contributes to C11 (+1)
    
    # For each Mk, we perform a 2x2 matrix multiplication.
    # The result is a 2x2 matrix.
    # This 2x2 matrix is added to the appropriate C blocks.
    # Since each Mk is a full 2x2 matrix multiplication, its result (a 2x2 matrix) is added to the target C block(s).
    # But wait, Strassen is for computing C = A*B.
    # M1 = (A11+A22)(B11+B22). This is a 2x2 matrix.
    # This 2x2 matrix is part of C11 and C22.
    # Specifically, the 2x2 matrix M1 is added to C11 and C22.
    # So if M1[0,0] = val, then C[0,0] += val and C[2,2] += val.
    
    # So for each Mk, we need to:
    # 1. Identify the U indices (linear combo of A)
    # 2. Identify the V indices (linear combo of B)
    # 3. Identify the target C blocks and their signs.
    # 4. Use the 2x2 Strassen decomposition to generate 7 scalar terms.
    # 5. For each scalar term, add it to U, V, and W (with appropriate signs and target indices).
    
    # Let's define the U and V indices for each Mk.
    # A11 = [0,1,4,5], A12 = [2,3,6,7], A21 = [8,9,12,13], A22 = [10,11,14,15]
    # B11 = [0,1,4,5], B12 = [2,3,6,7], B21 = [8,9,12,13], B22 = [10,11,14,15]
    
    # M1: U: A11+A22, V: B11+B22
    # M2: U: A21+A22, V: B11
    # M3: U: A11, V: B12-B22
    # M4: U: A22, V: B21-B11
    # M5: U: A11+A12, V: B22
    # M6: U: A21-A11, V: B11+B12
    # M7: U: A12-A22, V: B21+B22
    
    # Target blocks for each Mk:
    # M1: C11 (+1), C22 (+1)
    # M2: C21 (+1), C22 (-1)
    # M3: C12 (+1), C22 (+1)
    # M4: C11 (+1), C21 (+1)
    # M5: C11 (-1), C12 (+1)
    # M6: C22 (+1)
    # M7: C11 (+1)
    
    # For each Mk, we have 7 sub-terms (S1..S7).
    # Each sub-term has:
    # - U local indices (linear combo of a11,a12,a21,a22)
    # - V local indices (linear combo of b11,b12,b21,b22)
    # - W local indices (linear combo of c11,c12,c21,c22) with signs
    
    # Let's define the 2x2 Strassen sub-terms.
    # Local A indices: a11=0, a12=1, a21=2, a22=3 (in the 2x2 block)
    # Local B indices: b11=0, b12=1, b21=2, b22=3
    # Local C indices: c11=0, c12=1, c21=2, c22=3
    
    # S1: U: a11+a22, V: b11+b22
    # S2: U: a21+a22, V: b11
    # S3: U: a11, V: b12-b22
    # S4: U: a22, V: b21-b11
    # S5: U: a11+a12, V: b22
    # S6: U: a21-a11, V: b11+b12
    # S7: U: a12-a22, V: b21+b22
    
    # W contributions:
    # c11: S1 (+1), S4 (+1), S5 (-1), S7 (+1)
    # c12: S3 (+1), S5 (+1)
    # c21: S2 (+1), S4 (+1)
    # c22: S1 (+1), S2 (-1), S3 (+1), S6 (+1)
    
    # Now we need to map local indices to global indices.
    # For a given Mk, the local A indices (a11..a22) map to the global U indices defined for that Mk.
    # E.g., for M1, a11 maps to A11 indices, a12 maps to A12 indices, etc.
    # But wait, the U indices for M1 are A11+A22.
    # So a11 (local) corresponds to the first 2x2 block of A in M1, which is A11.
    # a22 (local) corresponds to A22.
    # What about a12 and a21? They are not used in M1's U definition?
    # No, M1 is (A11+A22)(B11+B22).
    # So the 2x2 matrix A_M1 is:
    # [[A11, 0],
    #  [0, A22]]
    # Wait, no. A11 and A22 are 2x2 matrices themselves.
    # So A_M1 is a 4x4 matrix? No.
    # A11 is a 2x2 matrix. A22 is a 2x2 matrix.
    # A11 + A22 is a 2x2 matrix (element-wise sum).
    # So the "A" in M1 is a 2x2 matrix whose entries are themselves 2x2 matrices?
    # No, that's not how Strassen works.
    # In Strassen for 4x4x4, we split A into four 2x2 blocks: A11, A12, A21, A22.
    # Each of these is a 2x2 matrix of scalars.
    # M1 = (A11 + A22) * (B11 + B22).
    # Here, A11 + A22 is a 2x2 matrix of scalars.
    # B11 + B22 is a 2x2 matrix of scalars.
    # So M1 is a 2x2 matrix of scalars.
    # This 2x2 matrix is then used to form C11 and C22.
    # So the "multiplication" in M1 is a 2x2 matrix multiplication of scalars.
    # And we use the rank-7 decomposition for 2x2x2 to compute this.
    
    # So for M1:
    # The "A" for the 2x2 multiplication is the 2x2 matrix (A11 + A22).
    # The "B" for the 2x2 multiplication is the 2x2 matrix (B11 + B22).
    # The "C" result is a 2x2 matrix, which is added to C11 and C22.
    
    # So we need to define the U indices for the 2x2 matrix (A11 + A22).
    # This 2x2 matrix has 4 entries:
    # (0,0): A[0,0] + A[2,2]
    # (0,1): A[0,1] + A[2,3]
    # (1,0): A[1,0] + A[3,2]
    # (1,1): A[1,1] + A[3,3]
    
    # U indices:
    # A[0,0] -> 0
    # A[0,1] -> 1
    # A[1,0] -> 4
    # A[1,1] -> 5
    # A[2,2] -> 10
    # A[2,3] -> 11
    # A[3,2] -> 14
    # A[3,3] -> 15
    
    # So for M1, the U indices for the 2x2 matrix are:
    # u0 (a11) = [0, 10] (coeff 1)
    # u1 (a12) = [1, 11] (coeff 1)
    # u2 (a21) = [4, 14] (coeff 1)
    # u3 (a22) = [5, 15] (coeff 1)
    
    # Similarly for V indices for (B11 + B22):
    # v0 (b11) = [0, 10]
    # v1 (b12) = [1, 11]
    # v2 (b21) = [4, 14]
    # v3 (b22) = [5, 15]
    
    # And the W indices for the result 2x2 matrix (which is added to C11 and C22):
    # w0 (c11) = C11 indices [0, 1, 4, 5]
    # w1 (c12) = C12 indices [2, 3, 6, 7]
    # w2 (c21) = C21 indices [8, 9, 12, 13]
    # w3 (c22) = C22 indices [10, 11, 14, 15]
    
    # But wait, the result of M1 is added to C11 and C22.
    # So the local c11 (0,0) of the result maps to C11[0,0] and C22[0,0].
    # C11[0,0] is C[0,0] -> W index 0.
    # C22[0,0] is C[2,2] -> W index 10 (k=2, i=2 -> 2*4+2=10).
    
    # So for M1, the W mapping is:
    # local c11 -> global W indices [0, 10] (coeff 1)
    # local c12 -> global W indices [1, 11] (coeff 1)
    # local c21 -> global W indices [4, 14] (coeff 1)
    # local c22 -> global W indices [5, 15] (coeff 1)
    
    # This is getting complex to hardcode. Let's write a helper to generate the U, V, W indices for each Mk.
    
    def get_m1_indices():
        # A: A11+A22, B: B11+B22
        # A11: [0,1,4,5], A22: [10,11,14,15]
        # B11: [0,1,4,5], B22: [10,11,14,15]
        # Result: C11 (+1), C22 (+1)
        # C11: [0,1,4,5], C22: [10,11,14,15]
        u = [[0,10], [1,11], [4,14], [5,15]]
        v = [[0,10], [1,11], [4,14], [5,15]]
        w = [[0,10], [1,11], [4,14], [5,15]]
        return u, v, w
        
    def get_m2_indices():
        # A: A21+A22, B: B11
        # A21: [8,9,12,13], A22: [10,11,14,15]
        # B11: [0,1,4,5]
        # Result: C21 (+1), C22 (-1)
        # C21: [8,9,12,13], C22: [10,11,14,15]
        u = [[8,10], [9,11], [12,14], [13,15]]
        v = [[0,0], [1,0], [4,0], [5,0]]
        # v is a 2x2 matrix, but B11 is the whole thing.
        # So v0=b11=[0], v1=b12=[0], v2=b21=[0], v3=b22=[0]? No.
        # B is B11. So the 2x2 matrix B is:
        # [[B11[0,0], B11[0,1]],
        #  [B11[1,0], B11[1,1]]]
        # So v0 = [0], v1 = [1], v2 = [4], v3 = [5]
        v = [[0], [1], [4], [5]]
        # W: local c11 -> C21[0,0] and C22[0,0] (with sign -1 for C22)
        # C21[0,0] is C[2,0] -> W index 8 (k=2, i=0 -> 2*4+0=8)
        # C22[0,0] is C[2,2] -> W index 10
        # So w0 = [8, 10] with signs [1, -1]
        # This is tricky because the signs are different for the same local term.
        # We need to handle this by adding two separate terms or by storing signed indices.
        # Let's modify add_term to accept a list of (idx, sign) for W.
        w = [[(8,1), (10,-1)], [(9,1), (11,-1)], [(12,1), (14,-1)], [(13,1), (15,-1)]]
        return u, v, w
        
    # This is getting messy. Let's use a more systematic approach.
    # We will generate the 49 terms by iterating through the 7 Mk's and the 7 sub-terms.
    # For each combination, we determine the U, V, W indices and signs.
    
    # Let's define the 2x2 Strassen sub-terms again.
    # S1: U: a11+a22, V: b11+b22, W: c11(+1), c22(+1)
    # S2: U: a21+a22, V: b11, W: c21(+1), c22(-1)
    # S3: U: a11, V: b12-b22, W: c12(+1), c22(+1)
    # S4: U: a22, V: b21-b11, W: c11(+1), c21(+1)
    # S5: U: a11+a12, V: b22, W: c11(-1), c12(+1)
    # S6: U: a21-a11, V: b11+b12, W: c22(+1)
    # S7: U: a12-a22, V: b21+b22, W: c11(+1)
    
    # For each Mk, we have a mapping from local indices (a11..a22, b11..b22, c11..c22) to global indices.
    # Let's define this mapping for each Mk.
    
    def get_mapping(mk):
        # Returns: u_map, v_map, w_map
        # u_map: list of 4 lists, u_map[i] is list of (global_idx, sign) for local a_i
        # v_map: list of 4 lists, v_map[i] is list of (global_idx, sign) for local b_i
        # w_map: list of 4 lists, w_map[i] is list of (global_idx, sign) for local c_i
        
        if mk == 1:
            # M1: A11+A22, B11+B22, C11(+1), C22(+1)
            u_map = [
                [(0,1), (10,1)],  # a11 -> A11[0,0] + A22[0,0]
                [(1,1), (11,1)],  # a12 -> A11[0,1] + A22[0,1]
                [(4,1), (14,1)],  # a21 -> A11[1,0] + A22[1,0]
                [(5,1), (15,1)]   # a22 -> A11[1,1] + A22[1,1]
            ]
            v_map = [
                [(0,1), (10,1)],  # b11
                [(1,1), (11,1)],  # b12
                [(4,1), (14,1)],  # b21
                [(5,1), (15,1)]   # b22
            ]
            w_map = [
                [(0,1), (10,1)],  # c11 -> C11[0,0] + C22[0,0]
                [(1,1), (11,1)],  # c12 -> C11[0,1] + C22[0,1]
                [(4,1), (14,1)],  # c21 -> C11[1,0] + C22[1,0]
                [(5,1), (15,1)]   # c22 -> C11[1,1] + C22[1,1]
            ]
        elif mk == 2:
            # M2: A21+A22, B11, C21(+1), C22(-1)
            u_map = [
                [(8,1), (10,1)],  # a11 -> A21[0,0] + A22[0,0]
                [(9,1), (11,1)],  # a12 -> A21[0,1] + A22[0,1]
                [(12,1), (14,1)], # a21 -> A21[1,0] + A22[1,0]
                [(13,1), (15,1)]  # a22 -> A21[1,1] + A22[1,1]
            ]
            v_map = [
                [(0,1)],  # b11 -> B11[0,0]
                [(1,1)],  # b12 -> B11[0,1]
                [(4,1)],  # b21 -> B11[1,0]
                [(5,1)]   # b22 -> B11[1,1]
            ]
            w_map = [
                [(8,1), (10,-1)],  # c11 -> C21[0,0] + C22[0,0]
                [(9,1), (11,-1)],  # c12 -> C21[0,1] + C22[0,1]
                [(12,1), (14,-1)], # c21 -> C21[1,0] + C22[1,0]
                [(13,1), (15,-1)]  # c22 -> C21[1,1] + C22[1,1]
            ]
        elif mk == 3:
            # M3: A11, B12-B22, C12(+1), C22(+1)
            u_map = [
                [(0,1)],  # a11 -> A11[0,0]
                [(1,1)],  # a12 -> A11[0,1]
                [(4,1)],  # a21 -> A11[1,0]
                [(5,1)]   # a22 -> A11[1,1]
            ]
            v_map = [
                [(2,1), (10,-1)],  # b11 -> B12[0,0] - B22[0,0]
                [(3,1), (11,-1)],  # b12 -> B12[0,1] - B22[0,1]
                [(6,1), (14,-1)],  # b21 -> B12[1,0] - B22[1,0]
                [(7,1), (15,-1)]   # b22 -> B12[1,1] - B22[1,1]
            ]
            w_map = [
                [(2,1), (10,1)],  # c11 -> C12[0,0] + C22[0,0]
                [(3,1), (11,1)],  # c12 -> C12[0,1] + C22[0,1]
                [(6,1), (14,1)],  # c21 -> C12[1,0] + C22[1,0]
                [(7,1), (15,1)]   # c22 -> C12[1,1] + C22[1,1]
            ]
        elif mk == 4:
            # M4: A22, B21-B11, C11(+1), C21(+1)
            u_map = [
                [(10,1)],  # a11 -> A22[0,0]
                [(11,1)],  # a12 -> A22[0,1]
                [(14,1)],  # a21 -> A22[1,0]
                [(15,1)]   # a22 -> A22[1,1]
            ]
            v_map = [
                [(8,1), (0,-1)],  # b11 -> B21[0,0] - B11[0,0]
                [(9,1), (1,-1)],  # b12 -> B21[0,1] - B11[0,1]
                [(12,1), (4,-1)], # b21 -> B21[1,0] - B11[1,0]
                [(13,1), (5,-1)]  # b22 -> B21[1,1] - B11[1,1]
            ]
            w_map = [
                [(0,1), (8,1)],  # c11 -> C11[0,0] + C21[0,0]
                [(1,1), (9,1)],  # c12 -> C11[0,1] + C21[0,1]
                [(4,1), (12,1)], # c21 -> C11[1,0] + C21[1,0]
                [(5,1), (13,1)]  # c22 -> C11[1,1] + C21[1,1]
            ]
        elif mk == 5:
            # M5: A11+A12, B22, C11(-1), C12(+1)
            u_map = [
                [(0,1), (2,1)],  # a11 -> A11[0,0] + A12[0,0]
                [(1,1), (3,1)],  # a12 -> A11[0,1] + A12[0,1]
                [(4,1), (6,1)],  # a21 -> A11[1,0] + A12[1,0]
                [(5,1), (7,1)]   # a22 -> A11[1,1] + A12[1,1]
            ]
            v_map = [
                [(10,1)],  # b11 -> B22[0,0]
                [(11,1)],  # b12 -> B22[0,1]
                [(14,1)],  # b21 -> B22[1,0]
                [(15,1)]   # b22 -> B22[1,1]
            ]
            w_map = [
                [(0,-1), (2,1)],  # c11 -> C11[0,0] - C12[0,0]
                [(1,-1), (3,1)],  # c12 -> C11[0,1] - C12[0,1]
                [(4,-1), (6,1)],  # c21 -> C11[1,0] - C12[1,0]
                [(5,-1), (7,1)]   # c22 -> C11[1,1] - C12[1,1]
            ]
        elif mk == 6:
            # M6: A21-A11, B11+B12, C22(+1)
            u_map = [
                [(8,1), (0,-1)],  # a11 -> A21[0,0] - A11[0,0]
                [(9,1), (1,-1)],  # a12 -> A21[0,1] - A11[0,1]
                [(12,1), (4,-1)], # a21 -> A21[1,0] - A11[1,0]
                [(13,1), (5,-1)]  # a22 -> A21[1,1] - A11[1,1]
            ]
            v_map = [
                [(0,1), (2,1)],  # b11 -> B11[0,0] + B12[0,0]
                [(1,1), (3,1)],  # b12 -> B11[0,1] + B12[0,1]
                [(4,1), (6,1)],  # b21 -> B11[1,0] + B12[1,0]
                [(5,1), (7,1)]   # b22 -> B11[1,1] + B12[1,1]
            ]
            w_map = [
                [(10,1)],  # c11 -> C22[0,0]
                [(11,1)],  # c12 -> C22[0,1]
                [(14,1)],  # c21 -> C22[1,0]
                [(15,1)]   # c22 -> C22[1,1]
            ]
        elif mk == 7:
            # M7: A12-A22, B21+B22, C11(+1)
            u_map = [
                [(2,1), (10,-1)],  # a11 -> A12[0,0] - A22[0,0]
                [(3,1), (11,-1)],  # a12 -> A12[0,1] - A22[0,1]
                [(6,1), (14,-1)],  # a21 -> A12[1,0] - A22[1,0]
                [(7,1), (15,-1)]   # a22 -> A12[1,1] - A22[1,1]
            ]
            v_map = [
                [(8,1), (10,1)],  # b11 -> B21[0,0] + B22[0,0]
                [(9,1), (11,1)],  # b12 -> B21[0,1] + B22[0,1]
                [(12,1), (14,1)], # b21 -> B21[1,0] + B22[1,0]
                [(13,1), (15,1)]  # b22 -> B21[1,1] + B22[1,1]
            ]
            w_map = [
                [(0,1)],  # c11 -> C11[0,0]
                [(1,1)],  # c12 -> C11[0,1]
                [(4,1)],  # c21 -> C11[1,0]
                [(5,1)]   # c22 -> C11[1,1]
            ]
        return u_map, v_map, w_map

    # Define the 2x2 Strassen sub-terms
    # Each sub-term is a tuple: (u_local, v_local, w_local)
    # u_local: list of (local_idx, sign)
    # v_local: list of (local_idx, sign)
    # w_local: list of (local_idx, sign)
    
    sub_terms = [
        # S1
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        # S2
        ([(2,1), (3,1)], [(0,1)], [(2,1), (3,-1)]),
        # S3
        ([(0,1)], [(1,1), (3,-1)], [(1,1), (3,1)]),
        # S4
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),
        # S5
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),
        # S6
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        # S7
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)])
    ]
    
    col = 0
    for mk in range(1, 8):
        u_map, v_map, w_map = get_mapping(mk)
        for u_local, v_local, w_local in sub_terms:
            # Generate the term
            for (u_l_idx, u_sign) in u_local:
                for (v_l_idx, v_sign) in v_local:
                    # This is not how it works.
                    # The sub-term is a single scalar multiplication.
                    # u_local is the linear combination for U.
                    # v_local is the linear combination for V.
                    # w_local is the linear combination for W.
                    # So we need to add u_sign * (linear combo of u_l_idx) to U, etc.
                    # But the linear combo is a sum of indices.
                    # So for each (u_l_idx, u_sign) in u_local, we add u_sign to U[u_map[u_l_idx][i][0]] for each (global_idx, global_sign) in u_map[u_l_idx].
                    # Wait, u_map[u_l_idx] is a list of (global_idx, global_sign).
                    # So the total sign is u_sign * global_sign.
                    
                    # Let's just iterate and add.
                    for (g_u_idx, g_u_sign) in u_map[u_l_idx]:
                        U[g_u_idx][col] += u_sign * g_u_sign
                    for (g_v_idx, g_v_sign) in v_map[v_l_idx]:
                        V[g_v_idx][col] += v_sign * g_v_sign
                    for (g_w_idx, g_w_sign) in w_map[w_l_idx]:
                        W[g_w_idx][col] += w_sign * g_w_sign
            col += 1
            
    # --- Part 2: Schoolbook for 4x4x3 (columns 49 to 96) ---
    # We need to compute C[:, 4:7] = A[:, :] * B[:, 4:7]
    # A is 4x4, B is 4x3 (rows 0..3, cols 0..2, but these are cols 4..6 of the original B)
    # Wait, B is 4x7. So B[:, 4:7] is a 4x3 matrix.
    # Let's call it B3.
    # B3[j, k] = B[j, k+4]
    # V index for B3: j*p + (k+4) = j*7 + k + 4.
    # W index for C: k*n + i = k*4 + i. But k here is 0..2, so W indices are 0..11?
    # No, W is p*n = 28x28.
    # The W index is k*n + i, where k is the row index of C (0..6) and i is the column index of C (0..3).
    # So for C[:, 4:7], k is 4..6.
    # So W indices are (4*4+i) to (6*4+i) = 16..27.
    
    # Schoolbook terms:
    # For i in 0..3, k in 4..6:
    #   C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    #   For each j:
    #       U index: i*m + j = i*4 + j
    #       V index: j*p + k = j*7 + k
    #       W index: k*n + i = k*4 + i
    #       Coeff: 1
    
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
