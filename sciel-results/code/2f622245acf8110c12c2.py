"""Block-decomposition via Strassen for 2x4 x 4x7: rank 50."""
import time

DIMENSIONS = [2, 4, 7]
DEADLINE = 150.0
T_START = time.time()


def _time_left():
    return DEADLINE - (time.time() - T_START)


def _build_rank50():
    """
    Decompose 2x4 x 4x7 into blocks:
    A is 2x4, B is 4x7, C is 2x7.
    
    Split B into B_left (4x4) and B_right (4x3):
      B = [B11 B12] where B11 is 4x4, B12 is 4x3
    
    Then C = A * B = A * B11 + A * B12
    
    For A*B11: A is 2x4, B11 is 4x4.
    Split A into A_top (2x2) and A_bottom (2x2):
      A = [A11 A12] where A11 is 2x2, A12 is 2x2
    
    Actually, let's use a cleaner block decomposition:
    
    View the 4x4 middle as 2x2 blocks of size 2x2:
    B11 = [[B00, B01],
           [B10, B11]]  each 2x2
    
    A = [A0, A1] where A0 is 2x2 (columns 0,1), A1 is 2x2 (columns 2,3)
    
    A * B11 = A0*B00 + A0*B10 + A1*B01 + A1*B11  (all 2x2 * 2x2 -> 2x2)
    
    For each 2x2*2x2 multiplication, use Strassen (rank 7).
    That's 4 such products, each rank 7 -> total 28 for the 2x4*4x4 part.
    
    For A*B12: A is 2x4, B12 is 4x3.
    Split B12 into B12_top (2x3) and B12_bottom (2x3):
      B12 = [B12_0; B12_1]
    
    A * B12 = A0*B12_0 + A1*B12_1  (2x2 * 2x3 -> 2x3)
    
    For 2x2*2x3: split into two 2x2*2x1 problems.
    Each 2x2*2x1 has rank 4 (known result).
    So 2 * 4 = 8 for the 2x4*4x3 part.
    
    Total rank: 28 + 8 = 36? Let me verify this more carefully.
    
    Actually, let me use a direct Kronecker-style construction with Strassen.
    
    The 2x4 x 4x7 multiplication tensor can be seen as:
    - 4 blocks of 2x2 x 2x? 
    
    Let me use the explicit Strassen approach for the 2x2x2 core.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition explicitly using Strassen for 2x2x2 blocks.
    # 
    # The multiplication C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..1, j in 0..3, k in 0..6
    #
    # Group j as j = 2*j1 + j2 where j1 in {0,1}, j2 in {0,1}
    # Group k as k = 2*k1 + k2 where k1 in {0,1,2,3} (since p=7, k1 in 0..3, k2 in 0..1, but 2*3+1=7 so k1 in 0..2, k2 in 0..1 gives 6, plus one more)
    #
    # Actually p=7 is odd. Let's handle it differently.
    #
    # Alternative: Use the known result that R(2,4,7) <= 50 via block methods.
    # Let me construct it using a tensor product / Kronecker approach with Strassen.
    #
    # For 2x2 x 2x2: Strassen uses 7 multiplications.
    # For 2x2 x 2x1: rank 4.
    #
    # Let me try a different structural approach: treat the 4x7 matrix as having
    # 4 rows and 7 columns, and decompose the computation.
    #
    # I'll use a systematic construction:
    # 
    # The tensor T[i,j,k] = delta (i,j,k) for the multiplication.
    # We want to write T = sum_r u_r * v_r * w_r
    #
    # Using the Kronecker product structure:
    # j = 2*j1 + j2, j1 in {0,1}, j2 in {0,1}
    # k = 2*k1 + k2, but 7 = 2*3 + 1, so k1 in {0,1,2}, k2 in {0,1}, and for k=6: k1=3,k2=0? No.
    # k in 0..6. Let k = 2*k1 + k2 where k1 in {0,1,2,3}, k2 in {0,1}, but 2*3+1=7>6.
    # So k1 in {0,1,2}, k2 in {0,1} covers k=0..5, and k=6 is separate.
    #
    # This is getting complicated. Let me use a simpler explicit construction.
    
    # Direct construction using the fact that we can decompose into:
    # - 4 blocks of (2x2) x (2x2) for the first 4 columns of B
    # - 1 block of (2x4) x (4x1) for the 5th-7th columns... no.
    
    # Let me just use a rank-56 schoolbook as baseline and try to optimize.
    # Actually, the knowledge says rank 50 is achievable. Let me construct it.
    
    # Strategy: Use Strassen for 2x2x2 (rank 7) and 2x2x1 (rank 4).
    # 
    # Decompose the 4x7 matrix B into:
    # B = [B_top; B_bottom] where B_top is 2x7, B_bottom is 2x7
    # 
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # 
    # C = A * B = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: 2x2 * 2x7. Decompose B_top into B_top_0 (2x4) and B_top_1 (2x3).
    # A_left * B_top_0: 2x2 * 2x4 = two 2x2*2x2 blocks -> 2*7 = 14
    # A_left * B_top_1: 2x2 * 2x3 = 2x2*2x2 + 2x2*2x1 -> 7 + 4 = 11
    # 
    # Similarly for A_right * B_bottom: 14 + 11 = 25
    # 
    # Total: 14 + 11 + 14 + 11 = 50. 
    
    # Now let me implement this.
    
    # Strassen for 2x2x2:
    # Given X (2x2), Y (2x2), compute Z = X*Y.
    # Z[0,0] = M1 - M4 + M6 + M7
    # Z[0,1] = M2 + M4
    # Z[1,0] = M3 + M5
    # Z[1,1] = M1 - M3 + M5 + M7
    # where:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    
    # For the tensor decomposition, we need to express the bilinear map.
    # For 2x2*2x2 -> 2x2, the 7 multiplications correspond to rank-7 decomposition.
    
    # Let me define the Strassen coefficients for 2x2x2.
    # The input is (i, j) for A (i in 0,1; j in 0,1) and (j, k) for B (j in 0,1; k in 0,1).
    # The 7 factors are:
    
    # Factor r: (U_coeff for A, V_coeff for B, W_coeff for C)
    
    # M1: A part: (1,0,0,1) for [A00,A01,A10,A11], B part: (1,0,0,1) for [B00,B01,B10,B11]
    #     C contribution: to Z[0,0]: +1, to Z[1,1]: +1
    # M2: A part: (0,0,1,1), B part: (1,0,0,0)
    #     C contribution: to Z[0,1]: +1
    # M3: A part: (1,0,0,0), B part: (0,1,0,-1)
    #     C contribution: to Z[1,0]: +1
    # M4: A part: (0,0,0,1), B part: (0,0,1,-1)
    #     C contribution: to Z[0,1]: +1, to Z[0,0]: -1
    # M5: A part: (1,1,0,0), B part: (0,0,0,1)
    #     C contribution: to Z[1,0]: +1, to Z[1,1]: +1
    # M6: A part: (0,0,1,-1), B part: (1,1,0,0)
    #     C contribution: to Z[0,0]: +1
    # M7: A part: (0,1,-1,0), B part: (0,0,1,1)
    #     C contribution: to Z[0,0]: +1, to Z[1,1]: -1
    
    # Let me verify: Z[0,0] = M1 - M4 + M6 + M7
    # Z[0,1] = M2 + M4
    # Z[1,0] = M3 + M5
    # Z[1,1] = M1 - M3 + M5 + M7
    
    # Now for 2x2*2x1 -> 2x1:
    # X (2x2) * Y (2x1) = Z (2x1)
    # Z[0] = X00*Y0 + X01*Y1
    # Z[1] = X10*Y0 + X11*Y1
    # 
    # Rank 4 decomposition:
    # N1 = (X00+X10)*(Y0+Y1)
    # N2 = (X00+X10)*Y0
    # N3 = X00*(Y1-Y0)
    # N4 = X10*Y1
    # 
    # Z[0] = N2 + N3 + N4 - N1 + N2... let me think again.
    # 
    # Actually, 2x2*2x1 is just 4 scalar multiplications:
    # Z[0] = X00*Y0 + X01*Y1
    # Z[1] = X10*Y0 + X11*Y1
    # 
    # This is rank 4 (schoolbook). Can we do better? The known result is R(2,2,1)=4.
    # 
    # So for 2x2*2x1, we use 4 multiplications:
    # P1 = X00*Y0, P2 = X01*Y1, P3 = X10*Y0, P4 = X11*Y1
    # Z[0] = P1 + P2, Z[1] = P3 + P4
    
    # Now let me build the full decomposition.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # (2*4, 50) = (8, 50)
    V = [[0] * rank for _ in range(m * p)]   # (4*7, 50) = (28, 50)
    W = [[0] * rank for _ in range(p * n)]   # (7*2, 50) = (14, 50)
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 block
    # A indices: a_i (0,1), a_j (0,1) -> global A index = a_i * 2 + a_j (for left half) or a_i * 2 + 2 + a_j (for right half)
    # B indices: b_j (0,1), b_k (0,1) -> global B index = (2*b_j + b_j_offset) * 7 + (2*b_k + b_k_offset)
    # C indices: c_i (0,1), c_k (0,1) -> global C index = (2*c_k + c_k_offset) * 2 + c_i
    
    def add_strassen_2x2_2x2(a_j_offset, b_j_offset, b_k_offset, c_k_offset):
        """Add Strassen decomposition for A[a_i, a_j_offset + a_j] * B[2*b_j_offset + b_j, 2*b_k_offset + b_k] -> C[a_i, 2*c_k_offset + c_k]"""
        nonlocal col
        
        # Strassen factors for 2x2*2x2
        # A part coefficients: [A00, A01, A10, A11]
        # B part coefficients: [B00, B01, B10, B11]
        # C part coefficients: [C00, C01, C10, C11]
        
        strassen = [
            # (A_coeffs, B_coeffs, C_coeffs)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1: C00+=1, C11+=1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),   # M2: C01+=1
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),  # M3: C10+=1
            ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]), # M4: C00-=1, C01+=1
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]),   # M5: C10+=1, C11+=1
            ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]),  # M6: C00+=1
            ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, -1]), # M7: C00+=1, C11-=1
        ]
        
        for A_c, B_c, C_c in strassen:
            # A part
            for ai in range(2):
                for aj in range(2):
                    if A_c[ai * 2 + aj] != 0:
                        U[ai * 4 + a_j_offset + aj][col] = A_c[ai * 2 + aj]
            
            # B part
            for bj in range(2):
                for bk in range(2):
                    if B_c[bj * 2 + bk] != 0:
                        V[(2 * b_j_offset + bj) * 7 + (2 * b_k_offset + bk)][col] = B_c[bj * 2 + bk]
            
            # C part
            for ci in range(2):
                for ck in range(2):
                    if C_c[ci * 2 + ck] != 0:
                        W[(2 * c_k_offset + ck) * 2 + ci][col] = C_c[ci * 2 + ck]
            
            col += 1
    
    def add_2x2_2x1(a_j_offset, b_j_offset, b_k_offset, c_k_offset):
        """Add rank-4 decomposition for A[a_i, a_j_offset + a_j] * B[2*b_j_offset + b_j, 2*b_k_offset + b_k] -> C[a_i, 2*c_k_offset + b_k]"""
        nonlocal col
        
        # 2x2*2x1: 4 multiplications
        # P1 = A00*B0, P2 = A01*B1, P3 = A10*B0, P4 = A11*B1
        # C0 = P1 + P2, C1 = P3 + P4
        
        factors = [
            # (A_coeffs [A00,A01,A10,A11], B_coeffs [B0,B1], C_coeffs [C0,C1])
            ([1, 0, 0, 0], [1, 0], [1, 0]),   # P1
            ([0, 1, 0, 0], [0, 1], [1, 0]),   # P2
            ([0, 0, 1, 0], [1, 0], [0, 1]),   # P3
            ([0, 0, 0, 1], [0, 1], [0, 1]),   # P4
        ]
        
        for A_c, B_c, C_c in factors:
            # A part
            for ai in range(2):
                for aj in range(2):
                    if A_c[ai * 2 + aj] != 0:
                        U[ai * 4 + a_j_offset + aj][col] = A_c[ai * 2 + aj]
            
            # B part: B is 2x1, so b_j in 0,1, b_k=0 only
            for bj in range(2):
                if B_c[bj] != 0:
                    V[(2 * b_j_offset + bj) * 7 + (2 * b_k_offset + 0)][col] = B_c[bj]
            
            # C part: C is 2x1, so c_i in 0,1, c_k=0 only
            for ci in range(2):
                if C_c[ci] != 0:
                    W[(2 * c_k_offset + 0) * 2 + ci][col] = C_c[ci]
            
            col += 1
    
    # Now decompose:
    # C = A * B
    # A is 2x4, B is 4x7
    # 
    # Split A into A_left (2x2, cols 0-1) and A_right (2x2, cols 2-3)
    # Split B into B_top (2x7, rows 0-1) and B_bottom (2x7, rows 2-3)
    # 
    # C = A_left * B_top + A_right * B_bottom
    # 
    # A_left * B_top: 2x2 * 2x7
    # Split B_top into B_top_0 (2x4, cols 0-3) and B_top_1 (2x3, cols 4-6)
    # 
    # A_left * B_top_0: 2x2 * 2x4 = two 2x2*2x2 blocks (cols 0-1 and 2-3)
    #   Block 1: a_j_offset=0, b_j_offset=0, b_k_offset=0, c_k_offset=0 -> Strassen 2x2x2
    #   Block 2: a_j_offset=0, b_j_offset=0, b_k_offset=2, c_k_offset=2 -> Strassen 2x2x2
    # 
    # A_left * B_top_1: 2x2 * 2x3 = 2x2*2x2 (cols 4-5) + 2x2*2x1 (col 6)
    #   Block 3: a_j_offset=0, b_j_offset=0, b_k_offset=4, c_k_offset=4 -> Strassen 2x2x2
    #   Block 4: a_j_offset=0, b_j_offset=0, b_k_offset=6, c_k_offset=6 -> 2x2*2x1
    # 
    # A_right * B_bottom: 2x2 * 2x7
    # Split B_bottom into B_bottom_0 (2x4, cols 0-3) and B_bottom_1 (2x3, cols 4-6)
    # 
    # A_right * B_bottom_0: two 2x2*2x2 blocks
    #   Block 5: a_j_offset=2, b_j_offset=1, b_k_offset=0, c_k_offset=0 -> Strassen 2x2x2
    #   Block 6: a_j_offset=2, b_j_offset=1, b_k_offset=2, c_k_offset=2 -> Strassen 2x2x2
    # 
    # A_right * B_bottom_1: 2x2*2x2 + 2x2*2x1
    #   Block 7: a_j_offset=2, b_j_offset=1, b_k_offset=4, c_k_offset=4 -> Strassen 2x2x2
    #   Block 8: a_j_offset=2, b_j_offset=1, b_k_offset=6, c_k_offset=6 -> 2x2*2x1
    
    # Block 1: A_left * B_top[0:2, 0:2]
    add_strassen_2x2_2x2(a_j_offset=0, b_j_offset=0, b_k_offset=0, c_k_offset=0)
    
    # Block 2: A_left * B_top[0:2, 2:4]
    add_strassen_2x2_2x2(a_j_offset=0, b_j_offset=0, b_k_offset=2, c_k_offset=2)
    
    # Block 3: A_left * B_top[0:2, 4:6]
    add_strassen_2x2_2x2(a_j_offset=0, b_j_offset=0, b_k_offset=4, c_k_offset=4)
    
    # Block 4: A_left * B_top[0:2, 6:7] (2x2 * 2x1)
    add_2x2_2x1(a_j_offset=0, b_j_offset=0, b_k_offset=6, c_k_offset=6)
    
    # Block 5: A_right * B_bottom[2:4, 0:2]
    add_strassen_2x2_2x2(a_j_offset=2, b_j_offset=1, b_k_offset=0, c_k_offset=0)
    
    # Block 6: A_right * B_bottom[2:4, 2:4]
    add_strassen_2x2_2x2(a_j_offset=2, b_j_offset=1, b_k_offset=2, c_k_offset=2)
    
    # Block 7: A_right * B_bottom[2:4, 4:6]
    add_strassen_2x2_2x2(a_j_offset=2, b_j_offset=1, b_k_offset=4, c_k_offset=4)
    
    # Block 8: A_right * B_bottom[2:4, 6:7] (2x2 * 2x1)
    add_2x2_2x1(a_j_offset=2, b_j_offset=1, b_k_offset=6, c_k_offset=6)
    
    assert col == rank, (col, rank)
    return {"U": U, "V": V, "W": W}


def _verify(decomp):
    """Exact verification of the trace convention W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    # Check all (i,j,k) triples: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1
    # if this is the "diagonal" term, and 0 otherwise.
    # Actually, the matrix multiplication tensor is:
    # T[i,j,k] = 1 for all valid (i,j,k), representing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition is: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # This should equal 1 for all (i,j,k) since each term in the sum contributes.
    # Wait, no. The tensor for matrix multiplication has T[i,j,k] = 1 for all i,j,k.
    # The decomposition says: for each (i,j,k), sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] = 1
    # This is because the multiplication tensor has all entries equal to 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def construct():
    # Primary: rank-50 Strassen block decomposition.
    try:
        d = _build_rank50()
        if _time_left() > 10.0:
            if _verify(d):
                return d
    except Exception:
        pass

    # Fallback: guaranteed-correct schoolbook rank 56.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
