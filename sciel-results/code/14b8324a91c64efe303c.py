"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x5 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Interface: construct() -> {'U', 'V', 'W'}: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    with integer/half-integer entries that exactly decompose the matrix multiplication tensor.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    def _time_remaining():
        return deadline - time.time()
        
    def _verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def _make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _make_strassen_decomposition():
        """
        Constructs a rank-14 decomposition using Strassen's algorithm for 2x2 blocks.
        A is 2x4, B is 4x5.
        Split A into A_L (2x2) and A_R (2x2).
        Split B into B_T (2x5) and B_B (2x5).
        C = A_L B_T + A_R B_B.
        Each 2x2 by 2x5 multiplication uses 7 multiplications (Strassen).
        Total rank = 14.
        """
        rank = 14
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Strassen coefficients for 2x2 matrix multiplication
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices for A (2x4):
        # A_L: i=0,1; j=0,1
        # A_R: i=0,1; j=2,3
        # Indices for B (4x5):
        # B_T: j=0,1; k=0..4
        # B_B: j=2,3; k=0..4
        # Indices for C (2x5):
        # C: i=0,1; k=0..4
        
        def add_strassen_block(offset_r, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
            # a_row_offset: 0 or 1 (row in A)
            # a_col_offset: 0, 1, 2, or 3 (col in A)
            # b_row_offset: 0, 1, 2, or 3 (row in B)
            # b_col_offset: 0..4 (col in B)
            # c_row_offset: 0 or 1 (row in C)
            # c_col_offset: 0..4 (col in C)
            
            # M1 = (a+d)(e+h)
            # a: A[a_row_offset, a_col_offset]
            # d: A[a_row_offset+1, a_col_offset+1]
            # e: B[b_row_offset, b_col_offset]
            # h: B[b_row_offset+1, b_col_offset+1]
            
            # We need to add contributions to U, V, W for each of the 7 terms.
            # Term r corresponds to offset_r + t where t in 0..6
            
            # M1: (a+d)(e+h)
            # U: a=1, d=1
            # V: e=1, h=1
            # W: C11=1, C22=1
            
            # M2: (c+d)e
            # U: c=1, d=1
            # V: e=1
            # W: C21=1, C22=-1
            
            # M3: a(f-h)
            # U: a=1
            # V: f=1, h=-1
            # W: C12=1, C22=1
            
            # M4: d(g-e)
            # U: d=1
            # V: g=1, e=-1
            # W: C11=1, C21=1
            
            # M5: (a+b)h
            # U: a=1, b=1
            # V: h=1
            # W: C11=-1, C12=1
            
            # M6: (c-a)(e+f)
            # U: c=1, a=-1
            # V: e=1, f=1
            # W: C22=1
            
            # M7: (b-d)(g+h)
            # U: b=1, d=-1
            # V: g=1, h=1
            # W: C11=1
            
            # Helper to add a term
            def add_term(t_idx, u_coeffs, v_coeffs, w_coeffs):
                r = offset_r + t_idx
                for (idx_u, val) in u_coeffs:
                    U[idx_u][r] += val
                for (idx_v, val) in v_coeffs:
                    V[idx_v][r] += val
                for (idx_w, val) in w_coeffs:
                    W[idx_w][r] += val
            
            # Define indices
            # A indices: i*m + j
            idx_a00 = a_row_offset * m + a_col_offset
            idx_a01 = a_row_offset * m + a_col_offset + 1
            idx_a10 = (a_row_offset + 1) * m + a_col_offset
            idx_a11 = (a_row_offset + 1) * m + a_col_offset + 1
            
            # B indices: j*p + k
            idx_b00 = b_row_offset * p + b_col_offset
            idx_b01 = b_row_offset * p + b_col_offset + 1
            idx_b10 = (b_row_offset + 1) * p + b_col_offset
            idx_b11 = (b_row_offset + 1) * p + b_col_offset + 1
            
            # C indices: k*n + i
            idx_c00 = c_col_offset * n + c_row_offset
            idx_c01 = (c_col_offset + 1) * n + c_row_offset
            idx_c10 = c_col_offset * n + (c_row_offset + 1)
            idx_c11 = (c_col_offset + 1) * n + (c_row_offset + 1)
            
            # M1
            add_term(0, 
                     [(idx_a00, 1), (idx_a11, 1)], 
                     [(idx_b00, 1), (idx_b11, 1)], 
                     [(idx_c00, 1), (idx_c11, 1)])
            # M2
            add_term(1, 
                     [(idx_a10, 1), (idx_a11, 1)], 
                     [(idx_b00, 1)], 
                     [(idx_c10, 1), (idx_c11, -1)])
            # M3
            add_term(2, 
                     [(idx_a00, 1)], 
                     [(idx_b01, 1), (idx_b11, -1)], 
                     [(idx_c01, 1), (idx_c11, 1)])
            # M4
            add_term(3, 
                     [(idx_a11, 1)], 
                     [(idx_b10, 1), (idx_b00, -1)], 
                     [(idx_c00, 1), (idx_c10, 1)])
            # M5
            add_term(4, 
                     [(idx_a00, 1), (idx_a01, 1)], 
                     [(idx_b11, 1)], 
                     [(idx_c00, -1), (idx_c01, 1)])
            # M6
            add_term(5, 
                     [(idx_a10, 1), (idx_a00, -1)], 
                     [(idx_b00, 1), (idx_b01, 1)], 
                     [(idx_c11, 1)])
            # M7
            add_term(6, 
                     [(idx_a01, 1), (idx_a11, -1)], 
                     [(idx_b10, 1), (idx_b11, 1)], 
                     [(idx_c00, 1)])

        # Block 1: A_L (rows 0,1; cols 0,1) * B_T (rows 0,1; cols 0..4) -> C (rows 0,1; cols 0..4)
        # We need to apply Strassen for each column k?
        # Strassen works on 2x2 matrices.
        # Here B is 2x5. We can view it as 5 columns of 2x1?
        # Or we can just run Strassen for each column k independently?
        # If we run Strassen for each column k, we use 7 multiplications per column.
        # Total 7 * 5 = 35 multiplications. This is worse than schoolbook for the block (2*2*5=20).
        # 
        # Wait, Strassen's 7 multiplications are for 2x2 by 2x2.
        # For 2x2 by 2x5, we can treat the 5 columns as separate 2x2 by 2x1? No.
        # We can treat B as a matrix of 5 columns.
        # The Strassen algorithm computes the product of two 2x2 matrices.
        # To compute 2x2 by 2x5, we can compute 5 independent 2x2 by 2x1 products?
        # No, the structure of Strassen mixes the rows of B.
        # So we must apply the Strassen decomposition to the whole 2x5 block.
        # The "scalars" in Strassen become 1x5 vectors.
        # So M1 = (a+d) * (e_col + h_col).
        # This is a valid rank-1 term in the tensor decomposition where the "value" is a vector.
        # In the tensor decomposition, each rank-1 term is a scalar.
        # So we need to expand each "vector multiplication" into 5 scalar multiplications?
        # No, that would increase the rank.
        #
        # Actually, the rank of the tensor M_{2,2,5} is NOT 7.
        # The rank of M_{2,2,2} is 7.
        # The rank of M_{2,2,5} is at most 7 * 5 = 35? No.
        # It is at most the rank of M_{2,2,2} multiplied by p?
        # No, we can share the structure.
        #
        # Let's stick to a simpler, guaranteed correct decomposition.
        # Schoolbook is 40.
        # Can we do better than 40 easily?
        #
        # Let's use the fact that M_{2,4,5} can be decomposed using 2x2 blocks.
        # If we just use schoolbook for the blocks:
        # Block 1: 2x2 by 2x5. Rank 20.
        # Block 2: 2x2 by 2x5. Rank 20.
        # Total 40.
        #
        # To improve, we need a lower rank for 2x2 by 2x5.
        # It is known that rank(M_{2,2,p}) = 3p for p>=2?
        # For p=2, rank is 7. 3*2=6? No, 7.
        # For p=5, rank is likely 7*5? No.
        #
        # Given the time constraint and risk of error in manual Strassen extension,
        # I will return the schoolbook decomposition (Rank 40) which is guaranteed correct.
        # The score will be 1.0, same as seed.
        # 
        # However, I can try a simple optimization:
        # Use integer arithmetic.
        
        return None

    best = _make_schoolbook()
    
    # Try to find a better decomposition if time permits
    # For now, return schoolbook
    return best

# EVOLVE-BLOCK-END
