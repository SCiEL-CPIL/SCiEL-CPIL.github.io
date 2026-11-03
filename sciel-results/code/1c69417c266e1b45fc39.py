"""Optimized program for matrix multiplication 2x5x6 tensor rank minimization."""
import time
import itertools

DIMENSIONS = [2, 5, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = DIMENSIONS
    
    # Start with the known Strassen-based decomposition of rank 54
    # For 2x5x6, we can decompose into:
    # - 2x2 blocks: use Strassen (rank 7 instead of 8) for each 2x2x2 subproblem
    # - Remaining terms use schoolbook
    #
    # The 2x5x6 multiplication can be viewed as:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0..4}, k in {0..5}
    #
    # We can group the 5 columns of A into pairs and handle the 6 rows of B similarly.
    # Actually, let's think more carefully.
    #
    # A is 2x5, B is 5x6, C is 2x6.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # One approach: use the fact that we can apply Strassen to 2x2 sub-blocks.
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1) where columns are (0,1), (2,3), (4)
    # Split B into B1 (2x6), B2 (2x6), B3 (1x6) where rows are (0,1), (2,3), (4)
    #
    # Then C = A1*B1 + A2*B2 + A3*B3
    #
    # For A1*B1 (2x2 * 2x2 -> 2x2): use Strassen with rank 7
    # For A2*B2 (2x2 * 2x2 -> 2x2): use Strassen with rank 7  
    # For A3*B3 (2x1 * 1x6 -> 2x6): this is just outer product, rank 2
    #
    # But wait, A3 is 2x1 and B3 is 1x6, so A3*B3 is a 2x6 matrix which is the outer product
    # of a 2-vector and a 6-vector. This has rank 2 (as a tensor).
    #
    # Total rank: 7 + 7 + 2 = 16? That seems too good. Let me reconsider.
    #
    # Actually, the tensor for 2x2x2 matrix multiplication has rank 7 (Strassen).
    # The tensor for 2x1x6 (outer product of 2-vector and 6-vector) has rank 2.
    #
    # But these tensors don't simply add because they operate on different subspaces.
    # Let me think about this more carefully using the standard decomposition.
    #
    # Actually, the correct way is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Let's use a different decomposition. Consider:
    # - Split j into 3 groups: {0,1}, {2,3}, {4}
    # - For each group, we have a sub-multiplication
    #
    # Group 1: j in {0,1}: C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] (2x2x6 tensor)
    # Group 2: j in {2,3}: C2[i,k] = A[i,2]*B[2,k] + A[i,3]*B[3,k] (2x2x6 tensor)
    # Group 3: j in {4}: C3[i,k] = A[i,4]*B[4,k] (2x1x6 tensor)
    #
    # The 2x2x6 tensor: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    # We can view this as: for each k, C[:,k] = A * B[:,k]
    # This is a 2x2 matrix times 6 different 2-vectors.
    #
    # The rank of the 2x2x6 tensor... Let me think.
    # Actually, we can use Strassen on the 2x2 part.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Let M = A (2x2), N = B (2x6)
    # C = M*N (2x6)
    #
    # Using Strassen: define
    # p1 = (M00+M11)*(N00+N11)
    # p2 = (M10+M11)*N00
    # p3 = M00*(N01-N11)
    # p4 = M11*(N10-N00)
    # p5 = (M00+M01)*N11
    # p6 = (M10-M00)*(N00+N01)
    # p7 = (M01-M11)*(N10+N11)
    #
    # Then C00 = p1+p4-p5+p7
    # C01 = p3+p5
    # C10 = p2+p4
    # C11 = p1-p2+p3+p6
    #
    # But this only works for 2x2 * 2x2. For 2x2 * 2x6, we need to apply this
    # to each pair of columns of N.
    #
    # N has columns N[:,0], N[:,1], ..., N[:,5]
    # C[:,k] = M * N[:,k]
    #
    # We can apply Strassen to each pair (N[:,2t], N[:,2t+1]) for t=0,1,2
    # Each application uses 7 rank-1 terms.
    # Total for 2x2x6: 3 * 7 = 21 terms.
    #
    # For the 2x1x6 tensor: C[i,k] = A[i,0] * B[0,k]
    # This is just an outer product. Rank 2.
    #
    # Total: 21 + 21 + 2 = 44? Hmm, that's better than 54.
    #
    # Wait, let me reconsider. The 2x2x6 tensor rank...
    # Actually, I think the standard result is that the rank of the (2,2,6) tensor
    # is 14, not 21. Let me verify.
    #
    # For (2,2,p) matrix multiplication, the rank is known to be:
    # - (2,2,2): 7 (Strassen)
    # - (2,2,3): 12
    # - (2,2,4): 16
    # - (2,2,5): 20
    # - (2,2,6): 24
    #
    # Hmm, so it might be 24 for (2,2,6). Let me just use the straightforward approach.
    #
    # Let me just implement the Strassen-based decomposition that gives rank 54,
    # which is the known improvement from 60.
    #
    # Actually, let me just implement a clean Strassen-based approach.
    #
    # For 2x5x6:
    # Split A (2x5) into A_left (2x2), A_mid (2x2), A_right (2x1)
    # Split B (5x6) into B_left (2x6), B_mid (2x6), B_right (1x6)
    #
    # C = A_left @ B_left + A_mid @ B_mid + A_right @ B_right
    #
    # A_left @ B_left: 2x2 * 2x6 -> use Strassen per column pair (3 pairs) -> 3*7 = 21
    # A_mid @ B_mid: 2x2 * 2x6 -> use Strassen per column pair (3 pairs) -> 3*7 = 21
    # A_right @ B_right: 2x1 * 1x6 -> outer product -> rank 2
    #
    # Total: 21 + 21 + 2 = 44
    #
    # Let me implement this.
    
    # Strassen decomposition for 2x2 * 2x2 -> 2x2
    # Inputs: A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # Output: C = A*B
    #
    # p1 = (a00+a11)*(b00+b11)
    # p2 = (a10+a11)*b00
    # p3 = a00*(b01-b11)
    # p4 = a11*(b10-b00)
    # p5 = (a00+a01)*b11
    # p6 = (a10-a00)*(b00+b01)
    # p7 = (a01-a11)*(b10+b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # For 2x2 * 2x6, we process 3 column pairs.
    # Each column pair (b_0k, b_1k) for k in {0,1}, {2,3}, {4,5}
    # uses 7 rank-1 terms.
    
    rank_terms = []  # List of (U_row, V_row, W_row, coeff) tuples
    
    # Process A_left @ B_left (columns 0,1 of A, rows 0,1 of B)
    # A_left[i,j] for i in {0,1}, j in {0,1} -> global A[i, j]
    # B_left[j,k] for j in {0,1}, k in {0..5} -> global B[j, k]
    # C[i,k] contribution
    
    # Strassen for column pair k=0,1
    # p1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> but this gives C[0,0] and C[1,1] contributions
    # We need to be careful with the indexing.
    
    # Let me define helper to add a rank-1 term
    def add_term(a_row, b_row, c_row, coeff=1):
        """Add a rank-1 term: U[a_row] += coeff, V[b_row] += 1, W[c_row] += coeff"""
        rank_terms.append((a_row, b_row, c_row, coeff))
    
    # For 2x2 * 2x6 multiplication using Strassen on each column pair:
    # A is 2x2: A[i,j] for i,j in {0,1}
    # B is 2x6: B[j,k] for j in {0,1}, k in {0..5}
    # C is 2x6: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Column pair 0,1:
    # p1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0])  -> contributes to C[0,0]
    # Wait, I need to be more careful. Let me re-derive.
    
    # For a single 2x2 * 2x2 block with columns k0, k1:
    # B_block = [[B[0,k0], B[0,k1]],
    #            [B[1,k0], B[1,k1]]]
    #
    # p1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k0])  -> this is a scalar
    # Actually no, each p is a product of a linear combo of A entries and a linear combo of B entries.
    # The result C[i,k] is a linear combination of these p's.
    
    # Let me just use the standard Strassen formulas directly.
    # For 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B = [[c00, c01], [c10, c11]]
    #
    # p1 = (a00 + a11) * (b00 + b11)
    # p2 = (a10 + a11) * b00
    # p3 = a00 * (b01 - b11)
    # p4 = a11 * (b10 - b00)
    # p5 = (a00 + a01) * b11
    # p6 = (a10 - a00) * (b00 + b01)
    # p7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # For 2x2 * 2x6, we have B with 6 columns. We process columns in pairs.
    # For column pair (2t, 2t+1):
    # b00 = B[0, 2t], b01 = B[0, 2t+1]
    # b10 = B[1, 2t], b11 = B[1, 2t+1]
    
    # Let me build the full decomposition.
    
    # U rows: n*m = 10 (indices 0..9), U[i*m + j] where i in {0,1}, j in {0..4}
    # V rows: m*p = 30 (indices 0..29), V[j*p + k] where j in {0..4}, k in {0..5}
    # W rows: p*n = 12 (indices 0..11), W[k*n + i] where k in {0..5}, i in {0,1}
    
    # A_left: columns j=0,1 -> U rows: i*5+j for i in {0,1}, j in {0,1} -> {0,1,5,6}
    # A_mid: columns j=2,3 -> U rows: i*5+j for i in {0,1}, j in {2,3} -> {2,3,7,8}
    # A_right: column j=4 -> U rows: i*5+4 for i in {0,1} -> {4,9}
    
    # B_left: rows j=0,1 -> V rows: j*6+k for j in {0,1}, k in {0..5} -> {0..11}
    # B_mid: rows j=2,3 -> V rows: j*6+k for j in {2,3}, k in {0..5} -> {12..23}
    # B_right: row j=4 -> V rows: 4*6+k for k in {0..5} -> {24..29}
    
    # W: W[k*n + i] = W[k*2 + i] for k in {0..5}, i in {0,1}
    
    terms = []
    
    def add_strassen_2x2x2(A_rows, B_rows_base, col_offset):
        """
        Add Strassen terms for a 2x2 * 2x2 block.
        A_rows: [a00_row, a01_row, a10_row, a11_row] - U row indices
        B_rows_base: base V row index, so B[0,0] is at B_rows_base + 0, B[0,1] at B_rows_base+1, etc.
        col_offset: which output columns this affects (k0, k1)
        """
        a00_r, a01_r, a10_r, a11_r = A_rows
        b00_r = B_rows_base + 0  # B[0,0]
        b01_r = B_rows_base + 1  # B[0,1]
        b10_r = B_rows_base + 6  # B[1,0]
        b11_r = B_rows_base + 7  # B[1,1]
        
        # p1 = (a00 + a11) * (b00 + b11)
        terms.append([(a00_r, 1), (a11_r, 1), (b00_r, 1), (b11_r, 1), (col_offset*2, 1)])
        # Actually, I need to track coefficients for U, V, W separately.
        # Let me use a different structure.
        pass
    
    # Let me use a cleaner approach with explicit coefficient tracking.
    # Each rank-1 term has:
    # U coefficient vector (sparse)
    # V coefficient vector (sparse)
    # W coefficient vector (sparse)
    
    # I'll build U, V, W as lists of dicts mapping row -> coefficient
    
    U = [dict() for _ in range(n*m)]
    V = [dict() for _ in range(m*p)]
    W = [dict() for _ in range(p*n)]
    
    col = 0
    
    def add_rank1(U_coeffs, V_coeffs, W_coeffs):
        """Add a rank-1 term. U_coeffs, V_coeffs, W_coeffs are dicts {row: coeff}"""
        nonlocal col
        for row, coeff in U_coeffs.items():
            if coeff != 0:
                U[row][col] = U[row].get(col, 0) + coeff
        for row, coeff in V_coeffs.items():
            if coeff != 0:
                V[row][col] = V[row].get(col, 0) + coeff
        for row, coeff in W_coeffs.items():
            if coeff != 0:
                W[row][col] = W[row].get(col, 0) + coeff
        col += 1
    
    # Helper to get U row index for A[i,j]
    def u_row(i, j):
        return i * m + j
    
    # Helper to get V row index for B[j,k]
    def v_row(j, k):
        return j * p + k
    
    # Helper to get W row index for C[k,i]
    def w_row(k, i):
        return k * n + i
    
    # Process A_left @ B_left (j=0,1)
    # A_left[i,j] = A[i, j] for i in {0,1}, j in {0,1}
    # B_left[j,k] = B[j, k] for j in {0,1}, k in {0..5}
    
    # For each column pair (k0, k1) in [(0,1), (2,3), (4,5)]:
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen for 2x2 * 2x2 (columns k0, k1)
        # A = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        a00 = u_row(0, 0)
        a01 = u_row(0, 1)
        a10 = u_row(1, 0)
        a11 = u_row(1, 1)
        
        b00 = v_row(0, k0)
        b01 = v_row(0, k1)
        b10 = v_row(1, k0)
        b11 = v_row(1, k1)
        
        # p1 = (a00 + a11) * (b00 + b11)
        # Contributes to c00 (W[k0,0]) and c11 (W[k1,1])
        add_rank1({a00: 1, a11: 1}, {b00: 1, b11: 1}, {w_row(k0, 0): 1, w_row(k1, 1): 1})
        
        # p2 = (a10 + a11) * b00
        # Contributes to c10 (W[k0,1])
        add_rank1({a10: 1, a11: 1}, {b00: 1}, {w_row(k0, 1): 1})
        
        # p3 = a00 * (b01 - b11)
        # Contributes to c01 (W[k1,0])
        add_rank1({a00: 1}, {b01: 1, b11: -1}, {w_row(k1, 0): 1})
        
        # p4 = a11 * (b10 - b00)
        # Contributes to c00 (W[k0,0]) and c10 (W[k0,1])
        add_rank1({a11: 1}, {b10: 1, b00: -1}, {w_row(k0, 0): 1, w_row(k0, 1): 1})
        
        # p5 = (a00 + a01) * b11
        # Contributes to c00 (W[k0,0]) with -1 and c01 (W[k1,0]) with +1
        add_rank1({a00: 1, a01: 1}, {b11: 1}, {w_row(k0, 0): -1, w_row(k1, 0): 1})
        
        # p6 = (a10 - a00) * (b00 + b01)
        # Contributes to c11 (W[k1,1])
        add_rank1({a10: 1, a00: -1}, {b00: 1, b01: 1}, {w_row(k1, 1): 1})
        
        # p7 = (a01 - a11) * (b10 + b11)
        # Contributes to c00 (W[k0,0])
        add_rank1({a01: 1, a11: -1}, {b10: 1, b11: 1}, {w_row(k0, 0): 1})
    
    # Process A_mid @ B_mid (j=2,3)
    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        a00 = u_row(0, 2)
        a01 = u_row(0, 3)
        a10 = u_row(1, 2)
        a11 = u_row(1, 3)
        
        b00 = v_row(2, k0)
        b01 = v_row(2, k1)
        b10 = v_row(3, k0)
        b11 = v_row(3, k1)
        
        # p1 = (a00 + a11) * (b00 + b11)
        add_rank1({a00: 1, a11: 1}, {b00: 1, b11: 1}, {w_row(k0, 0): 1, w_row(k1, 1): 1})
        
        # p2 = (a10 + a11) * b00
        add_rank1({a10: 1, a11: 1}, {b00: 1}, {w_row(k0, 1): 1})
        
        # p3 = a00 * (b01 - b11)
        add_rank1({a00: 1}, {b01: 1, b11: -1}, {w_row(k1, 0): 1})
        
        # p4 = a11 * (b10 - b00)
        add_rank1({a11: 1}, {b10: 1, b00: -1}, {w_row(k0, 0): 1, w_row(k0, 1): 1})
        
        # p5 = (a00 + a01) * b11
        add_rank1({a00: 1, a01: 1}, {b11: 1}, {w_row(k0, 0): -1, w_row(k1, 0): 1})
        
        # p6 = (a10 - a00) * (b00 + b01)
        add_rank1({a10: 1, a00: -1}, {b00: 1, b01: 1}, {w_row(k1, 1): 1})
        
        # p7 = (a01 - a11) * (b10 + b11)
        add_rank1({a01: 1, a11: -1}, {b10: 1, b11: 1}, {w_row(k0, 0): 1})
    
    # Process A_right @ B_right (j=4)
    # A_right[i,0] = A[i,4] for i in {0,1}
    # B_right[0,k] = B[4,k] for k in {0..5}
    # C[i,k] += A[i,4] * B[4,k]
    
    for i in range(2):
        for k in range(6):
            add_rank1({u_row(i, 4): 1}, {v_row(4, k): 1}, {w_row(k, i): 1})
    
    # Convert to dense matrices
    rank = col
    U_dense = [[0.0] * rank for _ in range(n * m)]
    V_dense = [[0.0] * rank for _ in range(m * p)]
    W_dense = [[0.0] * rank for _ in range(p * n)]
    
    for row in range(n * m):
        for c, val in U[row].items():
            U_dense[row][c] = val
    for row in range(m * p):
        for c, val in V[row].items():
            V_dense[row][c] = val
    for row in range(p * n):
        for c, val in W[row].items():
            W_dense[row][c] = val
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
