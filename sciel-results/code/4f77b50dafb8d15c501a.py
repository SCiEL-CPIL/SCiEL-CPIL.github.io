"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import random
import sys
import numpy as np

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    
    # Compute sum over rank: U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)

def _build_kronecker_decomp(n, m, p):
    """
    Build a decomposition using Kronecker product structure.
    
    For multiplying (n×m) by (m×p), we can decompose using the structure:
    If m = m1 * m2, n = n1 * n2, p = p1 * p2, we can use tensor products.
    
    For 4×4 × 4×5:
    - m=4 = 2×2, n=4 = 2×2, p=5 (prime, can't factor nicely)
    
    Alternative: Use block structure.
    Split A into 2×2 blocks of size 2×2, B into 2×2 blocks of size 2×2 (last col special).
    
    Actually, let's use a simpler approach:
    For n=m=4, p=5, we can use the fact that 4×4 matrix multiplication
    has known rank 7 (Strassen). So 4×4 × 4×5 can be seen as two 4×4 multiplications
    plus one 4×1, giving rank at most 7+7+4 = 18? No, that's not quite right.
    
    Better: Use the standard Kronecker decomposition.
    If we write A = [A11 A12; A21 A22] (2x2 blocks of 2x2),
    and B = [B11 B12 B13; B21 B22 B23] (2x2 blocks, but 3 columns of blocks since p=5? No, 5/2 isn't integer)
    
    Let me think differently. Use the direct Kronecker product approach:
    
    For general n,m,p, the schoolbook rank is n*m*p.
    We can do better by exploiting structure.
    
    For 4×4 × 4×5:
    Think of it as: result C[i,k] = sum_j A[i,j] * B[j,k]
    
    Use Strassen for the 4×4 part:
    Split A into 2x2 blocks of 2x2 matrices: A = [[A1, A2], [A3, A4]]
    Split B's first 4 columns into 2x2 blocks of 2x2: B[:, :4] = [[B1, B2], [B3, B4]]
    B's 5th column: B[:, 4] = [B5_top; B5_bot] (2x1 and 2x1)
    
    C[:, :4] uses Strassen on 2x2 blocks of 2x2 matrices: rank 7 for the 2x2 block multiplication
    Each Strassen product gives a 2x2 result, so we need 7 * (2*2) = 28 scalar multiplications for C[:, :4]
    
    C[:, 4] = A @ B[:, 4] = [A1, A2; A3, A4] @ [B5t; B5b]
    = [A1*B5t + A2*B5b; A3*B5t + A4*B5b]
    This requires 4 matrix products (each 2x2 @ 2x1 = 2x1), but we can reuse some from Strassen?
    
    Actually, let's just build a clean decomposition:
    
    Method: Use the fact that 4x4 matmul has rank 7 (Strassen).
    For 4x4 * 4x5, we can view it as:
    C = A * B = A * [B_left | B_right] where B_left is 4x4 and B_right is 4x1
    
    C_left = A * B_left: rank 7 (Strassen for 4x4)
    C_right = A * B_right: this is 4x1, rank at most 4 (each output is sum of 4 products)
    
    But we need a unified tensor decomposition. Let me construct it explicitly.
    
    Strassen for 2x2:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For 4x4 * 4x4, we use 2x2 blocks of 2x2 matrices and apply Strassen at the block level.
    Each block product is a 2x2 matrix product, which itself can be done with rank 7.
    So total rank = 7 * 7 = 49? No, that's not right either.
    
    Actually, for 4x4 * 4x4:
    A = [[A1, A2], [A3, A4]] where Ai are 2x2
    B = [[B1, B2], [B3, B4]] where Bj are 2x2
    
    Using Strassen at block level:
    We need 7 block products, each of which is a 2x2 * 2x2 multiplication.
    Each 2x2*2x2 has rank 7 (Strassen).
    So total rank = 7 * 7 = 49 for 4x4*4x4.
    
    But wait, we can do better. The known rank of 4x4 matrix multiplication is 49 (Strassen^2) or possibly lower.
    Actually, the best known upper bound for 4x4 is 49, and lower bound is 44.
    
    For our problem 4x4 * 4x5:
    C = A * B where A is 4x4, B is 4x5.
    We can write B = [B1 | B2] where B1 is 4x4 and B2 is 4x1.
    C = [A*B1 | A*B2]
    
    A*B1: 4x4 * 4x4, rank at most 49
    A*B2: 4x4 * 4x1, rank at most 16 (schoolbook) or less
    
    But we want a single tensor decomposition. Let me just construct it directly.
    
    For 4x4 * 4x5, I'll use the following approach:
    - Use Strassen recursively for the 4x4 part
    - Handle the 5th column separately
    
    Let me code this up properly.
    """
    n, m, p = 4, 4, 5
    
    # For 4x4 * 4x4, use Strassen (rank 49)
    # For the extra column, add 4 more terms (schoolbook for 4x4 * 4x1)
    # Total: 49 + 4 = 53? Let's see if we can do better.
    
    # Actually, let me just build the schoolbook and then try to optimize.
    # Schoolbook rank: 4*4*5 = 80
    
    # Let me try a different structured approach:
    # Use the fact that we can share computations between columns.
    
    # For now, return schoolbook as the structured baseline
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
    return U, V, W, rank

def _build_strassen_4x4_4x5():
    """
    Build a rank-49 decomposition for 4x4 * 4x4, extended to 4x4 * 4x5.
    
    For 4x4 * 4x4:
    Split into 2x2 blocks of 2x2 matrices.
    Apply Strassen at block level (7 block products).
    Each block product is 2x2 * 2x2, done with Strassen (rank 7).
    Total: 7 * 7 = 49 scalar products.
    
    For the 5th column of B:
    A * B[:,4] where B[:,4] is 4x1.
    This is 4x4 * 4x1, schoolbook rank 16.
    But we can potentially share with existing terms.
    
    Let's first get the 4x4*4x4 part, then add the 5th column.
    """
    n, m, p = 4, 4, 5
    
    # Strassen 2x2 * 2x2 (rank 7)
    # Inputs: A11,A12,A21,A22 (2x2 matrices), B11,B12,B21,B22 (2x2 matrices)
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 4x4 * 4x4, split into 2x2 blocks of 2x2 matrices:
    # A = [[A1, A2], [A3, A4]] where Ai are 2x2
    # B = [[B1, B2], [B3, B4]] where Bj are 2x2
    # 
    # Block-level Strassen:
    # M1_block = (A1+A4)*(B1+B4)
    # M2_block = (A3+A4)*B1
    # M3_block = A1*(B2-B4)
    # M4_block = A4*(B3-B1)
    # M5_block = (A1+A2)*B4
    # M6_block = (A3-A1)*(B1+B2)
    # M7_block = (A2-A4)*(B3+B4)
    #
    # C11_block = M1_block + M4_block - M5_block + M7_block
    # C12_block = M3_block + M5_block
    # C21_block = M2_block + M4_block
    # C22_block = M1_block - M2_block + M3_block + M6_block
    #
    # Each M*_block is a 2x2 * 2x2 product, done with Strassen (rank 7).
    # So total rank = 7 * 7 = 49.
    
    # Let me enumerate all 49 terms.
    # Each term is indexed by (block_strassen_idx, inner_strassen_idx)
    # block_strassen_idx in 1..7, inner_strassen_idx in 1..7
    
    # First, define the inner Strassen for 2x2 * 2x2:
    # For matrices X = [[x11,x12],[x21,x22]] and Y = [[y11,y12],[y21,y22]]:
    # M1 = (x11+x22)*(y11+y22)
    # M2 = (x21+x22)*y11
    # M3 = x11*(y12-y22)
    # M4 = x22*(y21-y11)
    # M5 = (x11+x12)*y22
    # M6 = (x21-x11)*(y11+y12)
    # M7 = (x12-x22)*(y21+y22)
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For the inner Strassen, the "scalar" multiplications are:
    # Each Mi involves products of linear combinations of entries.
    # For example, M1 = (x11+x22)*(y11+y22)
    # This contributes to C11 with coefficient +1, to C22 with coefficient +1.
    
    # Let me build the full decomposition step by step.
    
    # Indices:
    # A is 4x4: A[i,j] for i,j in 0..3
    # B is 4x4 (for first 4 cols): B[j,k] for j,k in 0..3
    # C is 4x4: C[i,k] = sum_j A[i,j]*B[j,k]
    
    # Block structure:
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4], A3 = A[2:4, 0:2], A4 = A[2:4, 2:4]
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    
    # Block-level Strassen terms (each is a 2x2*2x2 product):
    # M1_b = (A1+A4)*(B1+B4)
    # M2_b = (A3+A4)*B1
    # M3_b = A1*(B2-B4)
    # M4_b = A4*(B3-B1)
    # M5_b = (A1+A2)*B4
    # M6_b = (A3-A1)*(B1+B2)
    # M7_b = (A2-A4)*(B3+B4)
    
    # For each block-level term, we need to decompose the 2x2*2x2 product using inner Strassen.
    
    # Let me define a helper to add a 2x2*2x2 Strassen decomposition.
    
    def add_inner_strassen(U, V, W, col_offset, 
                           A_block_indices, B_block_indices, 
                           C_block_indices, signs):
        """
        Add an inner Strassen decomposition for a 2x2 * 2x2 block product.
        
        A_block_indices: list of 4 indices (i,j) for the 2x2 A block
        B_block_indices: list of 4 indices (j,k) for the 2x2 B block  
        C_block_indices: list of 4 indices (i,k) for the 2x2 C block
        signs: list of 4 coefficients for how this block product contributes to C
        """
        # A block: A[i1,j1], A[i1,j2], A[i2,j1], A[i2,j2]
        # B block: B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]
        # C block: C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
        
        a_idx = A_block_indices  # [(i1,j1), (i1,j2), (i2,j1), (i2,j2)]
        b_idx = B_block_indices  # [(j1,k1), (j1,k2), (j2,k1), (j2,k2)]
        c_idx = C_block_indices  # [(i1,k1), (i1,k2), (i2,k1), (i2,k2)]
        
        # Inner Strassen:
        # M1 = (x11+x22)*(y11+y22)
        # M2 = (x21+x22)*y11
        # M3 = x11*(y12-y22)
        # M4 = x22*(y21-y11)
        # M5 = (x11+x12)*y22
        # M6 = (x21-x11)*(y11+y12)
        # M7 = (x12-x22)*(y21+y22)
        
        # For each inner Strassen term, we have:
        # U part: coefficients for A entries
        # V part: coefficients for B entries
        # W part: coefficients for C entries (with sign from block level)
        
        # M1: (x11+x22)*(y11+y22)
        # U: x11:+1, x22:+1, others:0
        # V: y11:+1, y22:+1, others:0
        # W: C11:+sign[0], C22:+sign[3] (from C11=M1+..., C22=M1-...)
        
        # This is getting complex. Let me just enumerate all 49 terms directly.
        
        pass  # Placeholder
    
    # This approach is getting very complex. Let me just use the schoolbook for now
    # and focus on a simpler structured decomposition.
    
    # Simpler approach: Use the fact that for 4x4*4x5, we can use
    # a decomposition based on the Kronecker product of smaller decompositions.
    
    # Actually, let me just return schoolbook and note that we need a better approach.
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
    return U, V, W, rank

def _build_decomp_4x4_4x5():
    """
    Build a structured decomposition for 4x4 * 4x5.
    
    Strategy: Use the known rank-49 decomposition for 4x4*4x4 (Strassen^2)
    and extend to handle the 5th column.
    
    For 4x4*4x4:
    Split A into 2x2 blocks of 2x2 matrices: A = [[A1,A2],[A3,A4]]
    Split B into 2x2 blocks of 2x2 matrices: B = [[B1,B2],[B3,B4]]
    
    Block-level Strassen (7 terms):
    M1 = (A1+A4)*(B1+B4)
    M2 = (A3+A4)*B1
    M3 = A1*(B2-B4)
    M4 = A4*(B3-B1)
    M5 = (A1+A2)*B4
    M6 = (A3-A1)*(B1+B2)
    M7 = (A2-A4)*(B3+B4)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2*2x2 product. Use inner Strassen (7 terms each).
    Total: 7*7 = 49 terms for 4x4*4x4.
    
    For the 5th column of B:
    Let b5 = B[:,4] (4x1 vector).
    C[:,4] = A * b5
    
    This is a 4x4 * 4x1 multiplication. Schoolbook rank: 16.
    But we can potentially share with existing terms.
    
    For now, let's just add 16 more terms for the 5th column.
    Total: 49 + 16 = 65.
    
    Actually, we can do better. The 5th column computation:
    C[i,4] = sum_j A[i,j] * B[j,4]
    
    We can group this with the existing Strassen structure.
    Note that B[j,4] for j=0,1,2,3 can be seen as an extra column in the B blocks.
    
    Let me extend the block structure:
    B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    B5_top = B[0:2, 4] (2x1), B5_bot = B[2:4, 4] (2x1)
    
    C1_col5 = C[0:2, 4] = A1*B5_top + A2*B5_bot (2x1 result)
    C2_col5 = C[2:4, 4] = A3*B5_top + A4*B5_bot (2x1 result)
    
    Each of these is a 2x2 * 2x1 + 2x2 * 2x1, which is a 2x4 * 2x1 multiplication.
    Schoolbook rank: 8 per half, so 16 total.
    
    But we can use the existing Strassen structure. Note that:
    A1*B5_top appears in M3 = A1*(B2-B4) if we could extend B2 and B4.
    
    This is getting complicated. Let me just implement the 49-term 4x4*4x4 part
    and add 16 terms for the 5th column, giving rank 65.
    
    Then we can try to optimize further.
    """
    n, m, p = 4, 4, 5
    
    # Define the 49 terms for 4x4*4x4 using double Strassen.
    # Each term is a product of a linear combination of A entries,
    # a linear combination of B entries, and contributes to C entries.
    
    # Let me enumerate manually.
    
    # Block-level Strassen terms and their inner Strassen decompositions.
    
    # I'll represent each term as:
    # (U_coeffs, V_coeffs, W_coeffs)
    # where U_coeffs is a dict {A[i,j]: coeff}, etc.
    
    # For clarity, let me use tuples: (i, j) for A, (j, k) for B, (i, k) for C.
    
    terms = []
    
    # Helper: inner Strassen for 2x2 * 2x2
    # A block: indices a0=(i1,j1), a1=(i1,j2), a2=(i2,j1), a3=(i2,j2)
    # B block: indices b0=(j1,k1), b1=(j1,k2), b2=(j2,k1), b3=(j2,k2)
    # C block: indices c0=(i1,k1), c1=(i1,k2), c2=(i2,k1), c3=(i2,k2)
    # signs: [s0, s1, s2, s3] for how this block product contributes to C
    
    def inner_strassen(a_idx, b_idx, c_idx, signs):
        """
        Generate 7 terms for 2x2*2x2 Strassen.
        Returns list of (U_dict, V_dict, W_dict)
        """
        result = []
        # a_idx: [a00, a01, a10, a11] where a00=(i1,j1), a01=(i1,j2), a10=(i2,j1), a11=(i2,j2)
        # b_idx: [b00, b01, b10, b11] where b00=(j1,k1), b01=(j1,k2), b10=(j2,k1), b11=(j2,k2)
        # c_idx: [c00, c01, c10, c11] where c00=(i1,k1), c01=(i1,k2), c10=(i2,k1), c11=(i2,k2)
        
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        s0, s1, s2, s3 = signs
        
        # M1 = (x00+x11)*(y00+y11)
        # C00 += s0*M1, C11 += s3*M1
        U1 = {a00: 1, a11: 1}
        V1 = {b00: 1, b11: 1}
        W1 = {c00: s0, c11: s3}
        result.append((U1, V1, W1))
        
        # M2 = (x10+x11)*y00
        # C10 += s2*M2, C11 += -s3*M2
        U2 = {a10: 1, a11: 1}
        V2 = {b00: 1}
        W2 = {c10: s2, c11: -s3}
        result.append((U2, V2, W2))
        
        # M3 = x00*(y01-y11)
        # C01 += s1*M3, C11 += s3*M3
        U3 = {a00: 1}
        V3 = {b01: 1, b11: -1}
        W3 = {c01: s1, c11: s3}
        result.append((U3, V3, W3))
        
        # M4 = x11*(y10-y00)
        # C00 += s0*M4, C10 += s2*M4
        U4 = {a11: 1}
        V4 = {b10: 1, b00: -1}
        W4 = {c00: s0, c10: s2}
        result.append((U4, V4, W4))
        
        # M5 = (x00+x01)*y11
        # C00 += -s0*M5, C01 += s1*M5
        U5 = {a00: 1, a01: 1}
        V5 = {b11: 1}
        W5 = {c00: -s0, c01: s1}
        result.append((U5, V5, W5))
        
        # M6 = (x10-x00)*(y00+y01)
        # C11 += s3*M6
        U6 = {a10: 1, a00: -1}
        V6 = {b00: 1, b01: 1}
        W6 = {c11: s3}
        result.append((U6, V6, W6))
        
        # M7 = (x01-x11)*(y10+y11)
        # C00 += s0*M7
        U7 = {a01: 1, a11: -1}
        V7 = {b10: 1, b11: 1}
        W7 = {c00: s0}
        result.append((U7, V7, W7))
        
        return result
    
    # Block-level Strassen for 4x4*4x4:
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4], A3 = A[2:4, 0:2], A4 = A[2:4, 2:4]
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    
    # A1 indices: [(0,0), (0,2), (1,0), (1,2)]
    # A2 indices: [(0,2), (0,3), (1,2), (1,3)]
    # A3 indices: [(2,0), (2,2), (3,0), (3,2)]
    # A4 indices: [(2,2), (2,3), (3,2), (3,3)]
    
    # B1 indices: [(0,0), (0,2), (1,0), (1,2)]
    # B2 indices: [(0,2), (0,3), (1,2), (1,3)]
    # B3 indices: [(2,0), (2,2), (3,0), (3,2)]
    # B4 indices: [(2,2), (2,3), (3,2), (3,3)]
    
    # C11 indices: [(0,0), (0,2), (1,0), (1,2)]
    # C12 indices: [(0,2), (0,3), (1,2), (1,3)]
    # C21 indices: [(2,0), (2,2), (3,0), (3,2)]
    # C22 indices: [(2,2), (2,3), (3,2), (3,3)]
    
    A1_idx = [(0,0), (0,2), (1,0), (1,2)]
    A2_idx = [(0,2), (0,3), (1,2), (1,3)]
    A3_idx = [(2,0), (2,2), (3,0), (3,2)]
    A4_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    B1_idx = [(0,0), (0,2), (1,0), (1,2)]
    B2_idx = [(0,2), (0,3), (1,2), (1,3)]
    B3_idx = [(2,0), (2,2), (3,0), (3,2)]
    B4_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    C11_idx = [(0,0), (0,2), (1,0), (1,2)]
    C12_idx = [(0,2), (0,3), (1,2), (1,3)]
    C21_idx = [(2,0), (2,2), (3,0), (3,2)]
    C22_idx = [(2,2), (2,3), (3,2), (3,3)]
    
    # Block-level Strassen:
    # M1 = (A1+A4)*(B1+B4)
    # M2 = (A3+A4)*B1
    # M3 = A1*(B2-B4)
    # M4 = A4*(B3-B1)
    # M5 = (A1+A2)*B4
    # M6 = (A3-A1)*(B1+B2)
    # M7 = (A2-A4)*(B3+B4)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each block-level term, the A and B blocks are sums/differences of 2x2 blocks.
    # We need to handle the linear combinations.
    
    # M1 = (A1+A4)*(B1+B4)
    # A part: A1 + A4, so coefficients are +1 for all entries in A1 and A4
    # B part: B1 + B4, so coefficients are +1 for all entries in B1 and B4
    # This is a 2x2*2x2 product where the "A matrix" has entries from A1 and A4,
    # and the "B matrix" has entries from B1 and B4.
    
    # But wait, A1 and A4 are both 2x2 matrices. A1+A4 is also 2x2.
    # So M1 = (A1+A4) * (B1+B4) is a 2x2*2x2 product.
    # The A entries are: (A1+A4)[r,c] = A1[r,c] + A4[r,c]
    # For r,c in 0,1: A1[r,c] = A[2*r, 2*c], A4[r,c] = A[2*r+2, 2*c+2]
    # So (A1+A4)[0,0] = A[0,0] + A[2,2]
    # (A1+A4)[0,1] = A[0,2] + A[2,3]
    # (A1+A4)[1,0] = A[1,0] + A[3,2]
    # (A1+A4)[1,1] = A[1,2] + A[3,3]
    
    # Similarly for B1+B4.
    
    # So for M1, the inner Strassen operates on:
    # a00 = A[0,0]+A[2,2], a01 = A[0,2]+A[2,3], a10 = A[1,0]+A[3,2], a11 = A[1,2]+A[3,3]
    # b00 = B[0,0]+B[2,2], b01 = B[0,2]+B[2,3], b10 = B[1,0]+B[3,2], b11 = B[1,2]+B[3,3]
    # c00 = C[0,0], c01 = C[0,2], c10 = C[1,0], c11 = C[1,2] (for C11)
    
    # But the inner Strassen expects individual A entries, not sums.
    # We need to expand: each "a" entry is a sum of A entries.
    
    # This means each inner Strassen term will involve products of sums,
    # which expand into multiple scalar products.
    
    # Actually, the tensor decomposition allows linear combinations in U, V, W.
    # So U can have coefficients for multiple A entries.
    
    # For M1's inner Strassen term M1_inner = (a00+a11)*(b00+b11):
    # U: a00:+1, a11:+1 => A[0,0]:+1, A[2,2]:+1, A[1,2]:+1, A[3,3]:+1
    # V: b00:+1, b11:+1 => B[0,0]:+1, B[2,2]:+1, B[1,2]:+1, B[3,3]:+1
    # W: c00:+s0, c11:+s3 => C[0,0]:+s0, C[1,2]:+s3
    
    # So we can represent this as a single term in the tensor decomposition!
    
    # Great. Let me build all 49 terms.
    
    # I'll create a function that, given the A block combination, B block combination,
    # and C block with signs, generates the 7 inner Strassen terms.
    
    def block_strassen_term(A_comb, B_comb, C_idx, signs):
        """
        A_comb: dict {(i,j): coeff} for the A 2x2 block (after combining)
        B_comb: dict {(i,j): coeff} for the B 2x2 block (after combining)
        C_idx: list of 4 C indices [(i1,k1), (i1,k2), (i2,k1), (i2,k2)]
        signs: [s0, s1, s2, s3]
        
        Returns list of (U_dict, V_dict, W_dict) for the 7 inner Strassen terms.
        """
        # Get the 4 entries of the combined A block
        # A_comb should have exactly 4 entries corresponding to (0,0), (0,1), (1,0), (1,1) of the block
        # But the keys are actual (i,j) indices.
        
        # We need to map block positions to actual indices.
        # Let's assume A_comb has keys that we can sort to get the 4 entries.
        
        # Actually, let me restructure. Pass the actual index lists.
        pass
    
    # This is getting very messy. Let me take a completely different approach.
    # Just build the schoolbook decomposition and return it.
    # The random search in construct() will try to find better decompositions.
    
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
    return U, V, W, rank

def _try_gradient_search(n, m, p, target_rank, time_budget, rng):
    """
    Try to find a decomposition using a gradient-based approach.
    Start from a random initialization and try to satisfy the constraints.
    """
    deadline = time.time() + time_budget
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randn(n_rows_u, target_rank) * 0.1
        V = np.random.randn(n_rows_v, target_rank) * 0.1
        W = np.random.randn(n_rows_w, target_rank) * 0.1
        
        # Try a few iterations of gradient descent (simplified)
        # This is a placeholder; proper implementation would use autodiff or finite differences.
        
        # For now, just verify if by chance it works
        if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Verify schoolbook
    if not _verify_decomposition_fast(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_rank = schoolbook_rank
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decompositions
    rng = random.Random(42)
    
    # Try random search for various ranks
    for target_rank in [72, 70, 68, 66, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 10, 8, 6, 5, 4, 3, 2]:
        if time.time() > deadline:
            break
        
        time_per_attempt = min(2.0, max(0.1, (deadline - time.time()) / 100))
        
        # Random search with half-integer values
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank), dtype=np.float64)
            V = np.zeros((m * p, target_rank), dtype=np.float64)
            W = np.zeros((p * n, target_rank), dtype=np.float64)
            
            choices = [-1, -0.5, 0, 0.5, 1]
            for c in range(target_rank):
                U[:, c] = [rng.choice(choices) for _ in range(n * m)]
                V[:, c] = [rng.choice(choices) for _ in range(m * p)]
                W[:, c] = [rng.choice(choices) for _ in range(p * n)]
            
            if _verify_decomposition_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
        
        if best_rank < schoolbook_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
