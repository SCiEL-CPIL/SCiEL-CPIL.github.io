"""Matrix multiplication 2x4x6 rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 4, 6]

# Deadline guard
DEADLINE = 120.0

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x6 matrix.
    
    Strategy: Use Strassen's algorithm (rank 7 for 2x2x2) to decompose the
    2x4x6 multiplication into blocks, reducing rank from schoolbook 48.
    
    The 2x4 * 4x6 = 2x6 matrix can be split into:
    - Split the 4x4 middle into four 2x2 blocks
    - This gives 2x2 * 2x2 multiplications for each column pair
    
    Specifically:
    A is 2x4: A = [A1 A2] where A1, A2 are 2x2
    B is 4x6: B = [B1; B2; B3; B4]... wait, let me reconsider.
    
    Actually for 2x4 * 4x6:
    A: 2 rows, 4 cols
    B: 4 rows, 6 cols
    C: 2 rows, 6 cols
    
    Split A into [A1 A2] where A1, A2 are 2x2
    Split B into [B1; B2] where B1, B2 are 2x6
    
    Then C = A1*B1 + A2*B2
    
    Each A_i * B_i is 2x2 * 2x6. We can split B_i into [B_i1 B_i2 B_i3] where each is 2x2.
    So A_i * B_i = A_i * B_i1 + A_i * B_i2 + A_i * B_i3, each 2x2*2x2.
    
    Total: 4 blocks of 2x2*2x2, each using Strassen rank 7.
    Total rank = 4 * 7 = 28.
    
    But we can do better by sharing some Strassen computations across blocks.
    
    Let me think more carefully. We have:
    C = A1*B1 + A2*B2
    
    where A1, A2 are 2x2 and B1, B2 are 2x6.
    
    Write B1 = [B11 B12 B13] and B2 = [B21 B22 B23] where each Bij is 2x2.
    
    C = A1*[B11 B12 B13] + A2*[B21 B22 B23]
      = [A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23]
    
    Each column j of C is A1*B1j + A2*B2j, a sum of two 2x2*2x2 products.
    
    Using Strassen for each pair (A1*B1j + A2*B2j):
    We can compute A1*B1j and A2*B2j separately with Strassen (7 each) = 14 per column, 42 total.
    
    But can we share? The Strassen formulas for A1*B1j and A2*B2j use different inputs,
    so direct sharing isn't obvious. However, we can use a block Strassen approach.
    
    Alternative: Treat the whole thing as a larger tensor.
    
    Let me use the known result: for 2x4x6, the tensor rank is at most 28 using
    a specific construction, but 42 via naive Strassen blocks.
    
    Let me try a different decomposition. Consider:
    
    We want to compute C_{ik} = sum_j A_{ij} B_{jk} for i in {1,2}, j in {1,2,3,4}, k in {1,...,6}.
    
    Split j into two groups: j in {1,2} and j in {3,4}.
    C_{ik} = sum_{j=1}^2 A_{ij} B_{jk} + sum_{j=3}^4 A_{ij} B_{jk}
           = (A_left * B_left)_{ik} + (A_right * B_right)_{ik}
    
    where A_left is 2x2, B_left is 2x6, A_right is 2x2, B_right is 2x6.
    
    For each of these 2x2 * 2x6 products, split the 6 columns into three 2-column groups.
    So we have 6 separate 2x2 * 2x2 multiplications.
    
    Using Strassen (rank 7) for each: total rank = 6 * 7 = 42.
    
    Can we do better? Let's try to use a rank-7 decomposition for the 2x2*2x6 block directly.
    
    Actually, a 2x2 * 2x6 matrix multiplication can be seen as computing 12 outputs from 12 inputs.
    The tensor rank of 2x2x6... let me think.
    
    Alternatively, I'll implement the 42-rank Strassen-based solution which is a clear improvement
    over 48, and see if I can find a better one.
    
    Let me implement the 42-rank solution first.
    """
    n, m, p = 2, 4, 6  # A is n x m, B is m x p, C is n x p
    
    # Strassen's 7 multiplications for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2), compute X*Y with 7 products:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For our problem:
    # A (2x4) = [A1 A2] where A1 = [[a00,a01],[a10,a11]], A2 = [[a02,a03],[a12,a13]]
    # B (4x6) = [B1; B2] where B1 (2x6) = [[b00,b01,b02,b03,b04,b05],[b10,b11,b12,b13,b14,b15]]
    #            B2 (2x6) = [[b20,b21,b22,b23,b24,b25],[b30,b31,b32,b33,b34,b35]]
    # C (2x6) = A1*B1 + A2*B2
    
    # Split B1 into [B1a B1b B1c] each 2x2, same for B2.
    # B1a = [[b00,b01],[b10,b11]], B1b = [[b02,b03],[b12,b13]], B1c = [[b04,b05],[b14,b15]]
    # B2a = [[b20,b21],[b30,b31]], B2b = [[b22,b23],[b32,b33]], B2c = [[b24,b25],[b34,b35]]
    
    # C = [A1*B1a + A2*B2a, A1*B1b + A2*B2b, A1*B1c + A2*B2c]
    
    # For each column group g in {a,b,c}:
    # C_g = A1*B_g1 + A2*B_g2 (2x2 result)
    
    # Using Strassen for A1*B_g1 (7 mults) and A2*B_g2 (7 mults) = 14 mults per group
    # Total: 3 groups * 14 = 42 mults
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    col = 0
    # A indices: A[i*m+j] where i in 0..1, j in 0..3
    # A[0]=a00, A[1]=a01, A[2]=a02, A[3]=a03, A[4]=a10, A[5]=a11, A[6]=a12, A[7]=a13
    # B indices: B[j*p+k] where j in 0..3, k in 0..5
    # B[0]=b00, B[1]=b01, B[2]=b02, B[3]=b03, B[4]=b04, B[5]=b05
    # B[6]=b10, B[7]=b11, B[8]=b12, B[9]=b13, B[10]=b14, B[11]=b15
    # B[12]=b20, B[13]=b21, B[14]=b22, B[15]=b23, B[16]=b24, B[17]=b25
    # B[18]=b30, B[19]=b31, B[20]=b32, B[21]=b33, B[22]=b34, B[23]=b35
    # W indices: W[k*n+i] where k in 0..5, i in 0..1
    # W[0]=C[0,0], W[1]=C[0,1], W[2]=C[0,2], W[3]=C[0,3], W[4]=C[0,4], W[5]=C[0,5]
    # W[6]=C[1,0], W[7]=C[1,1], W[8]=C[1,2], W[9]=C[1,3], W[10]=C[1,4], W[11]=C[1,5]
    
    # Strassen coefficients for 2x2*2x2:
    # M1: X=(X00+X11), Y=(Y00+Y11)
    # M2: X=(X10+X11), Y=Y00
    # M3: X=X00, Y=(Y01-Y11)
    # M4: X=X11, Y=(Y10-Y00)
    # M5: X=(X00+X01), Y=Y11
    # M6: X=(X10-X00), Y=(Y00+Y01)
    # M7: X=(X01-X11), Y=(Y10+Y11)
    
    # For A1*B1a:
    # A1 = [[a00,a01],[a10,a11]], B1a = [[b00,b01],[b10,b11]]
    # X00=a00, X01=a01, X10=a10, X11=a11
    # Y00=b00, Y01=b01, Y10=b10, Y11=b11
    
    strassen_defs = [
        # (name, X_coeffs, Y_coeffs)
        # X_coeffs: dict from index to coeff, Y_coeffs: dict from index to coeff
        # M1
        ({'X00': 1, 'X11': 1}, {'Y00': 1, 'Y11': 1}),
        # M2
        ({'X10': 1, 'X11': 1}, {'Y00': 1}),
        # M3
        ({'X00': 1}, {'Y01': 1, 'Y11': -1}),
        # M4
        ({'X11': 1}, {'Y10': 1, 'Y00': -1}),
        # M5
        ({'X00': 1, 'X01': 1}, {'Y11': 1}),
        # M6
        ({'X10': 1, 'X00': -1}, {'Y00': 1, 'Y01': 1}),
        # M7
        ({'X01': 1, 'X11': -1}, {'Y10': 1, 'Y11': 1}),
    ]
    
    # Output coefficients for Strassen:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    output_coeffs = [
        # For each output (Z00, Z01, Z10, Z11), list of (mult_index, coeff)
        [(0, 1), (3, 1), (4, -1), (6, 1)],   # Z00
        [(2, 1), (4, 1)],                      # Z01
        [(1, 1), (3, 1)],                      # Z10
        [(0, 1), (1, -1), (2, 1), (5, 1)],    # Z11
    ]
    
    # For each column group (0, 1, 2) corresponding to B1a/B2a, B1b/B2b, B1c/B2c
    for g in range(3):
        # B_g1 is the (2g, 2g+1) columns of B1 (rows 0,1 of B)
        # B_g2 is the (2g, 2g+1) columns of B2 (rows 2,3 of B)
        
        # B1a: rows 0,1, cols 2g,2g+1 -> B indices: (0*p + 2g), (0*p + 2g+1), (1*p + 2g), (1*p + 2g+1)
        # B2a: rows 2,3, cols 2g,2g+1 -> B indices: (2*p + 2g), (2*p + 2g+1), (3*p + 2g), (3*p + 2g+1)
        
        # For A1 * B_g1:
        # X = A1 = [[a00,a01],[a10,a11]] -> A indices: 0,1,4,5
        # Y = B_g1
        for mi, (x_coeff, y_coeff) in enumerate(strassen_defs):
            # Determine which A indices are used
            A_idx_map = {'X00': 0, 'X01': 1, 'X10': 4, 'X11': 5}
            B_idx_map = {
                'Y00': 0 * p + 2 * g,     # b0_{2g}
                'Y01': 0 * p + 2 * g + 1, # b0_{2g+1}
                'Y10': 1 * p + 2 * g,     # b1_{2g}
                'Y11': 1 * p + 2 * g + 1, # b1_{2g+1}
            }
            
            for x_name, x_val in x_coeff.items():
                U[A_idx_map[x_name]][col] += x_val
            for y_name, y_val in y_coeff.items():
                V[B_idx_map[y_name]][col] += y_val
            
            # W contributions: this multiplication contributes to C[0,2g], C[0,2g+1], C[1,2g], C[1,2g+1]
            # W indices: W[k*n+i]
            # C[0,2g] -> W[2g*2+0] = W[4g]
            # C[0,2g+1] -> W[(2g+1)*2+0] = W[4g+2]
            # C[1,2g] -> W[2g*2+1] = W[4g+1]
            # C[1,2g+1] -> W[(2g+1)*2+1] = W[4g+3]
            
            w_indices = [4 * g, 4 * g + 2, 4 * g + 1, 4 * g + 3]  # Z00, Z01, Z10, Z11
            for zi, w_idx in enumerate(w_indices):
                for mult_idx, coeff in output_coeffs[zi]:
                    if mult_idx == mi:
                        W[w_idx][col] += coeff
            
            col += 1
        
        # For A2 * B_g2:
        # X = A2 = [[a02,a03],[a12,a13]] -> A indices: 2,3,6,7
        # Y = B_g2
        for mi, (x_coeff, y_coeff) in enumerate(strassen_defs):
            A_idx_map = {'X00': 2, 'X01': 3, 'X10': 6, 'X11': 7}
            B_idx_map = {
                'Y00': 2 * p + 2 * g,     # b2_{2g}
                'Y01': 2 * p + 2 * g + 1, # b2_{2g+1}
                'Y10': 3 * p + 2 * g,     # b3_{2g}
                'Y11': 3 * p + 2 * g + 1, # b3_{2g+1}
            }
            
            for x_name, x_val in x_coeff.items():
                U[A_idx_map[x_name]][col] += x_val
            for y_name, y_val in y_coeff.items():
                V[B_idx_map[y_name]][col] += y_val
            
            w_indices = [4 * g, 4 * g + 2, 4 * g + 1, 4 * g + 3]  # Z00, Z01, Z10, Z11
            for zi, w_idx in enumerate(w_indices):
                for mult_idx, coeff in output_coeffs[zi]:
                    if mult_idx == mi:
                        W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
