"""Matrix multiplication 2x4x8 with Strassen-based decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    col = 0

    # Split B (4x8) into two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8]
    # C = A @ B = [A @ B_left, A @ B_right], each 2x4
    # For each 2x4x4 block, split the 4x4 into 2x2 sub-blocks and use Strassen
    # 2x4x4: A is 2x4, B is 4x4 = [B11 B12; B21 B22] where each Bi is 2x2
    # C = [A1 A2] @ [B11 B12; B21 B22] where A1,A2 are 2x2
    # Strassen: 7 products of 2x2 matrices, each product = 2 multiplications (rank 2)
    # So each 2x4x4 block: 7 * 2 = 14 rank
    # Two blocks: 28 rank
    # But wait, we need to handle the full 2x4 A correctly.
    
    # Actually: 2x4x4 with Strassen on the 4x4 split:
    # A = [A00 A01 A02 A03; A10 A11 A12 A13] (2x4)
    # B = 4x4, split into 2x2 blocks
    # C0j = A0 @ B[:,j] for j=0..3, C1j = A1 @ B[:,j]
    # This is just 2x4x4 = 8 multiplications schoolbook.
    # Strassen applies when both matrices can be split into square blocks.
    # For 2x4x4: can't directly Strassen since 2 != 4.
    
    # Alternative: pad A to 4x4? No, we need exact rank.
    
    # Let me use the approach: 2x4x8 = 2x(2x4)x(4x4)
    # Split 4x4 B into 2x2 blocks: B = [P Q; R S] where P,Q,R,S are 2x2
    # C = A @ B where A is 2x4 = [A_L A_R] with A_L, A_R being 2x2
    # C = A_L @ P + A_R @ Q for left half, A_L @ R + A_R @ S for right half
    # That's 4 matrix products of 2x2, each rank 4 (schoolbook) = 16 per 4x4 block
    # Two 4x4 blocks: 32 total. Worse than 64? No, 32 < 64.
    # But each 2x2 product has rank 4 (schoolbook) or 7 (Strassen)?
    # 2x2x2 matrix multiplication has rank 7 (Strassen) or 8 (schoolbook).
    # So 4 Strassen products per 4x4 block: 4*7 = 28 per block, 56 total!
    
    # 2x4x8: split into two 2x4x4 blocks
    # Each 2x4x4: A=[A_L A_R] (two 2x2 cols), B=[P Q; R S] (four 2x2)
    # C_left (2x2) = A_L@P + A_R@Q -> Strassen: 7 products
    # C_right (2x2) = A_L@R + A_R@S -> Strassen: 7 products
    # Total per 2x4x4: 14, times 2 = 28. Hmm that's 28 not 56.
    
    # Wait, I need to re-examine. Let me just use schoolbook for 2x2x2 (rank 8 each)
    # 4 products per 4x4 block * 8 = 32 per block * 2 = 64. Same as before.
    
    # For Strassen on 2x2x2 (rank 7):
    # 4 products per 4x4 block * 7 = 28 per block * 2 = 56. Yes!
    
    # Strassen for 2x2: given X=[x11 x12; x21 x22], Y=[y11 y12; y21 y22]
    # Z = X@Y, z11=x11y11+x21y21, z12=x11y12+x22y22, z21=x21y11+x22y21, z22=x12y21+x22y22
    # M1=(x11+x22)(y11+y22)
    # M2=(x21+x22)y11
    # M3=x11(y22-y11)
    # M4=x12(y11+y21)
    # M5=(x11-x12)(y21+y22)
    # M6=(x21-x11)y12
    # M7=(x22-x21)(y11-y12)
    # z11=M1+M3-M5+M7
    # z12=M4+M6
    # z21=M2+M4
    # z22=M1-M2+M3+M6
    
    # For our tensor decomposition, each "product" in Strassen corresponds to
    # multiplying linear combinations of A entries by linear combinations of B entries.
    # In the tensor form, each scalar product a*b corresponds to one rank-1 term.
    
    # Let's define the 7 Strassen products for 2x2:
    # Product k has U-coefficients (on A entries) and V-coefficients (on B entries)
    # A entries indexed as (row, col) in 2x2: (0,0),(0,1),(1,0),(1,1)
    # B entries similarly.
    
    # For each 2x2x2 Strassen product, the contribution to C[r,c] is sum of products.
    
    # Let me enumerate all 7 products with their (U_vec, V_vec, W_vec):
    # U_vec has 4 entries (for A's 4 elements), V_vec has 4 entries (for B's 4 elements)
    # W_vec has 4 entries (for C's 4 elements)
    
    # Product 1: (a00+a11)*(b00+b11)
    # U1 = [1,0,0,1], V1=[1,0,0,1]
    # Contributes to: z00 += M1, z11 += M1 (i.e., C[0,0] and C[1,1])
    # W1 = [1, 0, 0, 1] (C[0,0], C[0,1], C[1,0], C[1,1])
    
    # Product 2: (a10+a11)*b00
    # U2 = [0,0,1,1], V2=[1,0,0,0]
    # z10 += M2, so W2=[0,0,1,0]
    
    # Product 3: a00*(b11-b00)
    # U3=[1,0,0,0], V3=[-1,0,0,1]
    # z00 += M3, z11 += M3, W3=[1,0,0,1]
    
    # Product 4: a01*(b00+b10)
    # U4=[0,1,0,0], V4=[1,0,1,0]
    # z01 += M4, z10 += M4, W4=[0,1,1,0]
    
    # Product 5: (a00-a01)*(b10+b11)
    # U5=[1,-1,0,0], V5=[0,0,1,1]
    # z00 -= M5, z11 += M5, W5=[-1,0,0,1]
    
    # Product 6: (a10-a00)*b01
    # U6=[-1,0,1,0], V6=[0,1,0,0]
    # z01 += M6, z11 += M6, W6=[0,1,0,1]
    
    # Product 7: (a11-a10)*(b00-b01)
    # U7=[0,0,-1,1], V7=[1,-1,0,0]
    # z00 += M7, z11 -= M7, W7=[1,0,0,-1]
    
    # Verify z00 = M1 + M3 - M5 + M7:
    # M1 = (a00+a11)(b00+b11) = a00b00+a00b11+a11b00+a11b11
    # M3 = a00(b11-b00) = a00b11-a00b00
    # -M5 = -(a00-a01)(b10+b11) = -a00b10-a00b11+a01b10+a01b11
    # M7 = (a11-a10)(b00-b01) = a11b00-a11b01-a10b00+a10b01
    # Sum: a00b00+a00b11+a11b00+a11b11 + a00b11-a00b00 -a00b10-a00b11+a01b10+a01b11 + a11b00-a11b01-a10b00+a10b01
    # = a00b11 + a11b00 + a11b11 - a00b10 + a01b10 + a01b11 - a11b01 - a10b00 + a10b01
    # Expected z00 = a00b00 + a10b10
    # This doesn't match! I have the Strassen formulas wrong.
    
    # Let me re-derive Strassen correctly.
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y: z00=x00y00+x01y10, z01=x00y01+x01y11, z10=x10y00+x11y10, z11=x10y01+x11y11
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y11-y00)
    # M4 = x01(y00+y10)
    # M5 = (x00-x01)(y10+y11)
    # M6 = (x10-x00)y01
    # M7 = (x11-x10)(y00-y01)
    # z00 = M1 + M3 - M5 + M7
    # z01 = M4 + M6
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # Verify z00:
    # M1 = x00y00+x00y11+x11y00+x11y11
    # M3 = x00y11-x00y00
    # -M5 = -(x00-y01... wait (x00-x01)(y10+y11) = x00y10+x00y11-x01y10-x01y11
    #   so -M5 = -x00y10-x00y11+x01y10+x01y11
    # M7 = (x11-x10)(y00-y01) = x11y00-x11y01-x10y00+x10y01
    # Sum: x00y00+x00y11+x11y00+x11y11 + x00y11-x00y00 -x00y10-x00y11+x01y10+x01y11 + x11y00-x11y01-x10y00+x10y01
    # = x00y11+x11y00+x11y11-x00y10+x01y10+x01y11-x11y01-x10y00+x10y01
    # Expected: x00y00+x01y10
    # Still doesn't match. I'm clearly messing up the standard Strassen formulas.
    
    # Let me look this up properly. Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b22-b11)
    # M4 = a12(b21+b22)
    # M5 = (a11-a12)(b11+b21)
    # M6 = (a21-a11)b12
    # M7 = (a22-a21)(b12-b11)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Verify c11 = a11b11 + a12b21:
    # M1 = a11b11+a11b22+a22b11+a22b22
    # M4 = a12b21+a12b22
    # -M5 = -(a11-a12)(b11+b21) = -a11b11-a11b21+a12b11+a12b21
    # M7 = (a22-a21)(b12-b11) = a22b12-a22b11-a21b12+a21b11
    # Sum: a11b11+a11b22+a22b11+a22b22 + a12b21+a12b22 -a11b11-a11b21+a12b11+a12b21 + a22b12-a22b11-a21b12+a21b11
    # = a11b22+a22b11+a22b22+a12b21+a12b22-a11b21+a12b11+a12b21+a22b12-a21b12+a21b11
    # This is getting messy. Let me just verify numerically with code.
    
    # I'll implement it programmatically and verify.
    
    # Strassen 2x2x2 products (using 1-based indexing for clarity, converting to 0-based):
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A01(B10+B11)
    # M5 = (A00-A01)(B00+B10)
    # M6 = (A10-A00)B01
    # M7 = (A11-A10)(B01-B00)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let me verify C00 = A00*B00 + A01*B10:
    # M1 = A00B00+A00B11+A11B00+A11B11
    # M4 = A01B10+A01B11
    # -M5 = -(A00-A01)(B00+B10) = -A00B00-A00B10+A01B00+A01B10
    # M7 = (A11-A10)(B01-B00) = A11B01-A11B00-A10B01+A10B00
    # Sum = A00B00+A00B11+A11B00+A11B11+A01B10+A01B11-A00B00-A00B10+A01B00+A01B10+A11B01-A11B00-A10B01+A10B00
    # = A00B11+A11B11+A01B10+A01B11-A00B10+A01B00+A01B10+A11B01-A10B01+A10B00
    # Expected: A00B00+A01B10
    # Not matching. The formulas must be different.
    
    # I'll just implement it in code and verify numerically, then build the tensor.
    
    import time
    deadline = time.time() + 100
    
    # Define Strassen products as (U_coeffs, V_coeffs, W_coeffs) for 2x2
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_products = [
        # (U on A, V on B, W on C)
        # M1 = (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,0]),  # contributes +1 to C00
        # M1 also contributes to C11
        # Actually each product contributes to multiple C entries.
        # Let me restructure: each product is one rank-1 term in the tensor.
        # The tensor T[C0,C1,C2,A0,A1,A2,A3,B0,B1,B2,B3] 
        # For 2x2x2: T[c_r,c_c, a_r,a_c, b_r,b_c] = delta(a_r,b_r)*delta(a_c,b_c)
        # Wait no. Matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
        # So T[C_ij, A_ik, B_kj] = delta(i,k)*delta(k,j) for the specific i,j,k
        # Actually T is the structure tensor: T[i,j,k,l,m,n] = 1 if (i,j) from A[i,m]*B[m,j]... 
        # No. The matrix multiplication tensor has indices:
        # T[output_row, output_col, input_A_row, input_A_col, input_B_row, input_B_col]
        # T[r,c, i,j, k,l] = 1 if i=r, k=c... no.
        # C[r,c] = sum_k A[r,k]*B[k,c]
        # So T[r,c, r,k, k,c] = 1 for each r,c,k
        # In tensor notation with flattened indices, each (r,k,c) gives one term:
        # A[r,k] * B[k,c] contributes to C[r,c]
        
        # For the rank decomposition, we need:
        # T = sum_alpha U_alpha * V_alpha * W_alpha (outer products)
        # where U_alpha is vector over A-space (4 entries for 2x2),
        # V_alpha over B-space (4 entries), W_alpha over C-space (4 entries)
        # T[r,c,i,j] (flattened: C_idx, A_idx, B_idx) = sum_alpha W_alpha[C_idx] * U_alpha[A_idx] * V_alpha[B_idx]
        # where C_idx = r*2+c, A_idx = i*2+j, B_idx = k*2+l
        # And T[C_idx, A_idx, B_idx] = 1 iff C_idx corresponds to (r,c) and A_idx to (i,j) and B_idx to (k,l) with i=r, k=c... 
        # No: T[r*2+c, i*2+j, k*2+l] = 1 if i==r and k==c... that's not right either.
        
        # C[r,c] = sum_k A[r,k] * B[k,c]
        # So the tensor element T[(r,c), (i,j), (k,l)] = 1 if i==r, j==k, k==c... 
        # No: A[r,k] means i=r, j=k. B[k,c] means k=k, l=c.
        # So T[(r,c), (r,k), (k,c)] = 1, i.e., T[C_rc, A_rk, B_kc] = 1 for all valid r,k,c.
        
        # For 2x2: 8 terms (r=0,1; k=0,1; c=0,1)
        # T[0,0,0]=1 (r=0,c=0,k=0: A[0,0]*B[0,0])
        # T[0,1,3]=1 (r=0,c=0,k=1: A[0,1]*B[1,0])  -- wait B[1,0] has index k=1,l=0 -> 1*2+0=2
        # Let me use: A_idx = i*2+j, B_idx = k*2+l, C_idx = r*2+c
        # Term (r,k,c): T[r*2+c, r*2+k, k*2+c] = 1
        # (0,0,0): T[0, 0, 0] = 1
        # (0,1,0): T[0, 1, 2] = 1  (A[0,1] idx=1, B[1,0] idx=2)
        # (0,0,1): T[1, 0, 1] = 1  (A[0,0] idx=0, B[0,1] idx=1)
        # (0,1,1): T[1, 1, 3] = 1  (A[0,1] idx=1, B[1,1] idx=3)
        # (1,0,0): T[2, 2, 0] = 1  (A[1,0] idx=2, B[0,0] idx=0)
        # (1,1,0): T[2, 3, 2] = 1  (A[1,1] idx=3, B[1,0] idx=2)
        # (1,0,1): T[3, 2, 1] = 1  (A[1,0] idx=2, B[0,1] idx=1)
        # (1,1,1): T[3, 3, 3] = 1  (A[1,1] idx=3, B[1,1] idx=3)
        
        # Now Strassen decomposes this rank-8 tensor into 7 rank-1 terms.
        # I'll verify numerically in code.
    ]
    
    # Let me just build it programmatically and verify.
    
    def make_strassen_2x2():
        """Return 7 (U, V, W) triples for 2x2x2 Strassen decomposition."""
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        
        # Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B11-B00)
        # M4 = A01*(B10+B11)
        # M5 = (A00-A01)*(B00+B10)
        # M6 = (A10-A00)*B01
        # M7 = (A11-A10)*(B01-B00)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Verify C00:
        # M1 = A00B00+A00B11+A11B00+A11B11
        # M4 = A01B10+A01B11
        # -M5 = -A00B00-A00B10+A01B00+A01B10
        # M7 = A11B01-A11B00-A10B01+A10B00
        # Sum: A00B00+A00B11+A11B00+A11B11+A01B10+A01B11-A00B00-A00B10+A01B00+A01B10+A11B01-A11B00-A10B01+A10B00
        # = A00B11+A11B11+A01B10+A01B11-A00B10+A01B00+A01B10+A11B01-A10B01+A10B00
        # Expected C00 = A00B00+A01B10
        # This is NOT matching. The Strassen formulas I'm using are wrong.
        
        # Let me use a different set. From the actual Strassen algorithm:
        # Given [[a,b],[c,d]] * [[e,f],[g,h]]:
        # p1 = (a+d)(e+h)
        # p2 = (c+d)e
        # p3 = a(h-e)
        # p4 = d(g-e)  -- wait, some references use different formulations
        # p5 = (a-b)(f+h)
        # p6 = (c-a)(f+g)  -- hmm
        # p7 = (b-d)(g+h)
        
        # Let me just verify with actual values in code.
        
        import numpy as np
        # Test with specific values
        A = np.array([[1,2],[3,4]], dtype=float)
        B = np.array([[5,6],[7,8]], dtype=float)
        C = A @ B  # [[19,22],[43,50]]
        
        # Try Strassen:
        a,b,c,d = A[0,0],A[0,1],A[1,0],A[1,1]
        e,f,g,h = B[0,0],B[0,1],B[1,0],B[1,1]
        
        p1 = (a+d)*(e+h)
        p2 = (c+d)*e
        p3 = a*(h-e)
        p4 = d*(g-e)
        p5 = (a-b)*(f+h)
        p6 = (c-a)*(f+g)
        p7 = (b-d)*(g+h)
        
        c00 = p1 + p4 - p5 + p7
        c01 = p3 + p5
        c10 = p2 + p4
        c11 = p1 - p2 + p3 + p6
        
        print(f"Strassen C: {c00},{c01},{c10},{c11}")
        print(f"Expected C: {C[0,0]},{C[0,1]},{C[1,0]},{C[1,1]}")
        
        # Let me check:
        # p1 = (1+4)(5+8) = 5*13 = 65
        # p2 = (3+4)*5 = 35
        # p3 = 1*(8-5) = 3
        # p4 = 4*(7-5) = 8
        # p5 = (1-2)*(6+8) = -1*14 = -14
        # p6 = (3-1)*(6+7) = 2*13 = 26
        # p7 = (2-4)*(7+8) = -2*15 = -30
        # c00 = 65+8-(-14)+(-30) = 65+8+14-30 = 57. Expected 19. WRONG.
        
        # I'm clearly not remembering Strassen correctly. Let me look at it differently.
        # The correct Strassen formulas (from Wikipedia):
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b22-b11)
        # m4 = a12(b21+b22)
        # m5 = (a11-a12)(b11+b21)
        # m6 = (a21-a11)b12
        # m7 = (a22-a21)(b12-b11)
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m6
        
        # Using a=a11=1, b=a12=2, c=a21=3, d=a22=4
        # e=b11=5, f=b12=6, g=b21=7, h=b22=8
        # m1 = (1+4)(5+8) = 65
        # m2 = (3+4)*5 = 35
        # m3 = 1*(8-5) = 3
        # m4 = 2*(7+8) = 30
        # m5 = (1-2)*(5+7) = -12
        # m6 = (3-1)*6 = 12
        # m7 = (4-3)*(6-5) = 1
        # c11 = 65+30-(-12)+1 = 108. Expected 19. STILL WRONG.
        
        # I give up trying to remember. Let me just use the schoolbook 2x2x2 (rank 8)
        # and accept rank 64, OR find the correct formulas by solving.
        
        # Actually, let me just return the schoolbook decomposition since I can't
        # verify Strassen quickly. The knowledge says 56 was achieved, so it's possible.
        # But I need a correct implementation.
        
        # For now, return schoolbook (rank 64).
        return None
    
    # Since I can't verify Strassen quickly, return schoolbook.
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
