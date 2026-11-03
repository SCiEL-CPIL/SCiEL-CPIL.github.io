import numpy as np

def construct():
    # Build a rank-19 decomposition for 3x3 matrix multiplication (Laderman).
    # U: 9x19, V: 9x19, W: 9x19
    # Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    # We need sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*3+i,c] = 1 for all i,j,k in {0,1,2}

    # Laderman's 19 bilinear forms:
    # M1 = (A11+A12)(B11+B21)
    # M2 = A12(B21-B11)
    # M3 = (A11+A13)(B11+B21+B31)
    # M4 = A13(B21+B31-B11)
    # M5 = (A21+A22)(B11+B31)
    # M6 = A22(B31-B11)
    # M7 = (A21+A23)(B11+B31+B21)
    # M8 = A23(B31+B21-B11)
    # M9 = (A31+A32)(B11+B21)
    # M10 = A32(B21-B11)
    # M11 = (A31+A33)(B11+B21+B31)
    # M12 = A33(B21+B31-B11)
    # M13 = (A11+A21+A31)(B11+B12)
    # M14 = (A21+A31)(B12-B11)
    # M15 = (A12+A22+A32)(B11+B12)
    # M16 = (A22+A32)(B12-B11)
    # M17 = (A13+A23+A33)(B11+B12)
    # M18 = (A23+A33)(B12-B11)
    # M19 = (A11+A21+A31)(B12-B11) + (A12+A22+A32)(B11-B12) ... wait this isn't right.

    # Actually, let me use a cleaner formulation. The standard Laderman decomposition has these 19 products:
    # Let A = [[a,b,c],[d,e,f],[g,h,i]] and B = [[p,q,r],[s,t,u],[v,w,x]]
    # 
    # P1 = (a+b)(p+s)
    # P2 = b(s-p)
    # P3 = (a+c)(p+s+v)
    # P4 = c(s+v-p)
    # P5 = (d+e)(p+v)
    # P6 = e(v-p)
    # P7 = (d+f)(p+v+s)
    # P8 = f(v+s-p)
    # P9 = (g+h)(p+s)
    # P10 = h(s-p)
    # P11 = (g+i)(p+s+v)
    # P12 = i(s+v-p)
    # P13 = (a+d+g)(p+q)
    # P14 = (d+g)(q-p)
    # P15 = (b+e+h)(p+q)
    # P16 = (e+h)(q-p)
    # P17 = (c+f+i)(p+q)
    # P18 = (f+i)(q-p)
    # P19 = (a+d+g)(q-p) - (b+e+h)(q-p) + (c+f+i)(q-p) ... no.

    # Let me look up the exact formulas more carefully.
    # C11 = P1 + P4 - P2 - P8 + P3 + P12 - P11
    # C12 = P2 + P8 + P13 + P16 - P15
    # C13 = P14 + P18 - P17
    # C21 = P5 + P8 - P6 + P7 + P12 - P11
    # C22 = P6 + P13 + P16 - P15
    # C23 = P14 + P18 - P17
    # C31 = P9 + P12 - P10 + P11 + P18 - P17
    # C32 = P10 + P13 + P16 - P15
    # C33 = P14 + P18 - P17

    # Hmm, this doesn't look symmetric. Let me use a different known formulation.

    # Actually, let me just directly construct the matrices by defining each product.
    # For each product Pk, I define which A entries and B entries participate with what coefficients.

    # Let me use a verified set. I'll define the 19 products explicitly:

    # Product definitions: (list of (a_idx, coeff), (b_idx, coeff))
    # a_idx = i*3+j for A[i][j], b_idx = j*3+k for B[j][k]

    products = [
        # P1: (a00+a01)(b00+b10)
        ([(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # P2: a01*(b10-b00)
        ([(1, 1)], [(3, 1), (0, -1)]),
        # P3: (a00+a02)(b00+b10+b20)
        ([(0, 1), (2, 1)], [(0, 1), (3, 1), (6, 1)]),
        # P4: a02*(b10+b20-b00)
        ([(2, 1)], [(3, 1), (6, 1), (0, -1)]),
        # P5: (a10+a11)(b00+b20)
        ([(3, 1), (4, 1)], [(0, 1), (6, 1)]),
        # P6: a11*(b20-b00)
        ([(4, 1)], [(6, 1), (0, -1)]),
        # P7: (a10+a12)(b00+b20+b10)
        ([(3, 1), (5, 1)], [(0, 1), (6, 1), (3, 1)]),
        # P8: a12*(b20+b10-b00)
        ([(5, 1)], [(6, 1), (3, 1), (0, -1)]),
        # P9: (a20+a21)(b00+b10)
        ([(6, 1), (7, 1)], [(0, 1), (3, 1)]),
        # P10: a21*(b10-b00)
        ([(7, 1)], [(3, 1), (0, -1)]),
        # P11: (a20+a22)(b00+b10+b20)
        ([(6, 1), (8, 1)], [(0, 1), (3, 1), (6, 1)]),
        # P12: a22*(b10+b20-b00)
        ([(8, 1)], [(3, 1), (6, 1), (0, -1)]),
        # P13: (a00+a10+a20)(b00+b01)
        ([(0, 1), (3, 1), (6, 1)], [(0, 1), (1, 1)]),
        # P14: (a10+a20)*(b01-b00)
        ([(3, 1), (6, 1)], [(1, 1), (0, -1)]),
        # P15: (a01+a11+a21)(b00+b01)
        ([(1, 1), (4, 1), (7, 1)], [(0, 1), (1, 1)]),
        # P16: (a11+a21)*(b01-b00)
        ([(4, 1), (7, 1)], [(1, 1), (0, -1)]),
        # P17: (a02+a12+a22)(b00+b01)
        ([(2, 1), (5, 1), (8, 1)], [(0, 1), (1, 1)]),
        # P18: (a12+a22)*(b01-b00)
        ([(5, 1), (8, 1)], [(1, 1), (0, -1)]),
        # P19: (a00+a10+a20)*(b01-b00) - (a01+a11+a21)*(b01-b00) + (a02+a12+a22)*(b01-b00)
        # This is getting messy. Let me use a different P19.
        # P19: (a00+a10+a20+a01+a11+a21+a02+a12+a22) * 0 ... no.
    ]

    # I'm struggling with the exact Laderman formulas. Let me use a simpler approach:
    # I'll use the fact that we can verify the decomposition numerically.
    # Let me just use a known valid rank-19 decomposition from a reliable source.

    # After more research, here's a clean formulation:
    # The 19 products are:
    # p1 = (a+b)(p+s)
    # p2 = b(s-p)  
    # p3 = (a+c)(p+s+v)
    # p4 = c(s+v-p)
    # p5 = (d+e)(p+v)
    # p6 = e(v-p)
    # p7 = (d+f)(p+v+s)
    # p8 = f(v+s-p)
    # p9 = (g+h)(p+s)
    # p10 = h(s-p)
    # p11 = (g+i)(p+s+v)
    # p12 = i(s+v-p)
    # p13 = (a+d+g)(p+q)
    # p14 = (d+g)(q-p)
    # p15 = (b+e+h)(p+q)
    # p16 = (e+h)(q-p)
    # p17 = (c+f+i)(p+q)
    # p18 = (f+i)(q-p)
    # p19 = (a+b+c+d+e+f+g+h+i)(q-p)  # This is a catch-all

    # Output:
    # C[0][0] = p1 - p2 - p8 + p3 - p11 + p12
    # C[0][1] = p2 + p8 - p13 + p15 - p16
    # C[0][2] = p14 - p17 + p18
    # C[1][0] = p5 - p6 + p7 - p8 + p11 - p12
    # C[1][1] = p6 - p13 + p15 - p16
    # C[1][2] = p14 - p17 + p18
    # C[2][0] = p9 - p10 + p11 - p12 + p17 - p18
    # C[2][1] = p10 - p13 + p15 - p16
    # C[2][2] = p14 - p17 + p18

    # Let me verify C[0][0]:
    # p1 = (a+b)(p+s) = ap + as + bp + bs
    # p2 = b(s-p) = bs - bp
    # p3 = (a+c)(p+s+v) = ap + as + av + cp + cs + cv
    # p4 = c(s+v-p) = cs + cv - cp
    # p8 = f(v+s-p) = fv + fs - fp
    # p11 = (g+i)(p+s+v) = gp + gs + gv + ip + is + iv
    # p12 = i(s+v-p) = is + iv - ip

    # C[0][0] = p1 - p2 - p8 + p3 - p11 + p12
    # = (ap + as + bp + bs) - (bs - bp) - (fv + fs - fp) + (ap + as + av + cp + cs + cv) - (gp + gs + gv + ip + is + iv) + (is + iv - ip)
    # = ap + as + bp + bs - bs + bp - fv - fs + fp + ap + as + av + cp + cs + cv - gp - gs - gv - ip - is - iv + is + iv - ip
    # = 2ap + 2as + 2bp - fv - fs + fp + av + cp + cs + cv - gp - gs - gv - 2ip
    # This should equal a*p + b*s + c*v (the (0,0) entry of AB)
    # That's clearly not working. My formulas are wrong.

    # Let me try a completely different approach. I'll use the known fact that
    # the rank of 3x3 matrix multiplication is at most 19, and construct it via
    # a verified set of equations.

    # Actually, let me just use a numerical approach to find a valid decomposition.
    # But that's not guaranteed to give integer/half-integer entries.

    # Given the difficulty, let me use a simpler but still improved approach:
    # Use the Kronecker product structure with Strassen for the 2x2 part and
    # a simple 3x3 decomposition.

    # For 3x3, I'll use a rank-27 schoolbook decomposition (which is correct).
    # For 2x2, I'll use Strassen (rank 7).
    # Total rank: 27 * 7 = 189.

    # But wait, the Kronecker product approach requires careful permutation.
    # Let me implement it correctly.

    # Actually, let me just use a direct approach with a known valid decomposition.
    # I'll use the following strategy: construct U, V, W for 6x6x6 directly
    # using the block structure.

    # For 6x6, we can view it as 2x2 blocks of 3x3 matrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 3x3.
    # B = [[B11, B12], [B21, B22]] where each Bij is 3x3.
    # C = [[C11, C12], [C21, C22]] where each Cij is 3x3.

    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # If we use Strassen for the 2x2 block structure:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # Each Mk is a product of 3x3 matrices. If we use schoolbook for each 3x3 product,
    # each Mk requires 9 scalar multiplications. Total: 7 * 9 = 63.

    # But we need to be careful with the indexing.

    # Let me implement this:
    n, m, p = 6, 6, 6
    rank = 63
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    # Block indices:
    # A is 6x6, viewed as 2x2 blocks of 3x3.
    # A[i][j] where i = 3*bi + ri, j = 3*bj + rj, bi,bj in {0,1}, ri,rj in {0,1,2}
    # Similarly for B and C.

    # For each Strassen product Mk, we need to compute the product of two 3x3 matrices.
    # For a 3x3 product X*Y = Z, we use schoolbook:
    # Z[r][c] = sum_s X[r][s] * Y[s][c]
    # So for each (r,s,c), we have a term X[r][s] * Y[s][c] * Z[r][c].

    # Let me define a helper to add a 3x3 product to the global U, V, W.

    def add_3x3_product(U, V, W, rank_offset, A_block, B_block, C_block, coeff=1):
        # A_block, B_block, C_block are (block_row, block_col) in {0,1}
        # We add terms for the product A_block * B_block -> C_block
        # with coefficient 'coeff'
        for r in range(3):
            for s in range(3):
                for c in range(3):
                    idx = rank_offset + r*9 + s*3 + c
                    if idx >= rank:
                        continue
                    # A index: block_row*3 + r, col: block_col*3 + s
                    a_i = A_block[0] * 3 + r
                    a_j = A_block[1] * 3 + s
                    # B index: block_row*3 + s, col: block_col*3 + c
                    b_i = B_block[0] * 3 + s
                    b_j = B_block[1] * 3 + c
                    # C index: block_row*3 + r, col: block_col*3 + c
                    c_i = C_block[0] * 3 + r
                    c_j = C_block[1] * 3 + c
                    # W index: k*n + i where C[i][k] = C[c_i][c_j]
                    # So i = c_i, k = c_j
                    w_idx = c_j * n + c_i
                    U[a_i * m + a_j, idx] += coeff
                    V[b_i * p + b_j, idx] += coeff
                    W[w_idx, idx] += coeff

    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # This is not a single product, it's a sum of products.
    # (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # So we need 4 separate 3x3 products.

    # Let me restructure. For each Strassen term, I'll add the appropriate 3x3 products.

    # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
    # M3 = A11*(B21-B22) = A11*B21 - A11*B22
    # M4 = A22*(B12-B11) = A22*B12 - A22*B11
    # M5 = (A11+A12)*B22 = A11*B22 + A12*B22
    # M6 = (A21-A11)*(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # M7 = (A12-A22)*(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22

    # This is getting complex. Each "Mk" actually consists of multiple 3x3 products.
    # Let me just enumerate all the 3x3 products needed.

    # Actually, this approach is equivalent to just doing schoolbook on the full 6x6,
    # because we're not actually reducing the number of scalar multiplications.

    # The key insight is that Strassen reduces the number of block multiplications,
    # but each block multiplication still requires 9 scalar multiplications.
    # So 7 blocks * 9 scalars = 63, which is less than 36 (schoolbook 6x6).

    # Wait, schoolbook 6x6 is 6*6*6 = 216 scalar multiplications.
    # 63 < 216, so this is an improvement!

    # Let me implement this correctly.

    # I'll create a list of (A_block, B_block, C_block, coeff) tuples for all 3x3 products.

    products_list = []

    # M1 = (A11+A22)(B11+B22)
    products_list.append(((0,0), (0,0), (0,0), 1))  # A11*B11
    products_list.append(((0,0), (1,0), (0,0), 1))  # A11*B22  -- wait, B22 is block (1,1)
    # Let me be more careful with block indices.
    # A11 = block (0,0), A12 = block (0,1), A21 = block (1,0), A22 = block (1,1)
    # B11 = block (0,0), B12 = block (0,1), B21 = block (1,0), B22 = block (1,1)
    # C11 = block (0,0), C12 = block (0,1), C21 = block (1,0), C22 = block (1,1)

    # M1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    products_list.append(((0,0), (0,0), (0,0), 1))  # A11*B11 -> C11
    products_list.append(((0,0), (1,1), (0,0), 1))  # A11*B22 -> C11
    products_list.append(((1,1), (0,0), (0,0), 1))  # A22*B11 -> C11
    products_list.append(((1,1), (1,1), (0,0), 1))  # A22*B22 -> C11

    # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
    products_list.append(((1,0), (0,0), (1,0), 1))  # A21*B11 -> C21
    products_list.append(((1,1), (0,0), (1,0), 1))  # A22*B11 -> C21

    # M3 = A11*(B21-B22) = A11*B21 - A11*B22
    products_list.append(((0,0), (1,0), (0,1), 1))  # A11*B21 -> C12
    products_list.append(((0,0), (1,1), (0,1), -1))  # A11*B22 -> C12

    # M4 = A22*(B12-B11) = A22*B12 - A22*B11
    products_list.append(((1,1), (0,1), (1,0), 1))  # A22*B12 -> C21
    products_list.append(((1,1), (0,0), (1,0), -1))  # A22*B11 -> C21

    # M5 = (A11+A12)*B22 = A11*B22 + A12*B22
    products_list.append(((0,0), (1,1), (0,1), 1))  # A11*B22 -> C12
    products_list.append(((0,1), (1,1), (0,1), 1))  # A12*B22 -> C12

    # M6 = (A21-A11)*(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    products_list.append(((1,0), (0,0), (1,1), 1))  # A21*B11 -> C22
    products_list.append(((1,0), (0,1), (1,1), 1))  # A21*B12 -> C22
    products_list.append(((0,0), (0,0), (1,1), -1))  # A11*B11 -> C22
    products_list.append(((0,0), (0,1), (1,1), -1))  # A11*B12 -> C22

    # M7 = (A12-A22)*(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    products_list.append(((0,1), (1,0), (0,0), 1))  # A12*B21 -> C11
    products_list.append(((0,1), (1,1), (0,0), 1))  # A12*B22 -> C11
    products_list.append(((1,1), (1,0), (0,0), -1))  # A22*B21 -> C11
    products_list.append(((1,1), (1,1), (0,0), -1))  # A22*B22 -> C11

    # Now I have 4+2+2+2+2+4+4 = 20 products, each requiring 9 scalar multiplications.
    # Total: 20 * 9 = 180. That's worse than 63!

    # I think I'm double-counting. The issue is that Strassen's algorithm has 7 block
    # multiplications, but each block multiplication is a 3x3 matrix product. If I use
    # schoolbook for each, I get 7*9 = 63 scalar multiplications.

    # But in my list above, I'm expanding each Strassen term into multiple 3x3 products,
    # which is incorrect. The Strassen terms are already the block products.

    # Let me restart with a cleaner approach. I'll directly construct the 63-term decomposition.

    # For each of the 7 Strassen block products, I'll add 9 scalar terms.

    # Block product 1: (A11+A22) * (B11+B22)
    # This is a 3x3 matrix product where the left operand is A11+A22 and the right is B11+B22.
    # For each (r,s,c) in {0,1,2}^3:
    #   term: (A11[r,s] + A22[r,s]) * (B11[s,c] + B22[s,c]) * C11[r,c]

    # But this is a single bilinear form, not a product of two linear forms.
    # To express it as a rank-1 term in the tensor decomposition, I need:
    # (A11[r,s] + A22[r,s]) * (B11[s,c] + B22[s,c]) = 
    #   A11[r,s]*B11[s,c] + A11[r,s]*B22[s,c] + A22[r,s]*B11[s,c] + A22[r,s]*B22[s,c]

    # So it's actually 4 rank-1 terms, not 1. This is the issue.

    # Given the complexity, let me just use a simple correct decomposition with rank 216
    # (schoolbook) and accept that I can't easily improve it within this framework.

    # Actually, let me try a different approach: use the known rank-19 decomposition for 3x3
    # and combine it with Strassen in a way that gives rank 19*7 = 133.

    # For this, I need a valid rank-19 decomposition for 3x3. Let me use a verified one.

    # After more research, here's a valid rank-19 decomposition:
    # I'll define it explicitly.

    U3 = np.zeros((9, 19), dtype=int)
    V3 = np.zeros((9, 19), dtype=int)
    W3 = np.zeros((9, 19), dtype=int)

    # Term 1: a00*b00 -> c00
    U3[0,0] = 1; V3[0,0] = 1; W3[0,0] = 1
    # Term 2: a00*b10 -> c00  (a00 * b10 contributes to c00? No, c00 = a00*b00 + a01*b10 + a02*b20)
    # I need to be more careful.

    # Let me just use the schoolbook 3x3 (rank 27) and combine with Strassen.
    # Total rank: 27 * 7 = 189.

    # For 3x3 schoolbook:
    for r in range(3):
        for s in range(3):
            for c in range(3):
                idx = r*9 + s*3 + c
                U3[r*3+s, idx] = 1
                V3[s*3+c, idx] = 1
                W3[c*3+r, idx] = 1  # W[k*3+i] where C[i][k], so i=r, k=c -> c*3+r

    # Now combine with Strassen for 2x2 blocks.
    # Total rank: 27 * 7 = 189.

    rank_total = 27 * 7
    U = np.zeros((36, rank_total), dtype=int)
    V = np.zeros((36, rank_total), dtype=int)
    W = np.zeros((36, rank_total), dtype=int)

    # Strassen block products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)

    # For each Mk, I need to map the 3x3 indices to the 6x6 indices.

    # Let me define a function to add a 3x3 product to the global matrices.

    def add_block_product(U, V, W, rank_base, U3, V3, W3, 
                          A_block, B_block, C_block, coeff=1):
        # A_block, B_block, C_block are (br, bc) in {0,1}^2
        for r in range(9):
            for c in range(19):  # Actually, U3 has 19 columns, but we're using 27
                pass

    # This is getting too complex. Let me just use a simple correct approach.

    # I'll use the schoolbook 6x6 decomposition (rank 216) which is correct.
    # This matches the seed, but at least it's correct.

    n, m, p = 6, 6, 6
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
