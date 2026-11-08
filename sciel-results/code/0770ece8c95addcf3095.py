"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strassen-style decomposition for 2x3x8 matrix multiplication
    # We split the 3x8 matrix into two 3x4 blocks: B = [B1 | B2]
    # and use Strassen-like formulas on the 2x3x4 subproblem twice.
    # 
    # For 2x3x4: A (2x3) * B (3x4) = C (2x4)
    # Split B into B_a (3x2) and B_b (3x2)
    # Split A into A1 (2x2), A2 (2x1)
    # 
    # Actually, let me use a known rank-44 construction.
    # 
    # Known result: rank(2,3,8) <= 44
    # Construction: split into two 2x3x4 problems, each of rank 22
    # 
    # For 2x3x4: rank <= 22
    # We can get this by: A(2x3)*B(3x4), split B into two 3x2 blocks
    # Each 2x3x2 has rank <= 11? No, schoolbook is 12.
    # 
    # Let me use a different approach. The known rank-44 for (2,3,8):
    # 
    # Actually, let me construct it carefully.
    # 
    # For 2x3x4 matrix multiplication:
    # A = [[a,b,c],[d,e,f]] (2x3), B = [[x1,x2,x3,x4],[x5,x6,x7,x8],[x9,x10,x11,x12]] (3x4)
    # C = A*B (2x4)
    # 
    # Split B into B1 = first 2 cols, B2 = last 2 cols (each 3x2)
    # C1 = A*B1 (2x2), C2 = A*B2 (2x2)
    # 
    # For 2x3x2: rank <= 11 (known)
    # Total: 2*11 = 22 for 2x3x4
    # 
    # For 2x3x8: split into two 2x3x4, total rank = 2*22 = 44
    # 
    # Now I need the rank-11 construction for 2x3x2.
    # 
    # 2x3x2: A(2x3)*B(3x2) = C(2x2)
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C = [[ag+bi+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    # 
    # Schoolbook: 12 products.
    # 
    # Known rank-11 construction for (2,3,2):
    # From literature, we can use:
    # 
    # Let me derive it. We need 11 rank-1 tensors.
    # 
    # Actually, let me just use a known explicit construction.
    # 
    # For (2,3,2), one rank-11 construction:
    # 
    # U (6x11), V (6x11), W (4x11)
    # 
    # Let me look this up mentally... The (2,3,2) problem has rank 11.
    # 
    # Here's a construction:
    # 
    # Products:
    # 1: a*g, 2: b*i, 3: c*k, 4: d*h, 5: e*j, 6: f*l,
    # 7: (a+b)*g, 8: (b+c)*i, 9: (d+e)*h, 10: (e+f)*j, 11: (c+f)*l
    # 
    # Hmm, that's not quite right. Let me think more carefully.
    # 
    # Actually, let me just use the schoolbook for (2,3,2) which is rank 12,
    # giving total rank 2*12*2 = 48. That's no better.
    # 
    # Let me try a different split. 
    # 
    # For (2,3,8), maybe use a Strassen-like approach on the 3x8 side.
    # 
    # Alternative: Use the known fact that rank(2,3,8) = 44.
    # 
    # Let me construct it via splitting the 8 into 4+4, and for each 2x3x4
    # use rank 22.
    # 
    # For 2x3x4 with rank 22:
    # A(2x3) * B(3x4) = C(2x4)
    # Split A into A1(2x2) and A2(2x1), B into B1(3x2) and B2(3x2)
    # C = A1*B1 + A2*B2... no that doesn't work directly.
    # 
    # Let me try: C = A*B where A is 2x3, B is 3x4.
    # Write A = [A_left(2x2) | A_right(2x1)]
    # Write B = [B_top(2x4); B_bottom(1x4)]... no, B is 3x4.
    # 
    # B = [B1(2x4); B2(1x4)] where B1 is first 2 rows, B2 is last row.
    # C = A_left * B1 + A_right * B2
    # 
    # A_left*B1: 2x2 * 2x4 = rank 8 (schoolbook)
    # A_right*B2: 2x1 * 1x4 = rank 8 (schoolbook)
    # Total: 16, not 22.
    # 
    # Hmm, that's better! 2*16 = 32 for (2,3,8).
    # 
    # Wait, let me verify. A is 2x3, split into A_left (2x2) and A_right (2x1).
    # B is 3x4, split into B1 (2x4, first 2 rows) and B2 (1x4, last row).
    # C = A_left * B1 + A_right * B2
    # 
    # A_left (2x2) * B1 (2x4): This is (2,2,4) matrix multiplication.
    # Rank of (2,2,4): schoolbook is 2*2*4 = 16.
    # Can we do better? For (2,2,4), we can use Strassen-like:
    # Actually (2,2,4) has rank 16 by schoolbook. Is there a better way?
    # 
    # A_right (2x1) * B2 (1x4): This is (2,1,4) which is just outer product, rank = 2*4 = 8.
    # 
    # Total: 16 + 8 = 24 for (2,3,4), so 2*24 = 48 for (2,3,8). Same as schoolbook.
    # 
    # Let me try a different split for (2,3,4).
    # 
    # A (2x3) * B (3x4).
    # Split B into 4 blocks of 3x1: B = [b1, b2, b3, b4] each 3x1.
    # C = [A*b1, A*b2, A*b3, A*b4]
    # Each A*bj is (2,3,1) which has rank 6 (schoolbook: 2*3*1=6).
    # Total: 4*6 = 24. Same.
    # 
    # Let me try to use Strassen on the 2x2 part.
    # 
    # For (2,3,4):
    # A = [[a,b,c],[d,e,f]]
    # B = [[x1,x2,x3,x4],[x5,x6,x7,x8],[x9,x10,x11,x12]]
    # 
    # C11 = a*x1 + b*x5 + c*x9
    # C12 = a*x2 + b*x6 + c*x10
    # C13 = a*x3 + b*x7 + c*x11
    # C14 = a*x4 + b*x8 + c*x12
    # C21 = d*x1 + e*x5 + f*x9
    # C22 = d*x2 + e*x6 + f*x10
    # C23 = d*x3 + e*x7 + f*x11
    # C24 = d*x4 + e*x8 + f*x12
    # 
    # Group columns: C = [C_left | C_right] where C_left = [C11,C12; C21,C22], C_right = [C13,C14; C23,C24]
    # 
    # C_left = A * B_left where B_left = [[x1,x2],[x5,x6],[x9,x10]] (3x2)
    # C_right = A * B_right where B_right = [[x3,x4],[x7,x8],[x11,x12]] (3x2)
    # 
    # Each is (2,3,2) matrix multiplication.
    # If rank(2,3,2) = 11, then total = 22 for (2,3,4), and 44 for (2,3,8).
    # 
    # So I need a rank-11 construction for (2,3,2).
    # 
    # (2,3,2): A(2x3)*B(3x2)=C(2x2)
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C11 = ag+bi+ck, C12 = ah+bj+cl, C21 = dg+ei+fk, C22 = dh+ej+fl
    # 
    # Schoolbook: 12 products.
    # 
    # Known rank-11 construction:
    # 
    # Products:
    # p1 = a*g
    # p2 = b*i  
    # p3 = c*k
    # p4 = d*h
    # p5 = e*j
    # p6 = f*l
    # p7 = (a+b+c)*(g+i+k)  -- no, that's not right
    # 
    # Let me think of it as: we want to save 1 product.
    # 
    # C11 = ag + bi + ck
    # C22 = dh + ej + fl
    # C12 = ah + bj + cl
    # C21 = dg + ei + fk
    # 
    # Note: C11 + C22 = ag + bi + ck + dh + ej + fl
    # C12 + C21 = ah + bj + cl + dg + ei + fk
    # 
    # Sum of all = ag+bi+ck+dh+ej+fl+ah+bj+cl+dg+ei+fk
    # 
    # Let me try:
    # p1 = a*(g+h)
    # p2 = b*(i+j)
    # p3 = c*(k+l)
    # p4 = d*(g+h)
    # p5 = e*(i+j)
    # p6 = f*(k+l)
    # p7 = (a+d)*g
    # p8 = (b+e)*i
    # p9 = (c+f)*k
    # p10 = (a+d)*h
    # p11 = (b+e)*j
    # 
    # That's 11 products. Let me check:
    # C11 = a*g + b*i + c*k
    # From p1: a*g + a*h, p2: b*i + b*j, p3: c*k + c*l
    # From p7: a*g + d*g, p8: b*i + e*i, p9: c*k + f*k
    # From p10: a*h + d*h, p11: b*j + e*j
    # 
    # Hmm, this is getting complicated. Let me just use a known explicit rank-11 for (2,3,2).
    # 
    # Actually, I recall that (2,3,2) has rank 11. Here's a construction from literature:
    # 
    # Let me use the following (verified):
    # 
    # U (6x11), V (6x11), W (4x11)
    # 
    # Row indices for U: (0,0)=a, (0,1)=b, (0,2)=c, (1,0)=d, (1,1)=e, (1,2)=f
    # Row indices for V: (0,0)=g, (0,1)=h, (1,0)=i, (1,1)=j, (2,0)=k, (2,1)=l
    # Row indices for W: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    # 
    # Column 1: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> a*g contributes to C11
    # Column 2: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> b*i contributes to C11
    # Column 3: U[0,2]=1, V[2,0]=1, W[0,0]=1  -> c*k contributes to C11
    # Column 4: U[1,0]=1, V[0,1]=1, W[1,1]=1  -> d*h contributes to C22
    # Column 5: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> e*j contributes to C22
    # Column 6: U[1,2]=1, V[2,1]=1, W[1,1]=1  -> f*l contributes to C22
    # Column 7: U[0,0]=1, V[0,1]=1, W[0,1]=1  -> a*h contributes to C12
    # Column 8: U[0,1]=1, V[1,1]=1, W[0,1]=1  -> b*j contributes to C12
    # Column 9: U[0,2]=1, V[2,1]=1, W[0,1]=1  -> c*l contributes to C12
    # Column 10: U[1,0]=1, V[0,0]=1, W[1,0]=1  -> d*g contributes to C21
    # Column 11: U[1,1]=1, V[1,0]=1, W[1,0]=1  -> e*i contributes to C21
    # 
    # Wait, that's 11 but I'm missing f*k from C21.
    # C21 = d*g + e*i + f*k
    # 
    # So I need f*k too. That would be 12.
    # 
    # The trick is to combine some terms. Let me try:
    # 
    # Column 11: U[1,1]=1, V[1,0]=1, W[1,0]=1  -> e*i
    # Column 12 would be: U[1,2]=1, V[2,0]=1, W[1,0]=1  -> f*k
    # 
    # To save one, I need to combine. For example:
    # Instead of separate e*i and f*k, use (e+f)*i and then adjust...
    # 
    # Let me try a different grouping.
    # 
    # C11 = ag + bi + ck
    # C12 = ah + bj + cl
    # C21 = dg + ei + fk
    # C22 = dh + ej + fl
    # 
    # Define:
    # p1 = a*(g+h)
    # p2 = b*(i+j)
    # p3 = c*(k+l)
    # p4 = d*(g+h)
    # p5 = e*(i+j)
    # p6 = f*(k+l)
    # p7 = (a+d)*g
    # p8 = (b+e)*i
    # p9 = (c+f)*k
    # p10 = (a+d)*h  -- wait, this is getting messy
    # 
    # Let me just use the schoolbook for (2,3,2) with rank 12, giving total 48.
    # But that's the same as the original.
    # 
    # Actually, let me try a completely different approach.
    # 
    # For (2,3,8), let me try to use rank 44 directly.
    # 
    # I'll use the following construction:
    # Split the 8 columns into 4+4.
    # For each 2x3x4 block, use rank 22.
    # 
    # For 2x3x4 with rank 22:
    # Split the 4 columns into 2+2, giving two 2x3x2 blocks.
    # For each 2x3x2, use rank 11.
    # 
    # So I really need rank-11 for (2,3,2).
    # 
    # Let me search my memory... 
    # 
    # (2,3,2) rank 11 construction:
    # 
    # A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # 
    # Products:
    # p1 = a*g
    # p2 = b*i
    # p3 = c*k
    # p4 = d*h
    # p5 = e*j
    # p6 = f*l
    # p7 = (a+b)*g
    # p8 = (b+c)*i
    # p9 = (d+e)*h
    # p10 = (e+f)*j
    # p11 = (c+f)*l
    # 
    # Let me verify:
    # C11 = ag + bi + ck = p1 + p2 + p3
    # C22 = dh + ej + fl = p4 + p5 + p6
    # C12 = ah + bj + cl
    # C21 = dg + ei + fk
    # 
    # From p7 = (a+b)*g = ag + bg
    # From p8 = (b+c)*i = bi + ci
    # From p9 = (d+e)*h = dh + eh
    # From p10 = (e+f)*j = ej + fj
    # From p11 = (c+f)*l = cl + fl
    # 
    # Hmm, I have extra terms. This doesn't seem to work directly.
    # 
    # Let me try yet another approach. I'll just use the schoolbook with rank 48
    # and accept the score of 1.0, since I can't easily find a better construction
    # within the time limit.
    # 
    # Actually wait - let me try a simpler optimization. 
    # 
    # For (2,3,8), I can split into 4 blocks of (2,3,2).
    # If I can get rank 11 for each (2,3,2), total is 44.
    # 
    # Let me try to find the rank-11 construction programmatically.
    # 
    # Actually, let me just return the schoolbook for now and focus on getting
    # a valid answer. The score will be 1.0.
    
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
