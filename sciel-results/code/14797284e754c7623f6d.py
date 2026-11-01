"""Rewritten program using Strassen-like block decomposition for 2x3 @ 3x4."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose as [A1 (1x3) @ B1 (3x4); A2 (1x3) @ B2 (3x4)]
    # where A1, A2 are rows of A, B1, B2 are columns of B? No.
    # A is 2x3. B is 3x4.
    # Result C is 2x4.
    # Let's split A into 1x3 and 1x3? No, split A by rows (2 rows) and B by cols (4 cols)?
    # Better: Split A into [A_top (1x3); A_bottom (1x3)] -> 1x3 blocks.
    # Split B into [B_left (3x2) | B_right (3x2)] -> 3x2 blocks.
    # Then C_top_left = A_top @ B_left (1x2)
    # C_top_right = A_top @ B_right (1x2)
    # C_bottom_left = A_bottom @ B_left (1x2)
    # C_bottom_right = A_bottom @ B_right (1x2)
    
    # Each 1x2 multiplication is just:
    # [a b] @ [[b11, b12], [b21, b22]] = [a*b11 + b*b21, a*b12 + b*b22]
    # This is not a square matrix multiplication, so Strassen doesn't apply directly.
    
    # Let's use a different splitting.
    # Split A (2x3) into two 2x2 blocks? No, 3 is odd.
    # Split A into A1 (2x1) and A2 (2x2).
    # Split B into B1 (1x4) and B2 (2x4).
    # C = A1 @ B1 (2x1) + A2 @ B2 (2x2).
    # A1@B1: 2x1 @ 1x4. Each row of A1 (2 entries) multiplies B1 (4 entries).
    # Actually A1 is 2x1, B1 is 1x4. Product is 2x4.
    # C1 = A1 @ B1. This is rank 1 in the tensor sense? No.
    # A1 = [a1; a2]. B1 = [b1, b2, b3, b4].
    # C1 = [a1*b1, a1*b2, a1*b3, a1*b4; a2*b1, a2*b2, a2*b3, a2*b4].
    # This can be decomposed into 4 terms:
    # t1 = a1*b1 -> C1[0,0]
    # t2 = a1*b2 -> C1[0,1]
    # t3 = a1*b3 -> C1[0,2]
    # t4 = a1*b4 -> C1[0,3]
    # t5 = a2*b1 -> C1[1,0]
    # t6 = a2*b2 -> C1[1,1]
    # t7 = a2*b3 -> C1[1,2]
    # t8 = a2*b4 -> C1[1,3]
    # So A1@B1 has rank 8.
    
    # A2 @ B2: 2x2 @ 2x4.
    # A2 = [[a11, a12], [a21, a22]]. B2 = [[b11, b12, b13, b14], [b21, b22, b23, b24]].
    # C2 = A2 @ B2. This is 2x2 @ 2x4.
    # We can split B2 into B21 (2x2) and B22 (2x2).
    # C2 = A2 @ B21 + A2 @ B22.
    # Each 2x2 @ 2x2 can be done with Strassen (rank 7).
    # So A2@B21 has rank 7, A2@B22 has rank 7. Total 14.
    # Total rank = 8 + 14 = 22.
    
    # Can we do better?
    # What if we split A into 1x3 and 1x3? (Horizontal split? No, A is 2x3, so 2 rows.)
    # Let's try splitting B into 3x1 and 3x3.
    # A @ B1 (2x3 @ 3x1) = 2x1. Rank 6.
    # A @ B2 (2x3 @ 3x3) = 2x3.
    # 2x3 @ 3x3 can be split into three 2x1 @ 1x3? No.
    # 2x3 @ 3x3 = [A_left (2x1) A_mid (2x1) A_right (2x1)] @ B2.
    # Each 2x1 @ 3x3 is rank 6. Total 18. Plus 6 = 24. No gain.
    
    # What if we split A into 2x1 and 2x2 (vertical split of A).
    # B into 1x4 and 2x4 (horizontal split of B).
    # We found rank 8 + 14 = 22.
    
    # Is there a way to reduce the 2x2 @ 2x4 part?
    # 2x2 @ 2x4 is equivalent to two 2x2 @ 2x2 multiplications.
    # Maybe we can combine them?
    # A2 (2x2) @ [B21 B22] (2x4).
    # This is multiplying a 2x2 by a 2x4.
    # This is the same as computing two 2x2 products: A2@B21 and A2@B22.
    # Since they share A2, maybe we can save some multiplications?
    # Let A2 = [[a, b], [c, d]].
    # B21 = [[e, f], [g, h]], B22 = [[i, j], [k, l]].
    # We need:
    # a*e + b*g, a*f + b*h, a*i + b*k, a*j + b*l
    # c*e + d*g, c*f + d*h, c*i + d*k, c*j + d*l
    
    # Strassen for 2x2 uses 7 multiplications.
    # Here we have 8 output elements.
    # Standard schoolbook for 2x2@2x2 is 8 mults. Strassen saves 1.
    # For 2x2@2x4, schoolbook is 16 mults.
    # Can we do better than 14?
    # Note that the outputs are linear combinations.
    # Let's try to find a decomposition of rank < 14 for 2x2@2x4.
    # Actually, 2x2@2x4 is just two 2x2@2x2 multiplications.
    # If we use Strassen for both, we get 7+7=14.
    # Is there a shared structure?
    # The inputs are (a,b,c,d) and (e,f,g,h,i,j,k,l).
    # The outputs are 8 values.
    # The tensor is M_{2,2,4} = M_{2,2,2} \otimes I_2 (roughly).
    # The rank of M_{2,2,4} is at most 14.
    # Is it 14?
    # It is known that the rank of 2x2x4 matrix multiplication is 14?
    # Actually, 2x2x4 is not a standard problem.
    # But since it's two independent 2x2x2 problems, the rank is sum of ranks = 14.
    # Unless there's a way to mix them.
    # But the outputs are distinct columns.
    # So 14 seems minimal for that block.
    
    # What about the 2x1 @ 1x4 part? Rank 8.
    # Total 22.
    
    # Can we do better by splitting differently?
    # Split A into 2x1 and 2x2.
    # Split B into 1x2, 1x1, 1x1? No, B is 3x4.
    
    # Let's try splitting A into 1x3 and 1x3? No, A is 2x3.
    # How about using the fact that 3 is odd?
    # Pad A to 2x4 and B to 4x4?
    # Then we have 2x4 @ 4x4.
    # Split A into 2x2 and 2x2.
    # Split B into 2x2 and 2x2 (horizontally).
    # C = A1@B1 + A1@B2 + A2@B1 + A2@B2.
    # But A is 2x3, so A2 is 2x1 (padded with 0).
    # B is 3x4, so B is 3x4 (padded to 4x4 with 0s in col 4? No, 3 rows -> 4 rows).
    # This padding might introduce unnecessary terms.
    
    # Let's stick with the 22 rank solution. It's better than 24.
    
    # Implementation of 22 rank solution:
    # A1 (2x1) @ B1 (1x4):
    # A1 = [A[0,0]; A[1,0]]. B1 = [B[0,0], B[0,1], B[0,2], B[0,3]].
    # Terms:
    # 0: A[0,0]*B[0,0] -> C[0,0]
    # 1: A[0,0]*B[0,1] -> C[0,1]
    # 2: A[0,0]*B[0,2] -> C[0,2]
    # 3: A[0,0]*B[0,3] -> C[0,3]
    # 4: A[1,0]*B[0,0] -> C[1,0]
    # 5: A[1,0]*B[0,1] -> C[1,1]
    # 6: A[1,0]*B[0,2] -> C[1,2]
    # 7: A[1,0]*B[0,3] -> C[1,3]
    
    # A2 (2x2) @ B2 (2x4):
    # A2 = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]].
    # B2 = [[B[1,0], B[1,1], B[1,2], B[1,3]], [B[2,0], B[2,1], B[2,2], B[2,3]]].
    # Split B2 into B21 (2x2) and B22 (2x2).
    # B21 = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]].
    # B22 = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]].
    
    # Strassen for A2 @ B21 (2x2 @ 2x2):
    # Let A2 = [[a, b], [c, d]] = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]].
    # Let B21 = [[e, f], [g, h]] = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]].
    # m1 = a*(e-h)
    # m2 = (a+b)*h
    # m3 = c*(e-f)
    # m4 = d*(f+g)
    # m5 = (a+d)*(e+g)
    # m6 = (b-d)*(g+h)
    # m7 = (a-c)*(e+f)
    # Result:
    # C21[0,0] = m5 + m4 - m2 + m6
    # C21[0,1] = m1 + m2
    # C21[1,0] = m3 + m4
    # C21[1,1] = m1 + m5 - m3 + m7
    
    # Wait, standard Strassen:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * (e + f)
    # p3 = (a + b) * (g + h)
    # p4 = (c + d) * (g + i) -- No, for 2x2:
    # p1 = (a + d) * (e + h)
    # p2 = (c + d) * (e + f)
    # p3 = (a + b) * (g + h)
    # p4 = d * (f - h)
    # p5 = a * (e - g)
    # p6 = (c - a) * (e + f)
    # p7 = (b - d) * (g + h)
    
    # Let's use the standard 7-term Strassen.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]].
    # m1 = (a + d) * (e + h)
    # m2 = (c + d) * (e + f)
    # m3 = (a + b) * (g + h)
    # m4 = d * (f - h)
    # m5 = a * (e - g)
    # m6 = (c - a) * (e + f)
    # m7 = (b - d) * (g + h)
    
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m4
    # C[1,0] = m2 + m5
    # C[1,1] = m1 - m2 + m6 + m7
    
    # We need to map these to U, V, W.
    # Each "m" corresponds to one column in U, V, W.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A1 @ B1
    # A1 is column 0 of A. B1 is row 0 of B.
    # Indices in U: i*m + j. A1 is j=0, i=0,1.
    # Indices in V: j*p + k. B1 is k=0, j=0.
    # Indices in W: k*n + i.
    
    # Term 0: A[0,0]*B[0,0]
    U[0*3+0][col] = 1 # i=0, j=0
    V[0*4+0][col] = 1 # j=0, k=0
    W[0*2+0][col] = 1 # k=0, i=0
    col += 1
    
    # Term 1: A[0,0]*B[0,1]
    U[0*3+0][col] = 1
    V[0*4+1][col] = 1 # k=1
    W[1*2+0][col] = 1
    col += 1
    
    # Term 2: A[0,0]*B[0,2]
    U[0*3+0][col] = 1
    V[0*4+2][col] = 1 # k=2
    W[2*2+0][col] = 1
    col += 1
    
    # Term 3: A[0,0]*B[0,3]
    U[0*3+0][col] = 1
    V[0*4+3][col] = 1 # k=3
    W[3*2+0][col] = 1
    col += 1
    
    # Term 4: A[1,0]*B[0,0]
    U[1*3+0][col] = 1 # i=1, j=0
    V[0*4+0][col] = 1
    W[0*2+1][col] = 1
    col += 1
    
    # Term 5: A[1,0]*B[0,1]
    U[1*3+0][col] = 1
    V[0*4+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    
    # Term 6: A[1,0]*B[0,2]
    U[1*3+0][col] = 1
    V[0*4+2][col] = 1
    W[2*2+1][col] = 1
    col += 1
    
    # Term 7: A[1,0]*B[0,3]
    U[1*3+0][col] = 1
    V[0*4+3][col] = 1
    W[3*2+1][col] = 1
    col += 1
    
    # Block 2: A2 @ B21
    # A2 = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    # B21 = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # a=A[0,1], b=A[0,2], c=A[1,1], d=A[1,2]
    # e=B[1,0], f=B[1,1], g=B[2,0], h=B[2,1]
    
    # m1 = (a+d)*(e+h)
    # U: a (i=0,j=1), d (i=1,j=2) -> U[3][col]+=1, U[5][col]+=1
    # V: e (j=1,k=0), h (j=2,k=1) -> V[4][col]+=1, V[7][col]+=1
    # W: e+h -> k=0,i=0? No, W is k*n+i.
    # The output of m1 contributes to C[0,0] and C[1,1] etc.
    # Wait, the Strassen terms are intermediate products.
    # We need to define the final W coefficients.
    
    # Let's define the 7 terms for A2@B21.
    # t1 = (a+d)*(e+h)
    # t2 = (c+d)*(e+f)
    # t3 = (a+b)*(g+h)
    # t4 = d*(f-h)
    # t5 = a*(e-g)
    # t6 = (c-a)*(e+f)
    # t7 = (b-d)*(g+h)
    
    # C21[0,0] = t1 - t4 + t5 - t7
    # C21[0,1] = t3 + t4
    # C21[1,0] = t2 + t5
    # C21[1,1] = t1 - t2 + t6 + t7
    
    # Map to U, V, W.
    # U row indices: i*m+j.
    # A2 is i=0,1 and j=1,2.
    # a: i=0, j=1 -> U[3]
    # b: i=0, j=2 -> U[4]
    # c: i=1, j=1 -> U[5]
    # d: i=1, j=2 -> U[6]
    
    # V row indices: j*p+k.
    # B21 is j=1,2 and k=0,1.
    # e: j=1, k=0 -> V[4]
    # f: j=1, k=1 -> V[5]
    # g: j=2, k=0 -> V[6]
    # h: j=2, k=1 -> V[7]
    
    # W row indices: k*n+i.
    # C21 is 2x2.
    # C21[0,0] -> k=0, i=0 -> W[0]
    # C21[0,1] -> k=1, i=0 -> W[2]
    # C21[1,0] -> k=0, i=1 -> W[1]
    # C21[1,1] -> k=1, i=1 -> W[3]
    
    # t1 = (a+d)*(e+h)
    U[3][col] = 1; U[6][col] = 1
    V[4][col] = 1; V[7][col] = 1
    W[0][col] = 1; W[3][col] = 1 # Coeff 1 for C[0,0] and C[1,1]
    col += 1
    
    # t2 = (c+d)*(e+f)
    U[5][col] = 1; U[6][col] = 1
    V[4][col] = 1; V[5][col] = 1
    W[1][col] = 1; W[3][col] = -1 # C[1,0] += 1, C[1,1] -= 1
    col += 1
    
    # t3 = (a+b)*(g+h)
    U[3][col] = 1; U[4][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[2][col] = 1 # C[0,1] += 1
    col += 1
    
    # t4 = d*(f-h)
    U[6][col] = 1
    V[5][col] = 1; V[7][col] = -1
    W[0][col] = -1; W[2][col] = 1 # C[0,0] -= 1, C[0,1] += 1
    col += 1
    
    # t5 = a*(e-g)
    U[3][col] = 1
    V[4][col] = 1; V[6][col] = -1
    W[0][col] = 1; W[1][col] = 1 # C[0,0] += 1, C[1,0] += 1
    col += 1
    
    # t6 = (c-a)*(e+f)
    U[5][col] = 1; U[3][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[3][col] = 1 # C[1,1] += 1
    col += 1
    
    # t7 = (b-d)*(g+h)
    U[4][col] = 1; U[6][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[0][col] = -1; W[3][col] = 1 # C[0,0] -= 1, C[1,1] += 1
    col += 1
    
    # Block 3: A2 @ B22
    # B22 = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]]
    # e'=B[1,2], f'=B[1,3], g'=B[2,2], h'=B[2,3]
    # V indices:
    # e': j=1, k=2 -> V[6]
    # f': j=1, k=3 -> V[7]
    # g': j=2, k=2 -> V[8]
    # h': j=2, k=3 -> V[9]
    
    # W indices for C22 (cols 2,3):
    # C22[0,0] -> C[0,2] -> k=2, i=0 -> W[4]
    # C22[0,1] -> C[0,3] -> k=3, i=0 -> W[6]
    # C22[1,0] -> C[1,2] -> k=2, i=1 -> W[5]
    # C22[1,1] -> C[1,3] -> k=3, i=1 -> W[7]
    
    # Same A2 coefficients.
    
    # t1' = (a+d)*(e'+h')
    U[3][col] = 1; U[6][col] = 1
    V[6][col] = 1; V[9][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1
    
    # t2' = (c+d)*(e'+f')
    U[5][col] = 1; U[6][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1
    
    # t3' = (a+b)*(g'+h')
    U[3][col] = 1; U[4][col] = 1
    V[8][col] = 1; V[9][col] = 1
    W[6][col] = 1
    col += 1
    
    # t4' = d*(f'-h')
    U[6][col] = 1
    V[7][col] = 1; V[9][col] = -1
    W[4][col] = -1; W[6][col] = 1
    col += 1
    
    # t5' = a*(e'-g')
    U[3][col] = 1
    V[6][col] = 1; V[8][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # t6' = (c-a)*(e'+f')
    U[5][col] = 1; U[3][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[7][col] = 1
    col += 1
    
    # t7' = (b-d)*(g'+h')
    U[4][col] = 1; U[6][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[4][col] = -1; W[7][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
