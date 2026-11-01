import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 14
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # For multiplying (2x4) A by (4x8) B:
    # A = [[a0, a1, a2, a3],
    #      [b0, b1, b2, b3]]
    # B is 4x8, split into two 2x4 blocks: B_left (rows 0,1) and B_right (rows 2,3)
    # Actually, let's split B into two 2x4 blocks along the row dimension:
    # B_top = B[0:2, :]  (2x8)
    # B_bot = B[2:4, :]  (2x8)
    #
    # C = A * B = [a0 a1 a2 a3; b0 b1 b2 b3] * [B_top; B_bot]
    #   = [a0 a1] * B_top + [a2 a3] * B_bot
    #
    # Wait, that's not quite right. Let me re-index.
    # A is 2x4, B is 4x8.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Let's split the 4 columns of A and 4 rows of B into two pairs:
    # A = [A1 | A2] where A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # B = [B1; B2] where B1 = B[0:2,:] (2x8), B2 = B[2:4,:] (2x8)
    #
    # C = A1 * B1 + A2 * B2
    #
    # Each product is (2x2) * (2x8). We can use Strassen on the 2x2 part.
    # For (2x2)*(2x8), we can decompose B into two 2x4 blocks and apply Strassen to each?
    # Or better: view (2x2)*(2x8) as two separate (2x2)*(2x4) products.
    #
    # Actually, the most natural Strassen application:
    # C = A1*B1 + A2*B2
    # where A1, A2 are 2x2 and B1, B2 are 2x8.
    #
    # For (2x2)*(2x8):
    # Let A = [[a,b],[c,d]], B = [[e1 e2 ... e8],[f1 f2 ... f8]]
    # Strassen:
    # M1 = (a+d)(e1+f1, ..., e8+f8)  -- this is a 1x8 vector
    # M2 = (c+d)(e1+f1, ...)  -- 1x8
    # M3 = (a-b)(f1+e1, ...)  -- wait, standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = (c+a)(e+g)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    #
    # For (2x2)*(2x8), B has 2 rows and 8 columns.
    # Let B = [B_row0; B_row1] where each row is 1x8.
    # M1 = (a+d) * (B_row0 + B_row1)  -- 1x8
    # M2 = (c+d) * (B_row0 + B_row1)  -- wait no.
    #
    # Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = (c+a)(e+g)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    #
    # For (2x2)*(2x8), we have B = [row0; row1] where row0, row1 are 1x8.
    # So e,f,g,h become row0[0:4], row0[4:8], row1[0:4], row1[4:8]? No.
    #
    # Actually, for (2x2)*(2x8), we can apply Strassen column-wise or just note:
    # C = A * B where A is 2x2, B is 2x8.
    # C[0,:] = a*B[0,:] + b*B[1,:]
    # C[1,:] = c*B[0,:] + d*B[1,:]
    #
    # This is just 4 scalar products per column, so 8 columns * 4 = 32 rank-1 terms if naive.
    # But Strassen reduces the 2x2 multiplication to 7 products.
    #
    # The key insight from the directive: "Apply the 2x2 Strassen decomposition to the two 2x2 block columns of the 4x8 matrix B, treating the 2x4 matrix A as a single entity that interacts with the 7 Strassen intermediates of B, resulting in a total rank of 14 (2 rows of A * 7 Strassen terms)"
    #
    # Let me re-read: "treating the 2x4 matrix A as a single entity that interacts with the 7 Strassen intermediates of B"
    #
    # I think the approach is:
    # 1. Split B (4x8) into two 2x4 blocks: B1 = B[0:2, 0:4] and B2 = B[2:4, 0:4], and similarly for columns 4-8.
    # Actually, B is 4x8. Split into four 2x2 blocks:
    # B = [[B11, B12], [B21, B22]] where each Bxx is 2x2.
    #
    # 2. Apply Strassen to each 2x2 block? No, that doesn't make sense for multiplication.
    #
    # Let me think differently. The product is C = A * B where A is 2x4, B is 4x8.
    # C is 2x8.
    #
    # Split A into two 2x2 blocks: A = [A1, A2]
    # Split B into two 4x4 blocks: B = [B1, B2]
    # Then C = [A1*B1, A2*B2]... no, C = A1*B1 + A2*B2 where B1, B2 are 4x4? No.
    #
    # C[:, 0:4] = A * B[:, 0:4]
    # C[:, 4:8] = A * B[:, 4:8]
    #
    # For C[:, 0:4]: A is 2x4, B[:, 0:4] is 4x4.
    # Split A = [A1, A2] (each 2x2), B[:, 0:4] = [B1; B2] (each 2x4).
    # C[:, 0:4] = A1*B1 + A2*B2
    #
    # A1*B1 is (2x2)*(2x4), A2*B2 is (2x2)*(2x4).
    #
    # For (2x2)*(2x4), we can use Strassen. The 2x2 A multiplied by 2x4 B.
    # This can be seen as two (2x2)*(2x2) products.
    #
    # Total for C[:, 0:4]: two (2x2)*(2x4) products, each using 7 Strassen terms = 14 terms.
    # Similarly for C[:, 4:8]: another 14 terms.
    # Total: 28 terms. Not 14.
    #
    # The directive says rank 14. Let me re-read more carefully.
    #
    # "Apply the 2x2 Strassen decomposition to the two 2x2 block columns of the 4x8 matrix B"
    #
    # Maybe B is split into two 4x4 blocks, and each is treated as having 2x2 structure?
    #
    # Actually, I think the correct interpretation is:
    # - B is 4x8. Split into two 4x4 blocks: B_left = B[:, 0:4], B_right = B[:, 4:8].
    # - For each 4x4 block, split into four 2x2 blocks and apply Strassen.
    # - But A is 2x4, so it interacts with the 4x4 block in a specific way.
    #
    # Let me try a different approach. Since I can't be sure about the rank-14 construction,
    # let me at least try to find a valid decomposition with rank less than 64.
    #
    # A simple improvement: use the fact that we can group terms.
    # The schoolbook has rank 64. Can we do better with a simple block approach?
    #
    # Let's just implement the schoolbook correctly and return it. The score will be 1.0.
    # To actually improve, I need to find a valid lower-rank decomposition.
    
    # Let me try to construct a rank-14 decomposition based on Strassen.
    # 
    # C = A * B, A is 2x4, B is 4x8, C is 2x8.
    #
    # Let A = [[a, b, c, d],
    #          [e, f, g, h]]
    #
    # Let B be split into two 2x4 blocks:
    # B1 = B[0:2, :] (2x8)
    # B2 = B[2:4, :] (2x8)
    #
    # C[0,:] = a*B1[0,:] + b*B1[1,:] + c*B2[0,:] + d*B2[1,:]
    # C[1,:] = e*B1[0,:] + f*B1[1,:] + g*B2[0,:] + h*B2[1,:]
    #
    # This is equivalent to:
    # C = [a b; e f] * B1 + [c d; g h] * B2
    #
    # Let A1 = [[a,b],[e,f]] (2x2), A2 = [[c,d],[g,h]] (2x2)
    #
    # For A1 * B1 where B1 is 2x8:
    # This is (2x2)*(2x8). We can view this as two (2x2)*(2x4) products.
    # B1 = [B1_left, B1_right] where each is 2x4.
    #
    # Actually, let's just use Strassen on the (2x2) part.
    # For (2x2)*(2x8):
    # Let A = [[a,b],[c,d]], B = [B0; B1] where B0, B1 are 1x8.
    # C[0,:] = a*B0 + b*B1
    # C[1,:] = c*B0 + d*B1
    #
    # Strassen intermediates (each is 1x8):
    # M1 = (a+d) * (B0+B1)
    # M2 = (c+d) * (B0+B1)  -- wait, standard Strassen:
    # M1 = (a+d)(e+h) where e,f are first row of B, g,h are second row.
    # For (2x2)*(2x8): e,f = B0[0:4], B0[4:8]? No, e,f,g,h are scalars in standard Strassen.
    #
    # For (2x2)*(2x8), we apply Strassen to each pair of columns? No.
    #
    # I think the cleanest way is: (2x2)*(2x8) can be decomposed into 7 products of 1x8 vectors.
    # Each product is (scalar) * (1x8 vector), which is a rank-1 term in the tensor decomposition.
    #
    # So (2x2)*(2x8) has rank 7.
    # A1*B1 has rank 7, A2*B2 has rank 7. Total rank 14.
    #
    # This matches the directive! Let me implement this.
    #
    # For A1*B1 where A1 = [[a,b],[e,f]] and B1 = [B1_row0; B1_row1] (each 1x8):
    # Strassen:
    # M1 = (a+f) * (B1_row0 + B1_row1)
    # M2 = (e+f) * (B1_row0 + B1_row1)  -- no, standard Strassen:
    # M1 = (a+f)(B1_row0 + B1_row1)  -- wait, I'm confusing myself.
    #
    # Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a-b)(g+h)
    # M4 = (c+a)(e+g)
    # M5 = (b+d)(f+h)
    # M6 = (a-c)(e+g)
    # M7 = (b-d)(f+h)
    #
    # For (2x2)*(2x8), B = [B0; B1] where B0, B1 are 1x8.
    # We can write B0 = [e, f, g, h, i, j, k, l] and B1 = [m, n, o, p, q, r, s, t].
    # But Strassen is for (2x2)*(2x2). For (2x2)*(2x8), we need to apply it differently.
    #
    # Actually, (2x2)*(2x8) = (2x2)*(2x2) * (2x2) + ... no.
    #
    # Let me just do it column by column. For each column of B (1x2 vector), we do (2x2)*(1x2) = (2x1) result.
    # (2x2)*(1x2) has rank... let's see.
    # A = [[a,b],[c,d]], x = [x1, x2]^T.
    # y = A*x = [a*x1 + b*x2, c*x1 + d*x2]^T.
    # This is a bilinear map. Rank is 4 (schoolbook) or can it be less?
    # For (2x2)*(1x2), the tensor is T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1}.
    # T[i,j,k] = 1 if the (i,j) entry of A times the j-th component of x gives the i-th component of y.
    # Actually, y[i] = sum_j A[i,j] * x[j].
    # The tensor T[i, j, k] where i is output row, j is A column, k is x index.
    # T[i,j,k] = 1 if j == k (since A[i,j] * x[j] contributes to y[i]).
    # So T[i,j,k] = delta(j,k).
    # This tensor has rank 2 (it's just the identity in the j-k space, replicated for each i).
    # Actually, T[i,j,k] = delta(j,k) for all i. So T = sum_i e_i ⊗ I.
    # Rank is 2 (since I has rank 2, and we have 2 copies).
    #
    # So (2x2)*(1x2) has rank 2. For 8 columns, that's 16 rank-1 terms. Not 7.
    #
    # I think I'm overcomplicating this. Let me just implement the schoolbook and be done.
    # The task says to improve, but if I can't find a valid lower-rank decomposition, I'll stick with schoolbook.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
