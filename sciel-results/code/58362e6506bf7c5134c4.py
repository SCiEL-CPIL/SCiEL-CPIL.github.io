"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # We use a hybrid approach:
    # The matrix multiplication tensor is M_{n,m,p}.
    # We can decompose the 'm' dimension (inner dimension) into blocks if we can align the block sizes with the 'n' and 'p' dimensions appropriately?
    # No, standard Kronecker product logic:
    # If M_{n1,m1,p1} has rank R1 and M_{n2,m2,p2} has rank R2, then M_{n1*n2, m1*m2, p1*p2} has rank R1*R2.
    # Our dimensions are 3, 4, 8.
    # 3 is prime. 4 = 2*2. 8 = 2*2*2.
    # We can decompose M_{3,4,8} into M_{3,2,4} \otimes M_{1,2,2}? No, dimensions must match.
    # Let's check if we can factor the dimensions.
    # M_{3,4,8}. Can we write 3=3*1, 4=2*2, 8=2*4?
    # Then M_{3,4,8} = M_{3,2,2} \otimes M_{1,2,4}?
    # Rank(M_{3,2,2}) * Rank(M_{1,2,4})
    # Rank(M_{1,2,4}) = 2*4 = 8 (since n=1, it's just linear combos, rank is m*p).
    # Rank(M_{3,2,2}): Schoolbook is 3*2*2=12. Is there better?
    # Lower bound for M_{3,2,2} is max(n*m, m*p, p*n) = max(6, 4, 6) = 6.
    # Actually, for M_{n,m,p}, the border rank is at least n*m*p? No.
    # For M_{2,2,2}, rank is 7 (Strassen).
    # For M_{3,2,2}, can we do better than 12?
    # 12 is schoolbook.
    # Let's try to use Strassen on the 4x8 part?
    # 4x8 = 2x4 * 4x8? No, middle dimension is 4.
    # Let's look at the factors:
    # 3 = 3
    # 4 = 2 * 2
    # 8 = 4 * 2
    # So M_{3,4,8} can be seen as M_{3, 2*2, 4*2}.
    # This doesn't form a clean tensor product of two smaller tensors unless the dimensions align as (n1*n2, m1*m2, p1*p2).
    # Here: n1*n2=3, m1*m2=4, p1*p2=8.
    # Possible splits:
    # 1. n1=3, n2=1; m1=4, m2=1; p1=8, p2=1 -> M_{3,4,8} \otimes M_{1,1,1} (useless)
    # 2. n1=3, n2=1; m1=2, m2=2; p1=4, p2=2 -> M_{3,2,4} \otimes M_{1,2,2}
    #    Rank = Rank(M_{3,2,4}) * Rank(M_{1,2,2}).
    #    Rank(M_{1,2,2}) = 2*2 = 4.
    #    Rank(M_{3,2,4})? Schoolbook 3*2*4=24.
    #    Total = 24 * 4 = 96. (Same as schoolbook 96)
    # 3. n1=1, n2=3; m1=2, m2=2; p1=8, p2=1 -> M_{1,2,8} \otimes M_{3,2,1}
    #    Rank(M_{1,2,8}) = 16.
    #    Rank(M_{3,2,1}) = 6.
    #    Total = 96.
    # 4. Can we use Strassen on M_{3,2,4}?
    #    3 is not a power of 2.
    #    However, we can decompose M_{3,2,4} into M_{1,2,4} + M_{2,2,4}? No, tensor addition doesn't add ranks linearly like that for decomposition.
    #    But we can pad M_{2,2,4} to M_{3,2,4} by adding a zero row?
    #    If we compute M_{2,2,4} with Strassen-like logic?
    #    M_{2,2,4} can be decomposed into M_{2,2,2} \otimes M_{1,1,2}.
    #    Rank(M_{2,2,2}) = 7.
    #    Rank(M_{1,1,2}) = 2.
    #    Rank(M_{2,2,4}) = 14.
    #    Then M_{3,2,4} contains M_{2,2,4}. We can compute the 2x2 block with rank 14, and the remaining 1x2 block with rank 2*4=8?
    #    Wait, the third row of A (3x2) multiplies B (2x4). This is a separate set of products.
    #    The tensor M_{3,2,4} = M_{1,2,4} + M_{2,2,4} (disjoint supports in the A-index).
    #    Rank(M_{3,2,4}) = Rank(M_{1,2,4}) + Rank(M_{2,2,4}) = 8 + 14 = 22.
    #    So for split 2: Rank(M_{3,2,4}) = 22.
    #    Total Rank = 22 * Rank(M_{1,2,2}) = 22 * 4 = 88.
    #    
    #    Let's verify this logic.
    #    M_{3,4,8} = M_{3, 2*2, 4*2} = M_{3,2,4} \otimes M_{1,2,2}.
    #    We need to check if the Kronecker product of decompositions works.
    #    Yes, if A = A1 \otimes A2, B = B1 \otimes B2, then AB = (A1B1) \otimes (A2B2).
    #    The decomposition of M_{3,2,4} into rank 22 and M_{1,2,2} into rank 4 yields a decomposition of M_{3,4,8} into rank 88.
    #    
    #    Is 88 < 96? Yes.
    #    
    #    Can we do better?
    #    What if we decompose M_{3,4,8} as M_{3,4,4} \otimes M_{1,1,2}?
    #    Rank(M_{1,1,2}) = 2.
    #    Rank(M_{3,4,4})?
    #    M_{3,4,4} = M_{3, 2*2, 2*2} = M_{3,2,2} \otimes M_{1,2,2}.
    #    Rank(M_{1,2,2}) = 4.
    #    Rank(M_{3,2,2})?
    #    M_{3,2,2} = M_{1,2,2} + M_{2,2,2}.
    #    Rank(M_{1,2,2}) = 4.
    #    Rank(M_{2,2,2}) = 7.
    #    Rank(M_{3,2,2}) = 4 + 7 = 11.
    #    So Rank(M_{3,4,4}) = 11 * 4 = 44.
    #    Total Rank = 44 * 2 = 88.
    #    Same rank.
    
    #    Is there a better decomposition for M_{3,2,2}?
    #    Schoolbook is 12. Strassen-based is 11.
    #    Lower bound for M_{3,2,2} is max(6,4,6)=6.
    #    Is 10 possible?
    #    If Rank(M_{3,2,2})=10, then Rank(M_{3,4,8}) = 10 * 4 * 2 = 80.
    #    
    #    Let's try to construct the rank 88 decomposition.
    #    It is safer than trying to find a non-standard low-rank decomposition for M_{3,2,2} which might not exist or is hard to find.
    #    
    #    Construction of M_{3,4,8} with rank 88:
    #    We use the decomposition:
    #    M_{3,4,8} = M_{3,4,4} \otimes M_{1,1,2}
    #    Where M_{3,4,4} has rank 44 and M_{1,1,2} has rank 2.
    #    
    #    Wait, M_{1,1,2} is multiplying a 1x1 matrix by a 1x2 matrix?
    #    Dimensions:
    #    A is 3x4, B is 4x8.
    #    Split B into B_left (4x4) and B_right (4x4)? No, 4x8 is (4x4) horizontally?
    #    Kronecker product M_{n,m,p} \otimes M_{n',m',p'} corresponds to:
    #    A (n x m), B (m x p).
    #    A' (n' x m'), B' (m' x p').
    #    The product is (A \otimes A') (B \otimes B') = (AB) \otimes (A'B').
    #    The dimensions of the result are (n*n') x (p*p').
    #    The middle dimension is m*m'.
    #    So for M_{3,4,8}, we need n1*n2=3, m1*m2=4, p1*p2=8.
    #    
    #    Option A:
    #    n1=3, m1=4, p1=4 -> M_{3,4,4}
    #    n2=1, m2=1, p2=2 -> M_{1,1,2}
    #    Check: n=3*1=3, m=4*1=4, p=4*2=8. Correct.
    #    
    #    We need decompositions for M_{3,4,4} (rank 44) and M_{1,1,2} (rank 2).
    #    
    #    Decomposition of M_{1,1,2}:
    #    A is 1x1 [a], B is 1x2 [b1, b2].
    #    C is 1x2 [ab1, ab2].
    #    Tensor: C[i,k] = sum_j A[i,j] B[j,k].
    #    Indices: i in {0}, j in {0}, k in {0,1}.
    #    T[i,j,k] = A[i,j] B[j,k].
    #    Decomposition:
    #    Rank 2.
    #    U (size 1*1 x 2 = 1x2):
    #    U[0,0]=1, U[0,1]=1
    #    V (size 1*2 x 2 = 2x2):
    #    V[0,0]=1, V[0,1]=0 (B[j,k] for j=0,k=0) -> 1
    #    V[1,0]=0, V[1,1]=1 (B[j,k] for j=0,k=1) -> 1
    #    Wait, V is indexed by j*p+k? No, V is m*p. Here m=1, p=2.
    #    V row index: j*p+k = j*2+k.
    #    V[0,0]=1, V[0,1]=0?
    #    Let's check the formula:
    #    C[i,k] = sum_r U[i,m? No, U is n*m. V is m*p. W is p*n.
    #    Wait, the interface is:
    #    U: (n*m, rank)
    #    V: (m*p, rank)
    #    W: (p*n, rank)
    #    And W[k*n+i, r].
    #    
    #    Let's re-verify the indices for M_{1,1,2}.
    #    n=1, m=1, p=2.
    #    U: 1*1 = 1 row.
    #    V: 1*2 = 2 rows.
    #    W: 2*1 = 2 rows. W[k*1+i, r] -> W[k, r].
    #    
    #    Rank 2 decomposition:
    #    r=0:
    #    U[0,0] = 1
    #    V[0,0] = 1 (j=0,k=0)
    #    V[1,0] = 0 (j=0,k=1)
    #    W[0,0] = 1 (k=0,i=0)
    #    W[1,0] = 0 (k=1,i=0)
    #    
    #    r=1:
    #    U[0,1] = 1
    #    V[0,1] = 0
    #    V[1,1] = 1 (j=0,k=1)
    #    W[0,1] = 0
    #    W[1,1] = 1 (k=1,i=0)
    #    
    #    Check:
    #    C[0,0] = sum_r U[0,r] * (V * ? )
    #    The tensor is T[i,j,k] = delta(i,j) * delta(j,0) * delta(k, k)? No.
    #    T[i,j,k] is 1 if i=0, j=0, k=0? No, T is the multiplication tensor.
    #    T[i,j,k] = 1 if the product A[i,j]B[j,k] contributes to C[i,k].
    #    Actually, the tensor T has entries T[i,j,k] = 1 if i,j,k are valid indices for the term A[i,j]B[j,k].
    #    But in the decomposition, we sum U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    #    For M_{1,1,2}:
    #    i=0, j=0. k=0,1.
    #    r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1. Product = 1. This corresponds to A[0,0]B[0,0] -> C[0,0].
    #    r=1: U[0,1]=1, V[1,1]=1, W[1,1]=1. Product = 1. This corresponds to A[0,0]B[0,1] -> C[0,1].
    #    This works.
    
    #    Now M_{3,4,4}.
    #    n=3, m=4, p=4.
    #    Decompose M_{3,4,4} = M_{3,2,2} \otimes M_{1,2,2}.
    #    n1=3, m1=2, p1=2.
    #    n2=1, m2=2, p2=2.
    #    n=3*1=3, m=2*2=4, p=2*2=4.
    #    
    #    Rank(M_{3,2,2}) = 11.
    #    Rank(M_{1,2,2}) = 4.
    #    Total Rank = 44.
    #    
    #    We need decompositions for M_{3,2,2} (rank 11) and M_{1,2,2} (rank 4).
    #    
    #    M_{1,2,2} (n=1, m=2, p=2):
    #    Schoolbook rank 4.
    #    U (1*2=2 rows), V (2*2=4 rows), W (2*1=2 rows).
    #    r=0: A[0,0]B[0,0] -> C[0,0]. U[0,0]=1, U[1,0]=0. V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0. W[0,0]=1, W[1,0]=0.
    #    r=1: A[0,0]B[0,1] -> C[0,1]. U[0,1]=1, U[1,1]=0. V[0,1]=0, V[1,1]=1, V[2,1]=0, V[3,1]=0. W[0,1]=0, W[1,1]=1.
    #    r=2: A[0,1]B[1,0] -> C[0,0]. U[0,2]=0, U[1,2]=1. V[0,2]=0, V[1,2]=0, V[2,2]=1, V[3,2]=0. W[0,2]=1, W[1,2]=0.
    #    r=3: A[0,1]B[1,1] -> C[0,1]. U[0,3]=0, U[1,3]=1. V[0,3]=0, V[1,3]=0, V[2,3]=0, V[3,3]=1. W[0,3]=0, W[1,3]=1.
    
    #    M_{3,2,2} (n=3, m=2, p=2):
    #    Decompose into M_{1,2,2} (first row) + M_{2,2,2} (last 2 rows).
    #    Rank 4 + 7 = 11.
    #    
    #    M_{1,2,2} part (rows 0 of A):
    #    Indices for M_{3,2,2}:
    #    U rows: 0..5 (3*2).
    #    V rows: 0..3 (2*2).
    #    W rows: 0..5 (2*3). W[k*3+i].
    #    
    #    Part 1: i=0.
    #    Uses M_{1,2,2} decomposition shifted to i=0.
    #    r=0..3:
    #    U[0,r]=U_part[0,r], U[1,r]=U_part[1,r]. Other U rows 0.
    #    V same.
    #    W[k*3+0, r] = W_part[k, r]. Other W rows 0.
    #    
    #    Part 2: i=1,2.
    #    Uses M_{2,2,2} decomposition (Strassen) shifted to i=1,2.
    #    Strassen for M_{2,2,2}:
    #    n=2, m=2, p=2.
    #    Rank 7.
    #    U (4x7), V (4x7), W (4x7).
    #    
    #    We need to map the local indices (i_loc, j_loc, k_loc) to global indices.
    #    i_global = 1 + i_loc.
    #    j_global = j_loc.
    #    k_global = k_loc.
    #    
    #    U_global[(1+i_loc)*2 + j_loc, r] = U_strassen[i_loc*2 + j_loc, r].
    #    V_global[j_loc*2 + k_loc, r] = V_strassen[j_loc*2 + k_loc, r].
    #    W_global[k_loc*3 + (1+i_loc), r] = W_strassen[k_loc*2 + i_loc, r].
    
    #    Let's hardcode the Strassen decomposition for M_{2,2,2}.
    #    U (4x7):
    #    A[0,0] A[0,1]
    #    A[1,0] A[1,1]
    #    
    #    M1 = A00(B00+B11) -> U: A00=1. V: B00=1, B11=1.
    #    M2 = (A10+A11)B00 -> U: A10=1, A11=1. V: B00=1.
    #    M3 = (A00+A01)B11 -> U: A00=1, A01=1. V: B11=1.
    #    M4 = A11(B10-B00) -> U: A11=1. V: B10=1, B00=-1.
    #    M5 = A00(B01-B11) -> U: A00=1. V: B01=1, B11=-1.
    #    M6 = (A10-A00)(B00+B01) -> U: A10=1, A00=-1. V: B00=1, B01=1.
    #    M7 = (A01-A11)(B10+B11) -> U: A01=1, A11=-1. V: B10=1, B11=1.
    
    #    W (4x7) for C:
    #    C00 = M1 + M4 - M5 + M7
    #    C01 = M3 + M5
    #    C10 = M2 + M4
    #    C11 = M1 - M2 + M3 + M6
    
    #    W indices: W[k*2+i].
    #    k=0, i=0 (C00): M1:1, M4:1, M5:-1, M7:1
    #    k=0, i=1 (C01): M3:1, M5:1
    #    k=1, i=0 (C10): M2:1, M4:1
    #    k=1, i=1 (C11): M1:1, M2:-1, M3:1, M6:1

    #    Now we build the full 88-rank decomposition.
    
    #    Step 1: Build M_{3,2,2} (Rank 11)
    #    n=3, m=2, p=2.
    #    U322 (6x11), V322 (4x11), W322 (6x11).
    
    U322 = [[0]*11 for _ in range(6)]
    V322 = [[0]*11 for _ in range(4)]
    W322 = [[0]*11 for _ in range(6)]
    
    # Part 1: i=0 (M_{1,2,2})
    # U: 2x4, V: 4x4, W: 2x4
    # r=0: A00 B00 -> C00
    U322[0][0] = 1
    V322[0][0] = 1
    W322[0][0] = 1 # k=0, i=0
    
    # r=1: A00 B01 -> C01
    U322[0][1] = 1
    V322[1][1] = 1
    W322[1][1] = 1 # k=1, i=0
    
    # r=2: A01 B10 -> C00
    U322[1][2] = 1
    V322[2][2] = 1
    W322[0][2] = 1 # k=0, i=0
    
    # r=3: A01 B11 -> C01
    U322[1][3] = 1
    V322[3][3] = 1
    W322[1][3] = 1 # k=1, i=0
    
    # Part 2: i=1,2 (M_{2,2,2} Strassen)
    # Offset i by 1.
    # U_strassen (4x7) -> U322 rows 2,3,4,5
    # V_strassen (4x7) -> V322 rows 0,1,2,3
    # W_strassen (4x7) -> W322 rows 0..5 (k*3 + i)
    
    # M1: U[0,0]=1, V[0,0]=1, V[3,0]=1, W[0,0]=1, W[3,0]=1 (C00, C11)
    # Wait, W for Strassen:
    # C00 (k=0,i=0): M1:1, M4:1, M5:-1, M7:1
    # C01 (k=0,i=1): M3:1, M5:1
    # C10 (k=1,i=0): M2:1, M4:1
    # C11 (k=1,i=1): M1:1, M2:-1, M3:1, M6:1
    
    # Let's map Strassen columns 0..6 to global columns 4..10.
    
    # Col 4 (M1):
    U322[2][4] = 1 # A00
    V322[0][4] = 1 # B00
    V322[3][4] = 1 # B11
    W322[0][4] = 1 # C00 (k=0,i=1) -> idx 1? No, W322 index is k*3+i.
    # For i=1, k=0 -> idx 1.
    # For i=2, k=0 -> idx 2? No, i=1 is local i=0. i=2 is local i=1.
    # Local i=0 -> Global i=1. Local i=1 -> Global i=2.
    # W322[k*3 + i_global].
    # C00 (local): k=0, i_loc=0 -> i_g=1 -> idx 1.
    # C11 (local): k=1, i_loc=1 -> i_g=2 -> idx 5.
    W322[1][4] = 1
    W322[5][4] = 1
    
    # Col 5 (M2):
    U322[4][5] = 1 # A10
    U322[5][5] = 1 # A11
    V322[0][5] = 1 # B00
    W322[3][5] = 1 # C10 (k=1,i_loc=0 -> i_g=1 -> idx 4? No, k*3+i. k=1, i=1 -> 4.)
    W322[4][5] = 1
    W322[5][5] = -1 # C11 (k=1,i_loc=1 -> i_g=2 -> idx 5)
    
    # Col 6 (M3):
    U322[2][6] = 1 # A00
    U322[3][6] = 1 # A01
    V322[3][6] = 1 # B11
    W322[1][6] = 1 # C01 (k=0,i_loc=1 -> i_g=2 -> idx 2)
    W322[2][6] = 1
    W322[5][6] = 1 # C11
    
    # Col 7 (M4):
    U322[5][7] = 1 # A11
    V322[2][7] = 1 # B10
    V322[0][7] = -1 # B00
    W322[1][7] = 1 # C00
    W322[4][7] = 1 # C10
    
    # Col 8 (M5):
    U322[2][8] = 1 # A00
    V322[1][8] = 1 # B01
    V322[3][8] = -1 # B11
    W322[1][8] = -1 # C00
    W322[2][8] = 1 # C01
    
    # Col 9 (M6):
    U322[4][9] = 1 # A10
    U322[2][9] = -1 # A00
    V322[0][9] = 1 # B00
    V322[1][9] = 1 # B01
    W322[5][9] = 1 # C11
    
    # Col 10 (M7):
    U322[3][10] = 1 # A01
    U322[5][10] = -1 # A11
    V322[2][10] = 1 # B10
    V322[3][10] = 1 # B11
    W322[1][10] = 1 # C00
    
    # Step 2: Build M_{1,2,2} (Rank 4)
    # n=1, m=2, p=2.
    U122 = [[0]*4 for _ in range(2)]
    V122 = [[0]*4 for _ in range(4)]
    W122 = [[0]*4 for _ in range(2)]
    
    # r=0: A00 B00 -> C00
    U122[0][0] = 1
    V122[0][0] = 1
    W122[0][0] = 1
    
    # r=1: A00 B01 -> C01
    U122[0][1] = 1
    V122[1][1] = 1
    W122[1][1] = 1
    
    # r=2: A01 B10 -> C00
    U122[1][2] = 1
    V122[2][2] = 1
    W122[0][2] = 1
    
    # r=3: A01 B11 -> C01
    U122[1][3] = 1
    V122[3][3] = 1
    W122[1][3] = 1

    # Step 3: Build M_{1,1,2} (Rank 2)
    # n=1, m=1, p=2.
    U112 = [[0]*2 for _ in range(1)]
    V112 = [[0]*2 for _ in range(2)]
    W112 = [[0]*2 for _ in range(2)]
    
    # r=0: A00 B00 -> C00
    U112[0][0] = 1
    V112[0][0] = 1
    W112[0][0] = 1
    
    # r=1: A00 B01 -> C01
    U112[0][1] = 1
    V112[1][1] = 1
    W112[1][1] = 1

    # Step 4: Combine M_{3,4,4} = M_{3,2,2} \otimes M_{1,2,2}
    # Rank = 11 * 4 = 44.
    U344 = [[0]*44 for _ in range(12)] # 3*4
    V344 = [[0]*44 for _ in range(16)] # 4*4
    W344 = [[0]*44 for _ in range(12)] # 4*3
    
    # Indices:
    # U344[(i1*2 + i2)*4 + (j1*2 + j2)] ?
    # No, Kronecker product index mapping:
    # U is n*m.
    # i = i1*n2 + i2?
    # n1=3, n2=1. i = i1*1 + i2 = i1 + i2. i2 in {0}. So i = i1.
    # Wait, A = A1 \otimes A2.
    # A1 is n1 x m1. A2 is n2 x m2.
    # A is (n1*n2) x (m1*m2).
    # A[i1*n2 + i2, j1*m2 + j2] = A1[i1,j1] * A2[i2,j2].
    # Here n1=3, n2=1. m1=2, m2=2.
    # A[i1, j1*2 + j2] = A1[i1, j1] * A2[0, j2].
    # So U344 row index: i = i1. col = j1*2 + j2.
    # But U is flattened i*m+j.
    # m=4.
    # U344[i1*4 + (j1*2 + j2), r]
    
    # r = r1*4 + r2.
    # U344[...] = U322[i1*2 + j1, r1] * U122[0*2 + j2, r2]
    
    # V344:
    # V is m*p.
    # V[j, k].
    # j = j1*m2 + j2 = j1*2 + j2.
    # k = k1*p2 + k2 = k1*2 + k2.
    # V344[(j1*2 + j2)*4 + (k1*2 + k2), r]
    # = V322[j1*2 + k1, r1] * V122[j2*2 + k2, r2]
    
    # W344:
    # W is p*n.
    # W[k*n + i].
    # k = k1*p2 + k2 = k1*2 + k2.
    # i = i1*n2 + i2 = i1.
    # W344[(k1*2 + k2)*3 + i1, r]
    # = W322[k1*3 + i1, r1] * W122[k2*1 + 0, r2] = W322[k1*3 + i1, r1] * W122[k2, r2]
    
    for r1 in range(11):
        for r2 in range(4):
            r = r1*4 + r2
            for i1 in range(3):
                for j1 in range(2):
                    for j2 in range(2):
                        u1 = U322[i1*2 + j1][r1]
                        u2 = U122[j2][r2]
                        if u1 != 0 and u2 != 0:
                            idx_u = i1*4 + (j1*2 + j2)
                            U344[idx_u][r] += u1 * u2
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            v1 = V322[j1*2 + k1][r1]
                            v2 = V122[j2*2 + k2][r2]
                            if v1 != 0 and v2 != 0:
                                idx_v = (j1*2 + j2)*4 + (k1*2 + k2)
                                V344[idx_v][r] += v1 * v2
            for k1 in range(2):
                for i1 in range(3):
                    for k2 in range(2):
                        w1 = W322[k1*3 + i1][r1]
                        w2 = W122[k2][r2]
                        if w1 != 0 and w2 != 0:
                            idx_w = (k1*2 + k2)*3 + i1
                            W344[idx_w][r] += w1 * w2

    # Step 5: Combine M_{3,4,8} = M_{3,4,4} \otimes M_{1,1,2}
    # Rank = 44 * 2 = 88.
    U = [[0]*88 for _ in range(12)]
    V = [[0]*88 for _ in range(32)]
    W = [[0]*88 for _ in range(24)]
    
    # n1=3, n2=1. m1=4, m2=1. p1=4, p2=2.
    # A[i1*1 + i2, j1*1 + j2] = A1[i1,j1] * A2[i2,j2].
    # i = i1. j = j1.
    # U[i*4 + j, r] where r = r1*2 + r2.
    # U344[i1*4 + j1, r1] * U112[0, r2]
    
    # V:
    # j = j1*1 + j2 = j1. (Wait, m2=1, so j2=0).
    # k = k1*2 + k2.
    # V[(j1)*8 + (k1*2 + k2), r]
    # = V344[j1*4 + k1, r1] * V112[0*2 + k2, r2]
    
    # W:
    # k = k1*2 + k2.
    # i = i1.
    # W[(k1*2 + k2)*3 + i1, r]
    # = W344[k1*3 + i1, r1] * W112[k2, r2]
    
    for r1 in range(44):
        for r2 in range(2):
            r = r1*2 + r2
            for i1 in range(3):
                for j1 in range(4):
                    u1 = U344[i1*4 + j1][r1]
                    u2 = U112[0][r2]
                    if u1 != 0 and u2 != 0:
                        idx_u = i1*4 + j1
                        U[idx_u][r] += u1 * u2
            for j1 in range(4):
                for k1 in range(4):
                    for k2 in range(2):
                        v1 = V344[j1*4 + k1][r1]
                        v2 = V112[k2][r2]
                        if v1 != 0 and v2 != 0:
                            idx_v = j1*8 + (k1*2 + k2)
                            V[idx_v][r] += v1 * v2
            for k1 in range(4):
                for i1 in range(3):
                    for k2 in range(2):
                        w1 = W344[k1*3 + i1][r1]
                        w2 = W112[k2][r2]
                        if w1 != 0 and w2 != 0:
                            idx_w = (k1*2 + k2)*3 + i1
                            W[idx_w][r] += w1 * w2

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
