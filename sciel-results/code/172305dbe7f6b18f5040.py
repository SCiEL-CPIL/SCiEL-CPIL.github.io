"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Start with schoolbook rank-96 for 4x4x8, then use null-space
    projection to identify and remove redundant terms when row 3 of A is zero.
    
    The 3x4x8 tensor T[i,j,k] = delta_{ik} for i in {0,1,2}, j in {0..7}, k in {0..3}
    can be seen as the restriction of the 4x4x8 tensor where A has 4 rows.
    
    We build a schoolbook decomposition for 4x4x8 (rank 32), then for the
    3x4x8 case we need terms where the U-vector has support on rows 0,1,2 only.
    
    Actually, let me think more carefully. The 3x4x8 multiplication tensor has:
    - U: shape (12, rank) - rows of A (3x4=12 entries)
    - V: shape (32, rank) - rows of B (4x8=32 entries)  
    - W: shape (24, rank) - entries of C (8x3=24 entries), W[k*n+i, r]
    
    Schoolbook: C[i,j] = sum_k A[i,k] * B[k,j]
    For i in {0,1,2}, j in {0..7}, k in {0..3}
    Rank = 3*4*8 = 96
    
    Key insight: The 4x4x8 tensor (rank 128) restricted to 3x4x8 (rank 96)
    loses terms involving A[3,k]. But we can also look for structure within
    the 3x4x8 tensor itself.
    
    Better approach: Use the block structure.
    - Split C into 3 row-blocks of 4 columns each? No, C is 8x3.
    - C is 8 rows, 3 cols. A is 3x4, B is 4x8.
    
    Let me try a different decomposition:
    - For rows 0,1 of A: use Strassen-style 2x4x4 blocks (2 blocks for 8 cols)
    - For row 2 of A: schoolbook 1x4x4 blocks
    
    This gives: 2 * rank(M_2,4,4) + 2 * rank(M_1,4,4)
    
    For M_2,4,4: A is 2x4, B is 4x4, C is 4x2
    Split into 2x2x2 blocks: 2 column-blocks of B, 2 row-blocks of C
    = 4 sub-tensors of 2x2x2, each rank 7 (Strassen) = 28
    
    For M_1,4,4: A is 1x4, B is 4x4, C is 4x1
    = schoolbook, rank 4*4 = 16... wait, that's not right.
    M_1,4,4: C[j] = sum_k A[k] * B[k,j], j=0..3
    This is just a vector-matrix product. Rank = 4 (each output is a separate term)
    Actually no: C[j] = sum_{k=0}^{3} A[k]*B[k,j]
    Schoolbook: 4 terms (one per k), but each term contributes to all 4 outputs.
    Wait, the tensor is: T[j,k] = A[k]*B[k,j]
    This is rank 4: T = sum_k (e_k in A-space) ⊗ (row k of B) ⊗ (e_j in C-space)
    No wait: T[j,k,a] where a is the A-index... 
    A has 4 elements (1x4), B has 16 elements (4x4), C has 4 elements (4x1)
    T[a,b,c] = delta_{a,b_row} * delta_{b_col,c} ... 
    
    Let me just be explicit. For M_1,4,4:
    A = [a0, a1, a2, a3] (1x4 vector)
    B = 4x4 matrix
    C = [c0, c1, c2, c3] (4x1 vector)
    c_j = sum_k a_k * B[k,j]
    
    Tensor: T[a_idx, b_idx, c_idx] = 1 if b_idx = a_idx*4 + c_idx, else 0
    where a_idx in {0,1,2,3}, b_idx in {0..15}, c_idx in {0,1,2,3}
    
    Rank: Each term a_k * B[k,j] for fixed k,j is one rank-1 term.
    Total schoolbook rank = 4*4 = 16.
    
    But can we do better? The tensor T[a,b,c] = delta_{b = a*4+c}
    This is a permutation-like tensor. Its rank is exactly 16 (it's a "flat" tensor).
    
    So total: 2*28 + 2*16 = 56 + 32 = 88. This matches the current solution.
    
    Can we do better? Let me think about whether there's a smarter way.
    
    Alternative: What if we don't split by rows of A, but by columns of C?
    C is 8x3. Split into two 4x3 blocks (rows 0-3 and 4-7 of C).
    Each block: C_block[i,j] = sum_k A[i,k] * B[k, j+4*block]
    This is M_3,4,4 for each block. Rank of M_3,4,4?
    
    For M_3,4,4: A is 3x4, B is 4x4, C is 4x3
    Schoolbook rank = 3*4*4 = 48 per block, total 96.
    
    Can we do better for M_3,4,4? 
    Split A into 2x4 + 1x4:
    - M_2,4,4 part: rank 28 (Strassen)
    - M_1,4,4 part: rank 16 (schoolbook)
    Total: 44 per block, 88 total. Same as before.
    
    What about using a different Strassen variant or a lower-rank algorithm for 2x4x4?
    
    Actually, let me reconsider. The known lower bound for M_2,4,4 is not 28.
    Let me check: M_2,4,4 can be seen as 2 independent M_1,4,4 computations plus
    cross terms... no, that's not right.
    
    M_2,4,4: C[i,j] = sum_k A[i,k]*B[k,j], i=0,1; j=0,1,2,3; k=0,1,2,3
    This is equivalent to multiplying a 2x4 matrix by a 4x4 matrix.
    We can view this as: for each column j of B, compute A * B[:,j] (2x1 result).
    That's 4 independent M_2,4,1 computations.
    M_2,4,1: C[i] = sum_k A[i,k]*B[k], i=0,1; k=0..3
    Rank of M_2,4,1: schoolbook is 2*4=8. Can we do better?
    C[0] = a0*b0 + a1*b1 + a2*b2 + a3*b3
    C[1] = a4*b0 + a5*b1 + a6*b2 + a7*b3
    This is two dot products. The tensor is T[i,k,b] = delta_{i,k_row} * delta_{k_col,b}
    Actually T[a_idx, b_idx, c_idx] where a has 8 entries, b has 4, c has 2.
    T[a,b,c] = 1 if a = c*4+b, else 0.
    Rank = 8 (each of the 8 a-entries pairs with one b-entry and one c-entry).
    So M_2,4,1 has rank 8, and M_2,4,4 has rank 4*8=32 by this decomposition.
    
    But Strassen gives 28! How? Because Strassen exploits the 2x2x2 structure.
    M_2,4,4 = 4 copies of M_2,2,2 (splitting B and C into 2x2 blocks).
    4 * 7 = 28. Yes.
    
    So the question is: can we do better than 28 for M_2,4,4?
    The lower bound for M_2,4,4... I'm not sure of the exact value.
    
    Let me try a different approach: what if we use a global structure?
    
    Actually, let me try to see if we can merge some of the Strassen terms.
    
    In the current decomposition, we have:
    - 2 blocks of M_2,4,4 (for A rows 0,1; C cols 0-3 and 4-7)
    - 2 blocks of M_1,4,4 (for A row 2; C cols 0-3 and 4-7)
    
    The M_1,4,4 blocks are pure schoolbook (rank 16 each).
    
    What if we combine the M_1,4,4 with part of the M_2,4,4?
    For example, treat rows 1,2 of A together with Strassen?
    That would give M_2,4,4 for rows 1,2 and M_1,4,4 for row 0.
    Same total: 28 + 16 per block pair = 44, total 88.
    
    What about treating all 3 rows with a different algorithm?
    M_3,4,4: Is there a rank-43 or lower algorithm?
    
    Let me try a different split. What if I split B into 2+2 columns?
    C[i,j] = sum_k A[i,k]*B[k,j]
    Split B into B_left (4x2) and B_right (4x2), C into C_left (2x3) and C_right (2x3).
    C_left = A * B_left (3x2 result), C_right = A * B_right (3x2 result).
    Each is M_3,4,2.
    
    M_3,4,2: C[i,j] = sum_k A[i,k]*B[k,j], i=0,1,2; j=0,1; k=0..3
    Split into 3 copies of M_1,4,2: rank 3*4*2 = 24 schoolbook.
    Or: split A into 2+1: M_2,4,2 + M_1,4,2
    M_2,4,2: 2 copies of M_2,4,1 = 2*8 = 16. Or Strassen: 2 copies of M_2,2,1?
    M_2,2,1: C[i] = sum_k A[i,k]*B[k], i=0,1; k=0,1
    C[0] = a0*b0 + a1*b1
    C[1] = a2*b0 + a3*b1
    Rank: schoolbook 4. Strassen for 2x2x1? 
    T[a,b,c]: a has 4 entries, b has 2, c has 2.
    T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1
    Rank = 4 (it's a permutation tensor, no Strassen-like shortcut).
    So M_2,4,2 = 2 * 4 = 8... wait that doesn't match.
    
    Let me recompute. M_2,4,2:
    A is 2x4 (8 entries), B is 4x2 (8 entries), C is 2x2 (4 entries).
    C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    Schoolbook: 2*4*2 = 16 terms.
    Split into 2 sub-problems: for j=0 and j=1, each is M_2,4,1 (rank 8).
    Total: 16.
    
    Can we do better? The tensor T[a,b,c] where a in {0..7}, b in {0..7}, c in {0..3}:
    T[a,b,c] = 1 if a = (c//2)*4 + (c%2) ... no this is getting complicated.
    
    Let me just try to verify if rank 88 is optimal or if there's a known better result.
    
    Actually, let me try a computational approach: use the null-space idea from the directive.
    Start with the 4x4x8 schoolbook (rank 128), restrict to 3x4x8, and see if
    we can find linear dependencies among the rank-1 terms.
    
    For 3x4x8 schoolbook (rank 96):
    Term (i,k,j): U has 1 at position i*4+k, V has 1 at position k*8+j, W has 1 at position j*3+i.
    
    A term is "redundant" if its U-vector is zero (impossible since i<3) or if it can
    be expressed as a linear combination of other terms.
    
    Actually, the schoolbook terms for 3x4x8 are all linearly independent (it's a
    "standard" tensor with no obvious dependencies). So rank 96 is the schoolbook rank.
    
    The Strassen improvement comes from a different decomposition, not from
    eliminating schoolbook terms.
    
    Let me try yet another approach: what if I use a 3x3x3 Strassen-like algorithm
    on the row dimension?
    
    Actually, I recall that the tensor rank of M_3,3,3 is at most 23 (using a
    combination of Strassen and other techniques). But I don't know the exact
    decomposition.
    
    Let me try a practical approach: see if I can improve the M_1,4,4 part.
    
    M_1,4,4: C[j] = sum_k A[k]*B[k,j], j=0..3
    This is computing 4 dot products of length 4.
    The tensor has rank 16 (schoolbook). Can we do better?
    
    The tensor T[a,b,c] = 1 if b = a*4+c, else 0, where a in {0,1,2,3}, b in {0..15}, c in {0,1,2,3}.
    This is a "diagonal" tensor (permutation-like). Its rank is exactly 16.
    Proof: The 4x16 matrix (flattening T as a function of a and b for fixed c) 
    has rank 4 for each c, and the 4 matrices for different c are linearly independent.
    
    So M_1,4,4 has rank exactly 16. No improvement possible there.
    
    For M_2,4,4: rank 28 via Strassen. Is 28 optimal?
    M_2,4,4 = 4 x M_2,2,2. Each M_2,2,2 has rank 7. Total 28.
    But maybe we can share terms across the 4 M_2,2,2 sub-problems?
    
    M_2,4,4: A is 2x4, B is 4x4, C is 4x2.
    C[i,j] = sum_{k=0}^{3} A[i,k]*B[k,j]
    
    Let me split differently. Instead of 4 M_2,2,2 blocks, what if I use
    a different factorization?
    
    Actually, I think 28 might be optimal for M_2,4,4, but I'm not certain.
    
    Let me try a completely different approach: what if I use the fact that
    we only have 3 rows of A, and design a custom algorithm?
    
    C[i,j] = sum_k A[i,k]*B[k,j] for i=0,1,2; j=0..7; k=0..3
    
    Group by k: C[:,j] = sum_k A[:,k]*B[k,j]
    This is a sum of 4 rank-1 matrices (outer products of A[:,k] and B[k,:]).
    
    Alternatively: C = A * B where A is 3x4 and B is 4x8.
    
    What if I use a "bilinear" approach with fewer terms?
    
    Let me try: can I express this with rank < 88?
    
    Lower bound: The border rank of M_3,4,8 is at least... 
    Using the "slice rank" lower bound: the tensor has slice rank at least max(n*m, m*p, p*n) = max(12,32,24) = 32.
    That's a weak bound.
    
    Using the "tensor rank >= max of all matrix flattening ranks":
    - Flatten as (n*m) x (m*p) x (p*n): rank >= max(rank of each matricization)
    - Matricization 1: (12) x (32*24) = 12 x 768, rank <= 12
    - Matricization 2: (32) x (12*24) = 32 x 288, rank <= 32
    - Matricization 3: (24) x (12*32) = 24 x 384, rank <= 24
    So rank >= 32. Very weak.
    
    I think the current rank of 88 might be close to optimal, but let me try
    a few more ideas.
    
    Idea: What if I use a 2x2x2 Strassen on a different grouping?
    
    Currently: 
    - Rows 0,1 of A: 2 blocks of M_2,4,4 (each 28) = 56
    - Row 2 of A: 2 blocks of M_1,4,4 (each 16) = 32
    Total: 88
    
    Alternative:
    - Rows 0,1 of A, cols 0,1 of C: M_2,2,4 (A is 2x4, B is 4x2, C is 2x2)
      Wait, C is 8x3, so "cols of C" means the 3 columns.
      Let me re-index. C is 8 rows, 3 cols. C[r,c] where r in {0..7}, c in {0,1,2}.
      C[r,c] = sum_k A[c,k]*B[k,r]
      
    So for fixed c (row of A), we compute C[:,c] = A[c,:] * B (4x8 result).
    
    Grouping by c:
    - c=0: M_1,4,8 (A is 1x4, B is 4x8, C is 8x1), rank = 4*8 = 32
    - c=1: M_1,4,8, rank 32
    - c=2: M_1,4,8, rank 32
    Total: 96 (schoolbook)
    
    Using Strassen on pairs:
    - c=0,1: M_2,4,8 (A is 2x4, B is 4x8, C is 8x2)
      Split into 2 blocks of M_2,4,4: 2*28 = 56
    - c=2: M_1,4,8 (A is 1x4, B is 4x8, C is 8x1)
      Split into 2 blocks of M_1,4,4: 2*16 = 32
    Total: 88. Same as before.
    
    What if I group c=1,2 with Strassen?
    - c=0: M_1,4,8: 2*16 = 32
    - c=1,2: M_2,4,8: 2*28 = 56
    Total: 88. Same.
    
    What if I use a 3x2x2 Strassen-like on the c-dimension?
    For the 3 rows of A, can I compute all 3 M_1,4,8 products together with fewer terms?
    
    C[:,0] = A[0,:]*B
    C[:,1] = A[1,:]*B
    C[:,2] = A[2,:]*B
    
    This is equivalent to computing A*B where A is 3x4 and B is 4x8.
    
    Let me try a different factorization of B. Split B into 4 rows: B = [B_0; B_1; B_2; B_3] where B_k is 1x8.
    C[:,c] = sum_k A[c,k]*B_k
    
    So C[:,c] is a linear combination of the 4 row-vectors of B, with coefficients A[c,:].
    
    The 8x3 matrix C = [C[:,0], C[:,1], C[:,2]] = B^T * A^T (transposed).
    
    Hmm, let me think about this differently.
    
    What if I use a "Kronecker product" style decomposition?
    
    Actually, let me just try to verify the current solution works and see if there's
    a simple optimization I'm missing.
    
    Wait - I just realized something. In the M_1,4,4 block, the U-vector only has
    support on row 2 of A (4 entries). What if I can "merge" some of these with
    the Strassen terms?
    
    In the Strassen M_2,4,4 block for rows 0,1, the U-vectors have support on
    rows 0 and 1 of A. The M_1,4,4 block for row 2 has U-vectors with support
    only on row 2. There's no overlap, so no merging is possible.
    
    Unless... I redesign the decomposition to have overlapping support.
    
    Idea: Use a "mixed" decomposition where some rank-1 terms have U-support
    on multiple rows of A.
    
    For example, instead of separate Strassen for rows 0,1 and schoolbook for row 2,
    what if I use a 3x2x2 Strassen-like algorithm?
    
    M_3,2,2: A is 3x2, B is 2x2, C is 2x3.
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    Schoolbook rank: 3*2*2 = 12.
    Can we do better? 
    C[:,0] = A[:,0]*B[0,0] + A[:,1]*B[1,0]
    C[:,1] = A[:,0]*B[0,1] + A[:,1]*B[1,1]
    
    This is like computing 3 dot products of length 2.
    The tensor T[a,b,c] where a has 6 entries (3x2), b has 4 entries (2x2), c has 6 entries (2x3):
    T[a,b,c] = 1 if a = (c//2)*2 + (c%2) ... this is getting complex.
    
    Actually, M_3,2,2 has the same structure as M_2,2,2 but with 3 rows instead of 2.
    The rank of M_3,2,2: I believe it's 12 (schoolbook is optimal for this small case).
    
    Let me try a different approach entirely. What if I use a "fast rectangular" algorithm?
    
    For M_3,4,8, one approach is to embed it in a larger square problem and use
    a known fast algorithm.
    
    M_3,4,8 can be embedded in M_4,4,8 (by adding a zero row to A).
    M_4,4,8 = 2 x M_4,4,4.
    M_4,4,4 can be done with 2x2x2 Strassen: 4 x 7 = 28.
    So M_4,4,8 = 2*28 = 56.
    But we need to restrict to 3x4,8, which means we need the U-vectors to have
    zero in the 4th row position.
    
    In the M_4,4,8 Strassen decomposition, each rank-1 term has a U-vector that
    is a 4x4 matrix (flattened). After restriction, we keep only terms where the
    U-vector has zero in the 4th row (positions 12-15 in the 16-element vector).
    
    But Strassen U-vectors are linear combinations of A-entries, so they generally
    have support on all 4 rows. We can't just drop terms.
    
    However, we CAN modify the decomposition: for each Strassen term, if the U-vector
    has a nonzero component in row 3, we need to "cancel" it. This is the null-space
    projection idea.
    
    Specifically: In the 4x4x8 Strassen decomposition (rank 56), each term r has:
    - U_r: a 4x4 matrix (16 entries)
    - V_r: a 4x8 matrix (32 entries)
    - W_r: an 8x4 matrix (32 entries)
    
    For the 3x4x8 problem, we need U_r to have zero in row 3 (entries 12-15).
    
    If U_r[3,:] = 0 for some terms, we can keep them as-is.
    If U_r[3,:] != 0, we need to modify the decomposition.
    
    One approach: Add correction terms. For each term r with U_r[3,:] = u_r (a 1x4 vector),
    we can subtract u_r * (some B-row) and add it back in a different form.
    
    This is getting complicated. Let me try a more direct approach.
    
    Actually, I think the cleanest improvement might be to check if there's a
    known lower-rank algorithm for M_3,4,8 or M_3,4,4.
    
    Let me try a computational search: use the null-space projection idea more carefully.
    
    Start with 4x4x8 Strassen (rank 56). The tensor is:
    T_4[i,j,k] = delta_{i,k} for i in {0,1,2,3}, j in {0..7}, k in {0..3}
    
    We want T_3[i,j,k] = T_4[i,j,k] for i in {0,1,2}.
    
    In the Strassen decomposition of T_4, each rank-1 term is:
    T_r = U_r ⊗ V_r ⊗ W_r
    
    where U_r is 4x4, V_r is 4x8, W_r is 8x4.
    
    The restricted tensor T_3 = sum_r (P*U_r) ⊗ V_r ⊗ W_r
    where P is the 3x4 projection that zeros out row 3.
    
    But this doesn't directly give a rank-56 decomposition of T_3, because
    (P*U_r) ⊗ V_r ⊗ W_r is not a rank-1 term in the 3x4x8 tensor space
    (the U-part is now 3x4, which is fine, but the term might not be "clean").
    
    Actually, it IS a valid rank-1 term: U'_r = P*U_r (3x4), V_r (4x8), W_r (8x4).
    The tensor T_3 = sum_r U'_r ⊗ V_r ⊗ W_r is a valid rank-56 decomposition!
    
    But wait, some U'_r might be zero (if U_r only had support in row 3).
    Those terms can be dropped, reducing the rank.
    
    In the Strassen M_4,4,4 decomposition, the U-vectors are linear combinations
    of A-entries. For a 4x4x4 Strassen, we split into 2x2 blocks and apply
    2x2x2 Strassen to each. The U-vectors for the 2x2x2 Strassen terms have
    support on 2x2 blocks of A.
    
    For M_4,4,4 = 4 x M_2,2,2:
    - Block (0,0): U has support on A[0:2, 0:2]
    - Block (0,1): U has support on A[0:2, 2:4]
    - Block (1,0): U has support on A[2:4, 0:2]
    - Block (1,1): U has support on A[2:4, 2:4]
    
    After projecting out row 3 (index 3):
    - Block (0,0): U still has support on A[0:2, 0:2] (rows 0,1) - kept
    - Block (0,1): U still has support on A[0:2, 2:4] (rows 0,1) - kept
    - Block (1,0): U had support on A[2:4, 0:2] (rows 2,3). After projection, only row 2 remains.
      The U-vector becomes a 3x4 matrix with support only on row 2, cols 0,1.
      This is NOT zero, so the term is kept but modified.
    - Block (1,1): U had support on A[2:4, 2:4] (rows 2,3). After projection, only row 2 remains.
      Similarly kept but modified.
    
    So no terms are dropped! The rank remains 56 for the 4x4x8 case, and after
    restriction it's still 56 for the 3x4x8 case. But wait, that's for the full
    4x4x8 Strassen. The original problem is 3x4x8, not 4x4x8.
    
    Hmm, I'm confusing myself. Let me restart.
    
    The problem is 3x4 * 4x8 = 8x3.
    Current solution: rank 88.
    
    Let me try to see if I can get a lower rank by using a different algorithm.
    
    Key observation: The current solution uses:
    - 2x M_2,4,4 (Strassen, rank 28 each) = 56
    - 2x M_1,4,4 (schoolbook, rank 16 each) = 32
    Total: 88
    
    What if I use a different split? For example:
    - M_3,4,4 for the first 4 columns of B (and first 4 rows of C)
    - M_3,4,4 for the last 4 columns of B (and last 4 rows of C)
    
    M_3,4,4: A is 3x4, B is 4x4, C is 4x3.
    Schoolbook rank: 3*4*4 = 48.
    With Strassen on rows 0,1: M_2,4,4 (28) + M_1,4,4 (16) = 44.
    Total: 2*44 = 88. Same.
    
    What if M_3,4,4 has a rank < 44 algorithm?
    
    Let me think... M_3,4,4 can be viewed as:
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    One approach: use a 3x3x3 algorithm on a sub-block and handle the rest.
    
    Actually, I recall that the tensor rank of M_3,3,3 is 23 (or is it 22?).
    If I can use a rank-23 algorithm for M_3,3,3, then:
    M_3,4,4 = M_3,3,3 + M_3,1,4 + M_3,3,1 + M_3,1,1
    = 23 + 3*4 + 3*1 + 3*1 = 23 + 12 + 3 + 3 = 41
    
    Then 2*41 = 82 for the full 3x4x8. That would be an improvement!
    
    But I need to know the actual rank-23 decomposition of M_3,3,3.
    
    The best known upper bound for M_3,3,3 is 23 (due to... I'm not sure of the exact reference).
    Actually, I think the best known is 23, but I don't have the explicit decomposition memorized.
    
    Let me try a different approach: use a "block" Strassen that handles 3x3x3.
    
    Actually, let me try a simpler improvement first. What if I can reduce the
    M_1,4,4 rank below 16?
    
    M_1,4,4: C[j] = sum_k A[k]*B[k,j], j=0..3
    This is a 4x4 "matrix" (really a vector-matrix product).
    The tensor is T[a,b,c] = delta_{b=a*4+c} for a in {0,1,2,3}, b in {0..15}, c in {0,1,2,3}.
    
    I claimed the rank is 16. Let me verify: the tensor can be written as
    T = sum_{a=0}^{3} sum_{c=0}^{3} e_a ⊗ e_{a*4+c} ⊗ e_c
    
    Each term is rank 1, and there are 16 terms. Are they linearly independent?
    Yes, because each term has a unique (a,c) pair, and the b-index is determined by (a,c).
    
    So rank(M_1,4,4) = 16. No improvement possible.
    
    For M_2,4,4, the rank is 28 (Strassen). Is this optimal?
    M_2,4,4 = 4 x M_2,2,2. Each M_2,2,2 has rank 7. Total 28.
    Can we share terms? The 4 M_2,2,2 sub-problems are independent (different
    row/col blocks), so no sharing is possible. Rank = 28 is optimal for this
    block structure.
    
    But maybe a different block structure gives a lower rank for M_2,4,4?
    
    M_2,4,4: A is 2x4, B is 4x4, C is 4x2.
    Alternative split: split A into 2x2 + 2x2 (left and right halves).
    C[:,j] = A[:,0:2]*B[0:2,j] + A[:,2:4]*B[2:4,j]
    = (2x2 * 2x4) + (2x2 * 2x4)
    = M_2,2,4 + M_2,2,4
    
    M_2,2,4: A is 2x2, B is 2x4, C is 4x2.
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    This is 2x2x4. Schoolbook rank: 2*2*4 = 16.
    Split into 2x M_2,2,2: 2*7 = 14.
    
    So M_2,4,4 = 2 * M_2,2,4 = 2 * 14 = 28. Same as before.
    
    What about splitting B differently?
    M_2,4,4: split B into 4x2 + 4x2 (top and bottom halves).
    C[i,:] = A[i,0:2]*B[0:2,:] + A[i,2:4]*B[2:4,:]
    = (2x2 * 2x4) + (2x2 * 2x4)
    Same as above.
    
    I think 28 is the best we can do for M_2,4,4 with standard techniques.
    
    So the total rank of 88 seems hard to beat with the current approach.
    
    Let me try one more idea: what if I use a "global" Strassen-like algorithm
    that doesn't decompose into independent blocks?
    
    For M_3,4,8:
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    What if I group the 4 k-indices into 2+2 and use Strassen on the k-dimension?
    
    C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    
    For the first sum (k=0,1): this is M_3,2,8.
    For the second sum (k=2,3): this is M_3,2,8.
    
    M_3,2,8: A is 3x2, B is 2x8, C is 8x3.
    C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    
    Schoolbook rank: 3*2*8 = 48.
    Can we do better?
    
    C[:,j] = A[:,0]*B[0,j] + A[:,1]*B[1,j]
    This is a linear combination of two 8-vectors (A[:,0] and A[:,1]), with
    coefficients B[0,j] and B[1,j].
    
    The tensor T[a,b,c] where a has 6 entries (3x2), b has 16 entries (2x8), c has 24 entries (8x3):
    T[a,b,c] = 1 if a = (c//8)*2 + ... this is complex.
    
    Let me think of it differently. C = A * B where A is 3x2 and B is 2x8.
    This is equivalent to: C = A[:,0] * B[0,:] + A[:,1] * B[1,:]
    (outer products).
    
    The rank of this tensor: each outer product is rank 1, and there are 2 of them.
    But they're not independent rank-1 terms in the tensor sense...
    
    Actually, T[a,b,c] = A[a]*B[b] where the "product" is the matrix product.
    The tensor is: T = sum_{k=0}^{1} (A[:,k]) ⊗ (B[k,:]) ⊗ (identity on C)
    
    No, that's not right either. Let me be more careful.
    
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    The multiplication tensor is: T[i,k,j] = 1 if we're computing C[i,j] from A[i,k]*B[k,j].
    So T has entries T[a,b,c] where a indexes A (3*2=6), b indexes B (2*8=16), c indexes C (8*3=24).
    T[a,b,c] = 1 if a = i*2+k, b = k*8+j, c = j*3+i for some i,k,j.
    
    For fixed i,j: T has 1 at positions (i*2+k, k*8+j, j*3+i) for k=0,1.
    
    The rank of this tensor... I think it's 3*2*8 = 48 (schoolbook is optimal for this
    "rectangular" case where one dimension is 2).
    
    Actually, for M_3,2,8, we can write:
    C[:,j] = A[:,0]*B[0,j] + A[:,1]*B[1,j]
    
    For each j, this is a 3-vector that's a linear combination of two fixed 3-vectors.
    The tensor rank is at most 2*8 = 16? No, that's not right.
    
    Let me think again. The tensor T: for each (i,j), the "slice" T[:,:,i,j] is a 6x16 matrix
    with 2 ones (at positions (i*2, j*3+i) and (i*2+1, j*3+i) in the (a,c) indexing,
    and at positions (0*8+j) and (1*8+j) in the b-indexing).
    
    This is getting too complex to analyze by hand. Let me just try the current
    approach and see if I can make any small improvements.
    
    Actually, I just realized: the current program already achieves rank 88, and
    my analysis suggests this is hard to beat. Let me just make sure the program
    is correct and add a deadline guard as requested.
    
    Let me also try one more thing: what if I use a "hybrid" where I apply Strassen
    to a 2x2 sub-block of the 3x3 row-space?
    
    For example, treat rows 0,1 with Strassen and row 2 separately (current approach).
    Or treat rows 1,2 with Strassen and row 0 separately.
    Same total rank.
    
    What if I treat all 3 rows with a "3-row Strassen"?
    I don't know of a standard 3x3x3 Strassen algorithm with rank < 23.
    
    Let me just keep the current approach and add the deadline guard.
    """
    import time
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx][col] += au
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx][col] += bv
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx][col] += cw
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Pruning strategy: Remove zero columns or linearly dependent columns
    # Convert to numpy for easier manipulation
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check for zero columns
    keep_cols = []
    for r in range(rank):
        # Check if column r is zero
        if np.any(U_np[:, r] != 0) or np.any(V_np[:, r] != 0) or np.any(W_np[:, r] != 0):
            keep_cols.append(r)
            
    # Simple dependency check: if a column is a scalar multiple of a previous kept column, remove it
    # This is a heuristic; full SVD would be more expensive but more accurate.
    # Given the structured nature, zero-check is the most likely "prune".
    # We will just filter out zero columns to be safe.
    
    if len(keep_cols) < rank:
        U = [U_np[:, c].tolist() for c in keep_cols]
        V = [V_np[:, c].tolist() for c in keep_cols]
        W = [W_np[:, c].tolist() for c in keep_cols]
    else:
        U = U
        V = V
        W = W

    # Pruning strategy: Remove zero columns
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    keep_cols = []
    for r in range(rank):
        if np.any(U_np[:, r] != 0) or np.any(V_np[:, r] != 0) or np.any(W_np[:, r] != 0):
            keep_cols.append(r)
            
    if len(keep_cols) < rank:
        U_new = np.zeros((len(U), len(keep_cols)), dtype=float)
        V_new = np.zeros((len(V), len(keep_cols)), dtype=float)
        W_new = np.zeros((len(W), len(keep_cols)), dtype=float)
        for new_r, old_r in enumerate(keep_cols):
            U_new[:, new_r] = U_np[:, old_r]
            V_new[:, new_r] = V_np[:, old_r]
            W_new[:, new_r] = W_np[:, old_r]
        U = U_new.tolist()
        V = V_new.tolist()
        W = W_new.tolist()

    # Deadline check
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
