"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    # Rank-42 decomposition using Strassen-style block decomposition
    # A (2x4) = [A1 | A2] where A1, A2 are 2x2
    # B (4x6) = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    # Use Strassen on the 2x2 block structure, with each sub-multiplication being 2x2 * 2x3 (rank 6)
    # Total rank: 7 * 6 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2 block multiplication
    # P1 = (A1+A2)*(B1+B2)
    # P2 = (A1+A2)*B2
    # P3 = A1*(B1-B2)
    # P4 = A2*(B2-B1)
    # P5 = (A1-A2)*B1
    # P6 = (A1+A2)*B1
    # P7 = (A2-A1)*(B2-B1)
    # C11 = P1 + P4 - P6 + P7
    # C12 = P3 + P6
    # C21 = P2 + P5
    # C22 = P1 - P3 + P5 + P7
    
    # But we need to handle the 2x6 blocks. Let's split B1, B2 into 2x3 sub-blocks.
    # B1 = [B1a | B1b], B2 = [B2a | B2b] where each is 2x3
    # C = [C1 | C2] where C1 is 2x3, C2 is 2x3
    # C1 = A1*B1a + A2*B2a
    # C2 = A1*B1b + A2*B2b
    
    # For each 2x2 * 2x3 multiplication, use rank-6 decomposition
    # The rank-6 decomposition of 2x2*2x3:
    # For M (2x2) * N (2x3) = R (2x3):
    # R[i,k] = sum_j M[i,j]*N[j,k]
    # We can write this as 6 terms: for each (i,j,k) with i in {0,1}, j in {0,1}, k in {0,1,2}
    # That's 2*2*3 = 12 terms for schoolbook. But rank 6 means we can do better.
    
    # Actually, let me reconsider. The tensor for 2x2*2x3 has dimensions 2x2x3.
    # Its rank: I'll use the fact that it can be decomposed as follows.
    # Each output element R[i,k] = M[i,0]*N[0,k] + M[i,1]*N[1,k]
    # This is a sum of 2 products for each (i,k) pair, giving 2*2*2 = 8 terms minimum?
    # No, the rank of the 2x2x3 tensor is actually 6.
    
    # Let me use a known rank-6 decomposition for 2x2*2x3:
    # Terms:
    # t1: M[0,0]*N[0,0] -> R[0,0]
    # t2: M[0,1]*N[1,0] -> R[0,0]
    # t3: M[0,0]*N[0,1] -> R[0,1]
    # t4: M[0,1]*N[1,1] -> R[0,1]
    # t5: M[1,0]*N[0,2] -> R[1,2]
    # t6: M[1,1]*N[1,2] -> R[1,2]
    # That's only 6 terms but misses some outputs. This isn't right.
    
    # Let me just use the schoolbook for the inner 2x2*2x3 (rank 12) and see if the total is 7*12=84. That's worse.
    
    # Alternative approach: Just directly construct a low-rank decomposition.
    # Let me try rank 42 by using Strassen at the top level with appropriate block sizes.
    
    # Actually, the simplest correct approach: use the known Strassen decomposition for 2x4*4x6.
    # Split A into 2x2 blocks (2 blocks), B into 2x6 blocks (2 blocks).
    # Use 7 Strassen products, each involving 2x2 * 2x6.
    # For each 2x2 * 2x6, use the schoolbook rank 24? No, that gives 7*24=168.
    
    # I think the key is that 2x2 * 2x6 can be done with rank 6 if we use a clever decomposition.
    # 2x2 * 2x6: M is 2x2, N is 2x6, R is 2x6.
    # R[i,k] = M[i,0]*N[0,k] + M[i,1]*N[1,k]
    # This is a 2x2x6 tensor. Its rank is 12 (schoolbook) or can it be lower?
    # The rank of 2x2x6 is at least 6 (since it contains 2x2x1 as a sub-tensor, rank >= 4, but actually...)
    
    # I'm overcomplicating this. Let me just use the measured rank-42 approach.
    # The knowledge says: "treating the 2x4x6 multiplication as a 2x2 block matrix multiplication 
    # where each block multiplication (1x2 by 2x3) has rank 6, yielding 7*6=42 terms."
    
    # So the block structure is: 2x2 blocks where each block multiplication is 1x2 * 2x3.
    # This means: A is split into 2x2 blocks of size 1x2? That doesn't make sense for 2x4.
    
    # Wait: 2x4 matrix, split into 2x2 grid of blocks. Each block is 1x2.
    # A = [[A11, A12], [A21, A22]] where each Aij is 1x2
    # B is 4x6, split into 2x2 grid. Each block is 2x3.
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x3
    # C = A*B is 2x6, split into 2x2 grid. Each block is 1x3.
    # C = [[C11, C12], [C21, C22]] where each Cij is 1x3
    
    # C11 = A11*B11 + A12*B21  (1x2 * 2x3 + 1x2 * 2x3)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Strassen for 2x2 block multiplication:
    # P1 = (A11+A12)*(B11+B21)  -- but A11+A12 is 1x2, B11+B21 is 2x3, product is 1x3
    # P2 = (A11+A12)*B21
    # P3 = A11*(B11-B21)
    # P4 = A22*(B22-B12)
    # P5 = (A21-A11)*B11
    # P6 = (A11+A12)*B11
    # P7 = (A22-A12)*(B22-B12)
    
    # C11 = P1 + P4 - P6 + P7
    # C12 = P3 + P6
    # C21 = P2 + P5
    # C22 = P1 - P3 + P5 + P7
    
    # Each Pi is a 1x2 * 2x3 multiplication, which has rank 6 (schoolbook: 1*2*3=6)
    # Total rank: 7 * 6 = 42
    
    # Now construct the decomposition:
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)  
    # W has shape (p*n, rank) = (12, 42)
    
    # Indexing:
    # U row: i*m + j where i in {0,1}, j in {0,1,2,3} -> 8 rows
    # V row: j*p + k where j in {0,1,2,3}, k in {0,1,2,3,4,5} -> 24 rows
    # W row: k*n + i where k in {0,1,2,3,4,5}, i in {0,1} -> 12 rows
    
    # Block indices:
    # A block (bi, bj): rows i in {bi}, cols j in {2*bj, 2*bj+1}
    #   U index: i*m + j = bi*4 + 2*bj + dj where dj in {0,1}
    # B block (bi, bj): rows j in {2*bi, 2*bi+1}, cols k in {3*bj, 3*bj+1, 3*bj+2}
    #   V index: j*p + k = (2*bi + dj)*6 + 3*bj + dk where dj in {0,1}, dk in {0,1,2}
    # C block (bi, bj): rows k in {3*bj, 3*bj+1, 3*bj+2}, cols i in {bi}
    #   W index: k*n + i = (3*bj + dk)*2 + bi where dk in {0,1,2}
    
    # Strassen P terms: each is (1x2) * (2x3) -> 1x3
    # For a P term with A-block (ai, aj) and B-block (bi, bj):
    # The 6 schoolbook terms for 1x2 * 2x3:
    # For dj in {0,1} (A's column within block), dk in {0,1,2} (B's column within block):
    #   U[ai*4 + 2*aj + dj] += coeff_A
    #   V[(2*bi + dj)*6 + 3*bj + dk] += coeff_B
    #   W[(3*bj + dk)*2 + ai] += 1
    
    # Strassen definitions (with A blocks indexed (r,c) for row r, col c; B blocks (r,c)):
    # P1: A blocks (0,0)+(0,1) times B blocks (0,0)+(1,0)
    #   A part: (0,0) and (0,1) with +1 each
    #   B part: (0,0) and (1,0) with +1 each
    # P2: A blocks (0,0)+(0,1) times B block (1,0)
    #   A part: (0,0) and (0,1) with +1 each
    #   B part: (1,0) with +1
    # P3: A block (0,0) times B blocks (0,0)-(1,0)
    #   A part: (0,0) with +1
    #   B part: (0,0) with +1, (1,0) with -1
    # P4: A block (1,1) times B blocks (1,1)-(0,1)
    #   A part: (1,1) with +1
    #   B part: (1,1) with +1, (0,1) with -1
    # P5: A blocks (1,0)-(0,0) times B block (0,0)
    #   A part: (1,0) with +1, (0,0) with -1
    #   B part: (0,0) with +1
    # P6: A blocks (0,0)+(0,1) times B block (0,0)
    #   A part: (0,0) and (0,1) with +1 each
    #   B part: (0,0) with +1
    # P7: A blocks (1,1)-(0,1) times B blocks (1,1)-(0,1)
    #   A part: (1,1) with +1, (0,1) with -1
    #   B part: (1,1) with +1, (0,1) with -1
    
    # C reconstruction:
    # C(0,0) = P1 + P4 - P6 + P7
    # C(0,1) = P3 + P6
    # C(1,0) = P2 + P5
    # C(1,1) = P1 - P3 + P5 + P7
    
    # For each P term, we add 6 schoolbook sub-terms (rank 6 per P)
    # Total: 7 * 6 = 42
    
    # Let's define the P terms as lists of (A_block, coeff_A, B_block, coeff_B)
    # where A_block is (ar, ac) and B_block is (br, bc)
    
    # P1: A: [(0,0,+1), (0,1,+1)], B: [(0,0,+1), (1,0,+1)]
    # P2: A: [(0,0,+1), (0,1,+1)], B: [(1,0,+1)]
    # P3: A: [(0,0,+1)], B: [(0,0,+1), (1,0,-1)]
    # P4: A: [(1,1,+1)], B: [(1,1,+1), (0,1,-1)]
    # P5: A: [(1,0,+1), (0,0,-1)], B: [(0,0,+1)]
    # P6: A: [(0,0,+1), (0,1,+1)], B: [(0,0,+1)]
    # P7: A: [(1,1,+1), (0,1,-1)], B: [(1,1,+1), (0,1,-1)]
    
    # For each P term, the 6 schoolbook terms are:
    # For each combination of A-sub-block and B-sub-block:
    #   A-sub-block: (ar, ac, da) where da in {0,1}
    #   B-sub-block: (br, bc, db) where db in {0,1}
    #   The product contributes to C-sub-block (ar, bc, dc) where dc in {0,1,2}
    #   U index: ar*4 + 2*ac + da
    #   V index: (2*br + db)*6 + 3*bc + dc  -- wait, this isn't right for the schoolbook
    #   
    # Actually for 1x2 * 2x3 schoolbook:
    # R[dc] = sum_{dj=0}^{1} A[aj, dj] * B[br, dj, bc, dc]
    # Each term: A[aj, dj] * B[br, dj, bc, dc]
    # U: A[aj, dj] -> index aj*4 + 2*ac + dj
    # V: B[br, dj, bc, dc] -> index (2*br + dj)*6 + 3*bc + dc
    # W: C[ar, bc, dc] -> index (3*bc + dc)*2 + ar
    
    # So for a P term with A contributions [(ar1, ac1, ca1), ...] and B contributions [(br1, bc1, cb1), ...]:
    # For each A contribution (ar, ac, ca) and each B contribution (br, bc, cb):
    #   For dj in {0,1} and dc in {0,1,2}:
    #     U[ar*4 + 2*ac + dj] += ca
    #     V[(2*br + dj)*6 + 3*bc + dc] += cb
    #     W[(3*bc + dc)*2 + ar] += 1
    
    # Wait, but this would give 6 terms per (A_contrib, B_contrib) pair, and multiple pairs per P term.
    # P1 has 2 A contribs and 2 B contribs, so 4 pairs, each with 6 terms = 24 terms. That's too many.
    
    # I think the issue is that when we have multiple A and B contributions in one P term, 
    # they're summed together before multiplication. So:
    # P = (sum of A parts) * (sum of B parts)
    # = sum_{a in A_parts, b in B_parts} a * b
    # Each a*b is a (1x2)*(2x3) product with rank 6.
    # So P1 has 2*2 = 4 such products, each of rank 6, total 24.
    # Total rank would be: P1:24, P2:12, P3:12, P4:12, P5:12, P6:12, P7:24 = 108. That's way more than 42.
    
    # I think the "rank 6 per block" means that the entire P term (which is a sum of products) 
    # can be computed with rank 6, not 6 per sub-product. That would require a more clever decomposition.
    
    # Alternatively, maybe the decomposition is different. Let me just use rank 48 (schoolbook) 
    # and see if I can find any improvement, or accept rank 42 if I can construct it correctly.
    
    # For now, let me just return the schoolbook rank-48 decomposition to ensure correctness,
    # and note that a rank-42 decomposition exists but requires more careful construction.
    
    rank = n * m * p  # 48
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
