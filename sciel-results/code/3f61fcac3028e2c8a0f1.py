"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 5  # A is 3x5, B is 5x5, C is 3x5
    # Split the middle dimension (5) into 1 + 4
    # A: 3x5 -> [A1: 3x1, A4: 3x4]
    # B: 5x5 -> [[B11: 1x1, B14: 1x4], [B41: 4x1, B44: 4x4]]
    
    # We need to build U (n*m=15 rows, rank cols), V (m*p=25 rows, rank cols), W (p*n=15 rows, rank cols)
    # Indexing:
    #   U row: i*m + j where i in [0,n), j in [0,m)  -> A[i][j]
    #   V row: j*p + k where j in [0,m), k in [0,p)  -> B[j][k]
    #   W row: k*n + i where k in [0,p), i in [0,n)  -> C[k][i]  (trace convention: W[k*n+i, r])
    
    # Wait, the convention says W[k*n+i,r]. Let me re-read:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This means C[i][k] = sum_r U[i*m+j... no wait.
    # Standard: C[i][k] = sum_j A[i][j] B[j][k]
    # U has rows indexed by (i,j) for A[i][j]
    # V has rows indexed by (j,k) for B[j][k]  
    # W has rows indexed by (k,i) for C[k][i]... 
    # Actually the convention W[k*n+i] suggests the output C is indexed as C[k][i] or the row is k*n+i.
    # Let me just follow: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # And the product is: C[i,k] = sum_r U[i*m+j... no, that's not right either.
    # 
    # The standard tensor decomposition for matmul:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}  (with appropriate indexing)
    # Where T_{i,j,k} = 1 if the indices match for C[i][k] = sum_j A[i][j]B[j][k]
    # 
    # Actually: C[i][k] = sum_j A[i][j] B[j][k]
    # So the tensor T has T[i,j,k] = 1 (for all i,j,k it's part of the bilinear form)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Rank 43 decomposition:
    # Part 1: A1 * B11 (3*1*1, rank 3)
    # A1 is A[:, 0], B11 is B[0, 0]
    # C contribution: A[i][0] * B[0][k] for each i,k
    # This is 3*1*1... wait, A1 is 3x1, B11 is 1x1? No.
    # Let me reconsider the split.
    
    # A (3x5): columns 0..4. Split into A[:,0] (3x1) and A[:,1:5] (3x4)
    # B (5x5): rows 0..4, cols 0..4. 
    #   B[0,0] is 1x1
    #   B[0,1:5] is 1x4
    #   B[1:5,0] is 4x1
    #   B[1:5,1:5] is 4x4
    #
    # C = A @ B, so C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #   = A[i,0]*B[0,k] + sum_{j=1}^{4} A[i,j]*B[j,k]
    #   = A[i,0]*B[0,0] + A[i,0]*B[0,k] for k=0 plus... 
    # Let me be more careful:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k] + A[i,4]*B[4,k]
    #
    # Group:
    # Term1: A[i,0]*B[0,k] for all i,k -> this is 3x1x5, rank 15
    # Term2: (A[i,1..4]) * (B[1..4,k]) for all i,k -> this is 3x4x5, rank 60
    # Total schoolbook: 75
    #
    # Better split: 
    # C[i,k] = A[i,0]*B[0,0] + A[i,0]*sum_{k'=1}^{4} B[0,k'] * [k==k'] + sum_{j=1}^{4} A[i,j]*B[j,k]
    # 
    # Let me use: split m=5 into m1=1, m4=4; split p=5 into p1=1, p4=4
    # B is 5x5, split into 4 blocks:
    #   B[0,0] (1x1), B[0,1:5] (1x4)
    #   B[1:5,0] (4x1), B[1:5,1:5] (4x4)
    #
    # C[i,k] = A[i,0]*B[0,0]*[k==0] + A[i,0]*B[0,k] for k=1..4 + 
    #          sum_{j=1}^{4} A[i,j]*B[j,0]*[k==0] + sum_{j=1}^{4} A[i,j]*B[j,k] for k=1..4
    #
    # This is getting complicated. Let me just use the known rank-16 4x4x4 Strassen.
    # 
    # For 4x4x4, Strassen gives rank 7^2 = 49? No, Strassen on 2x2x2 gives 7.
    # 4x4x4 can be seen as 2x2 blocks of 2x2, so rank 7*7=49.
    # But there are algorithms for 4x4 with rank 16? Actually no, the best known for 4x4x4 is 49 via Strassen.
    # 
    # Hmm, let me reconsider. For 3x5x5:
    # Known bounds: rank(3x4x4) is at most 16 (using Strassen on 2x2 blocks: 4x4x4 rank 49, no)
    # 
    # Actually, let me just use a simple block decomposition that gets below 75.
    # Split m=5 into 2+3:
    # A (3x5) = [A_a (3x2), A_b (3x3)]
    # B (5x5) = [B_a (2x5); B_b (3x5)]
    # C = A_a @ B_a + A_b @ B_b
    # Rank(A_a @ B_a): 3*2*5 = 30
    # Rank(A_b @ B_b): 3*3*5 = 45
    # Total: 75. No improvement.
    #
    # The key is to use Strassen on square subproblems. 
    # For 3x5x5, I can't directly apply Strassen.
    # 
    # Let me try: pad to 4x5x5? No, that increases rank.
    #
    # Alternative: use the fact that rank(3x3x3) = 23 (Bini et al.)
    # But I don't have the coefficients memorized.
    #
    # Safest approach that's guaranteed to work: use Strassen for the 4x4x4 part.
    # Split 5 = 1 + 4:
    # C[i,k] = A[i,0]*B[0,k] + sum_{j=1}^4 A[i,j]*B[j,k]
    # First term: 3*1*5 = 15
    # Second term: 3*4*5. Now split p=5 into 1+4:
    #   sum_{j=1}^4 A[i,j]*B[j,k] where k=0..4
    #   = sum_{j=1}^4 A[i,j]*B[j,0]*[k==0] + sum_{j=1}^4 A[i,j]*B[j,k] for k=1..4
    #   First part: 3*4*1 = 12
    #   Second part: 3*4*4 = 48 (schoolbook) or 16 (Strassen for 4x4x4? No, Strassen 4x4x4 is 49)
    #
    # So: 15 + 12 + 48 = 75. Still schoolbook.
    #
    # I need to actually use Strassen. Let me implement Strassen for 4x4x4 (rank 49 via 2-level Strassen) 
    # and see if the total is less than 75.
    # 15 + 12 + 49 = 76. Worse!
    #
    # OK, different approach: use Strassen directly where possible.
    # For 3x5x5, the best I can do with simple blocks:
    # Split into 3x3x3 (rank 23) + 3x3x2 (rank 18) + 3x2x3 (rank 18) + 3x2x2 (rank 12) = 71
    # 
    # Let me do: split m=5 as 3+2, p=5 as 3+2
    # A (3x5) = [A11 (3x3), A12 (3x2)]
    # B (5x5) = [[B11 (3x3), B12 (3x2)], [B21 (2x3), B22 (2x2)]]
    # C = A11@B11 + A11@B12 + A12@B21 + A12@B22
    # Ranks: 27 + 18 + 18 + 12 = 75 (schoolbook for each)
    # Using Strassen for B11 (3x3): not directly applicable.
    #
    # I think the safest bet to get below 75 is to use a known rank-19 algorithm for 3x3x3 
    # (which exists: rank(3x3x3) <= 19? No, it's 23).
    #
    # Let me just use the schoolbook but verify it works, then try to find a real improvement.
    # Actually, for 3x5x5, I recall that rank <= 43 via specific algorithms.
    # But without the exact coefficients, I can't implement it.
    
    # Let me implement the schoolbook correctly and return that (score 1.0),
    # then the evaluator will test it.
    n, m, p = 3, 5, 5
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
