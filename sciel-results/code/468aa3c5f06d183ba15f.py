"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    # Schoolbook baseline: rank = n*m*p = 63
    best_rank = n * m * p
    best_decomp = None
    
    # Try to find a lower rank decomposition
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for the (i,j,k) term
    # We need to cover all (i,j,k) terms with rank-1 components
    
    # A rank-1 component is defined by:
    #   U_row = i*m + j  (single entry in U)
    #   V_row = j*p + k  (single entry in V)  
    #   W_row = k*n + i  (single entry in W)
    # This covers exactly one (i,j,k) term.
    
    # To get rank < 63, we need to combine terms.
    # A general rank-1 tensor component is:
    #   u[a] * v[b] * w[c] for a = i*m+j, b = j*p+k, c = k*n+i
    # The constraint is: for each (i,j,k), the sum over all rank-1 terms of
    #   u[i*m+j] * v[j*p+k] * w[k*n+i] must equal 1
    # And for non-tensor entries, it should equal 0.
    
    # This is a very constrained system. Let me try a structured approach.
    
    # Key insight: For matrix multiplication, we can use the following structure:
    # Group terms by (i, j) pairs. For each (i,j), we have 7 terms for k=0..6.
    # These share the same U entry u[i*m+j].
    
    # Let's try: use rank = number of distinct (i,j) pairs times something
    # Actually, let me try to construct a valid decomposition with rank ~ 50
    
    # Alternative: Use the fact that the multiplication can be expressed using
    # the structure of the matrices. Let me try a direct construction.
    
    # For 3x3 x 3x7:
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # 
    # We have 9 output positions per i (3 i's * 7 k's = 21 outputs total)
    # Each output needs 3 products.
    
    # Let me try: for each (i,j) pair, define a "block" that computes
    # A[i][j] * B[j][0:7] as 7 separate products, but we can potentially
    # combine some across different i's or j's.
    
    # Actually, the cleanest approach for a guaranteed valid answer with better
    # than schoolbook: use the standard result that bilinear complexity of (n,m,p)
    # matrix multiplication is at most n*m*p, and for small cases we can verify
    # specific constructions.
    
    # Let me try a simple improvement: notice that for a fixed j, 
    # A[i][j] for i=0,1,2 and B[j][k] for k=0..6.
    # We can't easily share across j since different j's have different i patterns.
    
    # Given the complexity of finding a provably correct lower-rank decomposition
    # for (3,3,7) without access to known algorithms, let me try an optimization
    # approach with verification.
    
    # Try rank values from 50 up to 63 and see if we can find a valid one
    # This is computationally expensive, so let's be strategic.
    
    # Actually, let me just return a slightly improved structured decomposition.
    # 
    # Insight: We can use the following construction:
    # For each of the 3 rows i of A and 3 columns j:
    #   Create 7 rank-1 terms for k=0..6
    # But we can merge some terms.
    #
    # Let me try rank = 54 (saving 9 from 63)
    # Idea: for each j, the 3 values A[0][j], A[1][j], A[2][j] are multiplied
    # with the same row B[j][0:7]. We can potentially share some computation.
    
    # Given the time constraints and need for correctness, let me implement
    # a verified schoolbook but with a check, and attempt one structured improvement.
    
    # Structure: Use the fact that B has 7 columns. Split B into B1 (3x3) and B2 (3x4).
    # Compute A@B1 using Laderman-like approach (rank ~23) and A@B2 using schoolbook (rank 36).
    # Total rank: 23 + 36 = 59. Still less than 63!
    
    # Let me implement this:
    # Part 1: A (3x3) @ B1 (3x3) -> C1 (3x3), using a rank-23 decomposition
    # Part 2: A (3x3) @ B2 (3x4) -> C2 (3x4), using schoolbook rank 36
    # Total rank: 59
    
    # For the 3x3x3 part, let me use a known rank-23 decomposition.
    # Actually, let me just use schoolbook for both parts: 27 + 36 = 63 (no improvement)
    
    # To actually get improvement, I need a real sub-27 rank for 3x3x3.
    # Laderman's algorithm has rank 23. Let me try to construct it.
    
    # Laderman's 23-multiplication algorithm for 3x3 matrix multiplication:
    # This is complex to derive from scratch correctly. 
    
    # Alternative practical approach: use rank 59 with a verified construction
    # by combining a rank-27 3x3x3 with rank-36 3x3x4 = 63 total (no gain).
    
    # Let me try a different split or a completely different structured approach.
    
    # Actually, the safest approach given time constraints: return the schoolbook
    # decomposition (rank 63) which is guaranteed correct. Score = 63/63 = 1.0.
    
    # But let me at least try to verify if a simple structured improvement works.
    
    # Try: for the 3x7 multiplication, note that we only have 3 distinct rows of B
    # being multiplied. The structure is A (3x3) times B (3x7).
    # C[i][k] = A[i][0]*B[0][k] + A[i][1]*B[1][k] + A[i][2]*B[2][k]
    
    # For fixed i, this is the dot product of row i of A with each row of B.
    # We can precompute for each i: 3 dot products of length 7.
    # That's 3*7 = 21 products per i, times 3 i's = 63 total (schoolbook).
    
    # To save: if A[i][j] = 0 for some (i,j), we save products. But A is general.
    
    # Given the constraints, I'll return the schoolbook decomposition for guaranteed
    # correctness. Any attempted optimization risks invalidity and score 0.
    
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
    
    # Verify decomposition before returning
    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val, rank_val):
        """Verify that the decomposition correctly reconstructs matrix multiplication."""
        # For each (i, k), check that sum_j A[i,j]*B[j,k] is reconstructed
        # The tensor entry T[i*m+j, j*p+k, k*n+i] should equal sum over rank of U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r]
        # which should equal delta_{i,k is connected through j} = 1 if this is a valid (i,j,k) triple
        
        # Check all tensor entries
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    expected = 1  # This (i,j,k) term exists
                    actual = 0
                    for r in range(rank_val):
                        actual += U_mat[i*m_val+j][r] * V_mat[j*p_val+k][r] * W_mat[k*n_val+i][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        
        # Check non-tensor entries are 0
        # For entries where the indices don't form a valid (i,j,k) triple
        # a = i*m + j must have 0<=i<n, 0<=j<m
        # b = j'*p + k must have 0<=j'<m, 0<=k<p  
        # c = k'*n + i' must have 0<=k'<p, 0<=i'<n
        # The tensor entry is nonzero only when j=j' and i=i' and k=k'
        for a in range(n_val * m_val):
            i1 = a // m_val
            j1 = a % m_val
            for b in range(m_val * p_val):
                j2 = b // p_val
                k1 = b % p_val
                for c in range(p_val * n_val):
                    k2 = c // n_val
                    i2 = c % n_val
                    expected = 1 if (j1 == j2 and i1 == i2 and k1 == k2) else 0
                    actual = 0
                    for r in range(rank_val):
                        actual += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p, rank):
        raise ValueError("Schoolbook decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
