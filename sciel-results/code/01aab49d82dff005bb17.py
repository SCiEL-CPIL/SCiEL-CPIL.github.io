"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7) for A (2x2)
    V shape: (4, 7) for B (2x2)
    W shape: (4, 7) for C (2x2), with W[k*2+i, r] convention
    """
    # Strassen 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: rows are A elements (00,01,10,11), cols are the 7 multiplications
    U = np.zeros((4, 7), dtype=int)
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    U[0, 0] = 1  # A00 in M1
    U[3, 0] = 1  # A11 in M1
    U[2, 1] = 1  # A10 in M2
    U[3, 1] = 1  # A11 in M2
    U[0, 2] = 1  # A00 in M3
    U[3, 3] = 1  # A11 in M4
    U[0, 4] = 1  # A00 in M5
    U[1, 4] = 1  # A01 in M5
    U[2, 5] = 1  # A10 in M6
    U[0, 5] = -1 # A00 in M6
    U[1, 6] = 1  # A01 in M7
    U[3, 6] = -1 # A11 in M7
    
    # V: rows are B elements (00,01,10,11), cols are the 7 multiplications
    V = np.zeros((4, 7), dtype=int)
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    V[0, 0] = 1  # B00 in M1
    V[3, 0] = 1  # B11 in M1
    V[0, 1] = 1  # B00 in M2
    V[1, 2] = 1  # B01 in M3
    V[3, 2] = -1 # B11 in M3
    V[2, 3] = 1  # B10 in M4
    V[0, 3] = -1 # B00 in M4
    V[3, 4] = 1  # B11 in M5
    V[0, 5] = 1  # B00 in M6
    V[1, 5] = 1  # B01 in M6
    V[2, 6] = 1  # B10 in M7
    V[3, 6] = 1  # B11 in M7
    
    # W: rows are C elements with W[k*n+i] where n=2, so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # Wait, the convention is W[k*n+i, r]. For 2x2x2, n=2, p=2.
    # W[k*n+i] where k is column index of C, i is row index of C? 
    # Actually re-reading: "W[k*n+i,r]" - this is for the output factor.
    # For matrix mult C = A*B where A is n x m, B is m x p, C is n x p.
    # The tensor is T[i,j,k] = delta, and W has shape (p*n, rank).
    # W[k*n+i] means: for output C[i,j] (i=row, j=col), the index in W is j*n + i.
    # So W[j*n + i, r] contributes to C[i,j].
    
    # For 2x2x2: C[i,j] for i,j in {0,1}
    # W[j*2 + i, r]
    # C[0,0]: W[0, r]
    # C[0,1]: W[2, r]  (j=1, i=0 -> 1*2+0=2)
    # C[1,0]: W[1, r]  (j=0, i=1 -> 0*2+1=1)
    # C[1,1]: W[3, r]  (j=1, i=1 -> 1*2+1=3)
    
    W = np.zeros((4, 7), dtype=int)
    # C00 = M1 + M4 - M5 + M7 -> W[0]
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # -M5
    W[0, 6] = 1   # M7
    # C01 = M3 + M5 -> W[2]
    W[2, 2] = 1   # M3
    W[2, 4] = 1   # M5
    # C10 = M2 + M4 -> W[1]
    W[1, 1] = 1   # M2
    W[1, 3] = 1   # M4
    # C11 = M1 - M2 + M3 + M6 -> W[3]
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # -M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return U, V, W

def construct():
    """Construct rank-56 decomposition for 4x4 x 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # We decompose this as two independent 4x4x4 multiplications
    # (left 4 cols of B and right 4 cols of B)
    # Each 4x4x4 is decomposed into 4 Strassen 2x2x2 sub-problems
    # Total rank = 2 * 4 * 7 = 56
    
    # For a 4x4x4 multiplication:
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    #
    # C00 (top-left 2x2) = A00*B00 + A01*B10
    # C01 (top-right 2x2) = A00*B01 + A01*B11
    # C10 (bottom-left 2x2) = A10*B00 + A11*B10
    # C11 (bottom-right 2x2) = A10*B01 + A11*B11
    #
    # Each of these is a SUM of two 2x2x2 products, not a single Strassen.
    # So this approach doesn't directly give rank 28.
    
    # Alternative: use 8 Strassen sub-problems by splitting all 3 dimensions.
    # 4x4x4 tensor -> split each dim in half -> 8 sub-tensors of 2x2x2.
    # But the 4x4x4 tensor T[i,j,k] = delta_{ik} (for the multiplication tensor)
    # doesn't split into independent 2x2x2 sub-tensors.
    
    # Let me just use the direct approach: for 4x4x8, use rank 56 by
    # treating it as 8 independent 2x2x2 Strassen problems.
    # 
    # Actually, the cleanest correct approach: 
    # 4x4 x 4x8 = sum over k of A[:,k] * B[k,:]
    # This is a sum of 4 rank-32 terms (each k gives outer product of 4x1 and 1x8 = rank 1 per element, but as a tensor it's different)
    
    # Let me just build it explicitly with the schoolbook approach but optimized.
    # Schoolbook rank for 4x4x8 is 4*4*8 = 128.
    # 
    # For a better decomposition, let me use the fact that we can group operations.
    # 
    # Actually, let me just use rank 56 via the following construction:
    # View 4x4x8 as having A (4x4), B (4x8), C (4x8).
    # Split B into B1 (4x4) and B2 (4x4).
    # C = A*B1 concatenated with A*B2.
    # Each A*Bj is a 4x4x4 multiplication.
    # 
    # For 4x4x4, I'll use a rank-28 decomposition based on 4 Strassen blocks.
    # The key insight: we can write the 4x4x4 tensor as a sum of 4 terms,
    # each involving 2x2x2 Strassen-like structure.
    
    # Let me just implement the straightforward rank-56 by using 8 Strassen 2x2x2
    # sub-problems where we partition the index sets appropriately.
    
    # For 4x4x8:
    # A[i,k] for i in 0..3, k in 0..3
    # B[k,j] for k in 0..3, j in 0..7
    # C[i,j] for i in 0..3, j in 0..7
    #
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    #
    # Partition: let i = 2*i' + i'', k = 2*k' + k'', j = 4*j' + j'' where
    # i',k' in {0,1}, i'',k'' in {0,1}, j' in {0,1}, j'' in {0,1,2,3}
    #
    # This is getting too complex. Let me just use a simple correct construction.
    
    # Simple approach: rank 56 using the following block structure.
    # For each of the 4 rows of A and 4 cols of B, we have 8 output entries.
    # 
    # I'll use a hybrid: Strassen on the 4x4x4 blocks.
    # 
    # Final approach: Just construct rank-56 directly.
    # Use 7 Strassen multiplications for each of 8 "sub-problems".
    
    # Actually, the simplest correct rank-56: 
    # 4x4x8 can be seen as 2 copies of 4x4x4.
    # For each 4x4x4, use rank 28.
    # 
    # For 4x4x4 with rank 28: I'll use the following construction.
    # Write the 4x4x4 multiplication tensor as a sum of 4 terms, each of rank 7.
    # Term t (for t in 0..3) handles a specific "block pattern".
    
    # Let me just use numpy to construct and verify.
    
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with margin
    
    # Build rank-56 decomposition
    # Strategy: 8 Strassen 2x2x2 sub-decompositions
    # Partition indices:
    # A: (4,4) -> split into 4 blocks of (2,2): A[a',a''] where a'=block_row, a''=local
    # B: (4,8) -> split into 8 blocks of (2,2): B[b',b''] where b'=block_col, b''=local  
    # C: (4,8) -> split into 8 blocks of (2,2): C[c',c'']
    
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Let i = 2*i_b + i_l, j = 2*j_b + j_l, k = 2*k_b + k_l
    # where i_b, j_b, k_b in {0,1} and i_l, j_l, k_l in {0,1}
    # j_b ranges 0..3 since j goes 0..7, so j_b in {0,1,2,3}
    
    # C[2*i_b+i_l, 2*j_b+j_l] = sum_{k_b,k_l} A[2*i_b+i_l, 2*k_b+k_l] * B[2*k_b+k_l, 2*j_b+j_l]
    
    # This groups into terms indexed by (i_b, j_b, k_b) with inner 2x2x2 structure.
    # For fixed (i_b, j_b, k_b), we have a 2x2x2 sub-tensor.
    # There are 2*4*2 = 16 such sub-tensors, but they're not all independent.
    
    # This approach gives rank 16*7 = 112, which is worse than schoolbook.
    
    # Better: just use the direct Strassen-on-blocks approach.
    # 
    # I'll construct the decomposition by explicitly building U, V, W for rank 56.
    
    rank = 56
    U = np.zeros((16, rank), dtype=int)  # n*m = 4*4 = 16
    V = np.zeros((32, rank), dtype=int)  # m*p = 4*8 = 32
    W = np.zeros((32, rank), dtype=int)  # p*n = 8*4 = 32
    
    # For each of the two halves of B (cols 0-3 and cols 4-7),
    # use a rank-28 decomposition for 4x4x4.
    
    # Rank-28 for 4x4x4: use 4 Strassen 2x2x2 decompositions.
    # The 4x4x4 tensor T[i,j,k] = 1 if i==k else 0 (for the multiplication tensor).
    # 
    # Decompose into 4 parts based on the 2x2 block structure:
    # Part 0: involves A blocks row 0, B blocks col 0
    # Part 1: involves A blocks row 0, B blocks col 1  
    # Part 2: involves A blocks row 1, B blocks col 0
    # Part 3: involves A blocks row 1, B blocks col 1
    
    # For part (a, b) where a is A row-block, b is B col-block:
    # C[2*a+a_l, 2*b+b_l] += sum_{k_b} A[2*a+a_l, 2*k_b+k_l] * B[2*k_b+k_l, 2*b+b_l]
    
    # This still couples k_b. 
    
    # I'll just use the simplest correct approach: rank 56 via explicit construction.
    # For each output element C[i,j], it's a dot product of row i of A and col j of B.
    # I'll use Strassen-style grouping.
    
    # Let me just use the known result: 4x4x4 has rank at most 28 (via Strassen recursion:
    # 4x4x4 = 8 sub-problems of 2x2x2, each rank 7, total rank 56... wait that's for the full tensor).
    
    # Actually, 4x4x4 matrix multiplication tensor: 
    # If we use Strassen's algorithm on 4x4 matrices by splitting into 2x2 blocks,
    # we need 7 "multiplications" of 2x2 matrices, but each 2x2 matrix multiplication
    # itself can be done with Strassen (rank 7). So total rank = 7 * 7 = 49? No...
    
    # Strassen on 4x4: split into 2x2 blocks. The 7 Strassen combinations are 2x2 matrices.
    # To multiply each pair of 2x2 matrices, we need rank 7 each.
    # So total rank = 7 * 7 = 49 for 4x4x4.
    
    # Then for 4x4x8: two 4x4x4 multiplications, rank 49*2 = 98.
    
    # Hmm, but the observed improvement was to rank 98 (score 1.306 = 128/98).
    # So rank 98 is achievable. Let me aim for that.
    
    # Rank 98 construction:
    # 4x4x8 = two 4x4x4 multiplications.
    # Each 4x4x4 uses Strassen recursion: 7 sub-multiplications of 2x2x2, each rank 7.
    # Total per 4x4x4: 7*7 = 49.
    # Total: 98.
    
    # Let me construct this.
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen, shapes (4,7), (4,7), (4,7)."""
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices (W[k*n+i]): C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1
        
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[2,2]=1; W[3,2]=1
        
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[1,3]=1
        
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1; W[2,4]=1
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1
        
        return U, V, W
    
    def strassen_4x4x4():
        """Return (U, V, W) for 4x4x4 using Strassen recursion, rank 49.
        U shape: (16, 49), V shape: (16, 49), W shape: (16, 49).
        W index: W[k*4+i, r] for output C[i,k]... wait.
        
        For 4x4x4: A is 4x4, B is 4x4, C is 4x4.
        U has shape (n*m, rank) = (16, rank)
        V has shape (m*p, rank) = (16, rank)  
        W has shape (p*n, rank) = (16, rank)
        W[k*n+i, r] where n=4, so W[j*4+i, r] for C[i,j].
        """
        sU, sV, sW = strassen_2x2x2()
        rank = 49
        U4 = np.zeros((16, rank), dtype=int)
        V4 = np.zeros((16, rank), dtype=int)
        W4 = np.zeros((16, rank), dtype=int)
        
        # Strassen on 4x4:
        # A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
        # B = [[B00, B01], [B10, B11]]
        # 
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # Each Mk is a 2x2 matrix = sum of products of 2x2 blocks.
        # To compute each Mk, we use Strassen 2x2x2 (rank 7).
        
        # A block indices in U4 (16 elements):
        # A00: rows 0-3 (i=0,1; k=0,1) -> U4[0:4]
        # A01: rows 4-7 (i=0,1; k=2,3) -> U4[4:8]
        # A10: rows 8-11 (i=2,3; k=0,1) -> U4[8:12]
        # A11: rows 12-15 (i=2,3; k=2,3) -> U4[12:16]
        
        # B block indices in V4 (16 elements):
        # B00: cols 0-3 (k=0,1; j=0,1) -> V4[0:4]
        # B01: cols 4-7 (k=0,1; j=2,3) -> V4[4:8]
        # B10: cols 8-11 (k=2,3; j=0,1) -> V4[8:12]
        # B11: cols 12-15 (k=2,3; j=2,3) -> V4[12:16]
        
        # For each Strassen combination m (0..6), we compute a 2x2 result matrix.
        # The result matrix R_m = X_m * Y_m where X_m and Y_m are 2x2 matrices.
        # To compute X_m * Y_m (2x2x2), we use Strassen with rank 7.
        # Each sub-strassen uses 7 scalar multiplications.
        # Total: 7 * 7 = 49.
        
        # For Strassen combination m:
        # X_m is a linear combination of A blocks, Y_m is a linear combination of B blocks.
        # The specific coefficients depend on m.
        
        # Strassen combinations:
        # m=0: X=A00+A11, Y=B00+B11
        # m=1: X=A10+A11, Y=B00
        # m=2: X=A00, Y=B01-B11
        # m=3: X=A11, Y=B10-B00
        # m=4: X=A00+A01, Y=B11
        # m=5: X=A10-A00, Y=B00+B01
        # m=6: X=A01-A11, Y=B10+B11
        
        # For each m, X_m is a 2x2 matrix formed from A blocks.
        # Y_m is a 2x2 matrix formed from B blocks.
        # We need to compute X_m * Y_m and add to appropriate C blocks.
        
        # The 2x2 matrix X_m has entries that are linear combinations of A block entries.
        # Similarly for Y_m.
        
        # Let me define the block structure more carefully.
        # A is 4x4, split into 2x2 blocks of size 2x2.
        # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01
        # A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
        
        # For U4, row index = i*4 + k where i is row of A, k is col of A.
        # A00: i in {0,1}, k in {0,1} -> rows 0,1,4,5
        # A01: i in {0,1}, k in {2,3} -> rows 2,3,6,7
        # A10: i in {2,3}, k in {0,1} -> rows 8,9,12,13
        # A11: i in {2,3}, k in {2,3} -> rows 10,11,14,15
        
        # Wait, I need to be consistent. U has shape (n*m, rank) = (16, rank).
        # The index is i*m + k = i*4 + k for A[i,k].
        # So:
        # A[0,0]=U4[0], A[0,1]=U4[1], A[0,2]=U4[2], A[0,3]=U4[3]
        # A[1,0]=U4[4], A[1,1]=U4[5], A[1,2]=U4[6], A[1,3]=U4[7]
        # A[2,0]=U4[8], A[2,1]=U4[9], A[2,2]=U4[10], A[2,3]=U4[11]
        # A[3,0]=U4[12], A[3,1]=U4[13], A[3,2]=U4[14], A[3,3]=U4[15]
        
        # B is 4x4, V has shape (m*p, rank) = (16, rank).
        # Index is k*p + j = k*4 + j for B[k,j].
        # B[0,0]=V4[0], B[0,1]=V4[1], B[0,2]=V4[2], B[0,3]=V4[3]
        # B[1,0]=V4[4], B[1,1]=V4[5], B[1,2]=V4[6], B[1,3]=V4[7]
        # B[2,0]=V4[8], B[2,1]=V4[9], B[2,2]=V4[10], B[2,3]=V4[11]
        # B[3,0]=V4[12], B[3,1]=V4[13], B[3,2]=V4[14], B[3,3]=V4[15]
        
        # C is 4x4, W has shape (p*n, rank) = (16, rank).
        # W[j*4 + i, r] for C[i,j].
        # C[0,0]=W4[0], C[1,0]=W4[1], C[2,0]=W4[2], C[3,0]=W4[3]
        # C[0,1]=W4[4], C[1,1]=W4[5], C[2,1]=W4[6], C[3,1]=W4[7]
        # C[0,2]=W4[8], C[1,2]=W4[9], C[2,2]=W4[10], C[3,2]=W4[11]
        # C[0,3]=W4[12], C[1,3]=W4[13], C[2,3]=W4[14], C[3,3]=W4[15]
        
        # Strassen 4x4:
        # M0 = (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
        # M1 = (A10+A11)B00 -> contributes to C10, C11
        # M2 = A00(B01-B11) -> contributes to C00, C01
        # M3 = A11(B10-B00) -> contributes to C00, C10
        # M4 = (A00+A01)B11 -> contributes to C00, C01
        # M5 = (A10-A00)(B00+B01) -> contributes to C10, C11
        # M6 = (A01-A11)(B10+B11) -> contributes to C00, C01, C10, C11
        
        # C00 = M0 + M3 - M4 + M6 (all 2x2 matrices)
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # Each Mx is a 2x2 matrix product. To compute Mx = Xx * Yx where Xx, Yx are 2x2:
        # Use Strassen 2x2x2 (rank 7).
        
        # For each m in 0..6, define X_m and Y_m as 2x2 matrices.
        # X_m entries are linear combos of A block entries.
        # Y_m entries are linear combos of B block entries.
        
        # Let me define the 2x2 matrices explicitly.
        # A00 = [[A00_00, A00_01],[A00_10, A00_11]] = [[A[0,0],A[0,1]],[A[1,0],A[1,1]]]
        # etc.
        
        # For Strassen 2x2x2 to compute X*Y where X,Y are 2x2:
        # We need to express X and Y in terms of their 4 entries,
        # apply the 7 Strassen multiplications, and get the 4 output entries.
        
        # The key: for each Strassen combination m (0..6), and each sub-Strassen s (0..6),
        # we have a specific set of coefficients.
        
        # This is getting very complex. Let me just build it programmatically.
        
        # For each top-level Strassen m (0..6):
        #   X_m is a 2x2 matrix with entries that are linear combos of A entries.
        #   Y_m is a 2x2 matrix with entries that are linear combos of B entries.
        #   To compute X_m * Y_m, use Strassen 2x2x2.
        #   The Strassen 2x2x2 gives 7 "scalar" multiplications.
        #   Each scalar multiplication corresponds to one column in U4, V4, W4.
        
        # Total: 7 * 7 = 49 columns.
        
        # Let me define the A and B block matrices for each m.
        
        # A block matrices (2x2 each):
        # A00: A[0:2, 0:2], A01: A[0:2, 2:4], A10: A[2:4, 0:2], A11: A[2:4, 2:4]
        
        # For m=0: X = A00+A11, Y = B00+B11
        # For m=1: X = A10+A11, Y = B00
        # For m=2: X = A00, Y = B01-B11
        # For m=3: X = A11, Y = B10-B00
        # For m=4: X = A00+A01, Y = B11
        # For m=5: X = A10-A00, Y = B00+B01
        # For m=6: X = A01-A11, Y = B10+B11
        
        # For each m, X and Y are 2x2 matrices.
        # X[i,j] is a linear combination of A entries.
        # Y[i,j] is a linear combination of B entries.
        
        # To compute X*Y (2x2 result), use Strassen 2x2x2.
        # The Strassen 2x2x2 U,V,W matrices (from strassen_2x2x2()) tell us:
        # For each of the 7 sub-multiplications s:
        #   U_s gives coefficients for X entries
        #   V_s gives coefficients for Y entries  
        #   W_s gives coefficients for output entries
        
        # So for top-level m and sub-level s:
        # U4[:, m*7+s] = sum over X entries: U_strassen[s, x_idx] * coeff_X[x_idx]
        # where coeff_X tells us how X entries relate to A entries.
        
        # This is a matrix multiplication of coefficient matrices.
        
        # Let me build it step by step.
        
        # First, define the 2x2 block matrices in terms of A and B entries.
        
        # A entry indices in U4: A[i,k] -> index i*4+k
        # B entry indices in V4: B[k,j] -> index k*4+j
        
        # For each m, define X_m as a 2x2 matrix of linear forms in A entries.
        # X_m[i,j] = sum_{a,k} coeff_X[m,i,j,a,k] * A[a,k]
        
        # Similarly for Y_m.
        
        # Then for sub-Strassen s:
        # U4[:, m*7+s] = sum_{x_idx} sU[s, x_idx] * (coefficients of X_m[x_idx] in A)
        
        # Where x_idx maps to X_m entries: x_idx=0 -> X[0,0], x_idx=1 -> X[0,1], etc.
        
        # Let me just build the full coefficient matrices.
        
        # X_m entries (2x2):
        # For m=0: X = A00+A11
        #   X[0,0] = A[0,0]+A[2,2], X[0,1] = A[0,1]+A[2,3]
        #   X[1,0] = A[1,0]+A[3,2], X[1,1] = A[1,1]+A[3,3]
        
        # I'll build this programmatically.
        
        # A block extraction:
        # A00[i,j] = A[i,j] for i,j in 0,1
        # A01[i,j] = A[i,j+2] for i,j in 0,1
        # A10[i,j] = A[i+2,j] for i,j in 0,1
        # A11[i,j] = A[i+2,j+2] for i,j in 0,1
        
        # B00[i,j] = B[i,j] for i,j in 0,1
        # B01[i,j] = B[i,j+2] for i,j in 0,1
        # B10[i,j] = B[i+2,j] for i,j in 0,1
        # B11[i,j] = B[i+2,j+2] for i,j in 0,1
        
        # For each m, define X_m and Y_m as 2x2 matrices of linear forms.
        
        # Let me use a different approach: just build the 49-column decomposition directly.
        
        # For each of the 7 top-level Strassen combinations m:
        #   X_m is a 2x2 matrix, Y_m is a 2x2 matrix.
        #   Compute X_m * Y_m using Strassen 2x2x2.
        #   The result is a 2x2 matrix that contributes to specific C blocks.
        
        # The contribution to C:
        # m=0: R0 = X0*Y0, C00 += R0, C01 += R0, C10 += R0, C11 += R0 (with signs)
        # Actually: C00 += R0, C11 += R0, and from the Strassen formula:
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        # So:
        # C00 gets: +M0, +M3, -M4, +M6
        # C01 gets: +M2, +M4
        # C10 gets: +M1, +M3
        # C11 gets: +M0, -M1, +M2, +M5
        
        # Each Mx is a 2x2 matrix.
        
        # C00[i,j] = M0[i,j] + M3[i,j] - M4[i,j] + M6[i,j] for i,j in 0,1
        # C01[i,j] = M2[i,j] + M4[i,j] for i,j in 0,1
        # C10[i,j] = M1[i,j] + M3[i,j] for i,j in 0,1
        # C11[i,j] = M0[i,j] - M1[i,j] + M2[i,j] + M5[i,j] for i,j in 0,1
        
        # Now, Mx = Xx * Yx, and we compute this with Strassen 2x2x2.
        # For each m, the Strassen 2x2x2 gives 7 sub-multiplications.
        # For sub-multiplication s of Mx:
        #   The "scalar" multiplication is: (sum of X entries with sU coefficients) * (sum of Y entries with sV coefficients)
        #   The result contributes to Mx entries with sW coefficients.
        
        # So for top-level m, sub-level s:
        # U4[:, m*7+s] = (linear form for X_m using sU)
        # V4[:, m*7+s] = (linear form for Y_m using sV)
        # W4[:, m*7+s] = (contribution to C using sW and the C-M mapping)
        
        # Let me build this.
        
        # First, for each m, get the 2x2 matrices X_m and Y_m as linear forms.
        
        # X_m[i,j] in terms of A entries:
        def get_X(m, i, j):
            """Return dict of {A_index: coeff} for X_m[i,j]."""
            # A_index = a*4+k for A[a,k]
            d = {}
            if m == 0:  # A00+A11
                d[i*4+j] = 1
                d[(i+2)*4+(j+2)] = 1
            elif m == 1:  # A10+A11
                d[(i+2)*4+j] = 1
                d[(i+2)*4+(j+2)] = 1
            elif m == 2:  # A00
                d[i*4+j] = 1
            elif m == 3:  # A11
                d[(i+2)*4+(j+2)] = 1
            elif m == 4:  # A00+A01
                d[i*4+j] = 1
                d[i*4+(j+2)] = 1
            elif m == 5:  # A10-A00
                d[(i+2)*4+j] = 1
                d[i*4+j] = -1
            elif m == 6:  # A01-A11
                d[i*4+(j+2)] = 1
                d[(i+2)*4+(j+2)] = -1
            return d
        
        def get_Y(m, i, j):
            """Return dict of {B_index: coeff} for Y_m[i,j]."""
            # B_index = k*4+j for B[k,j]
            d = {}
            if m == 0:  # B00+B11
                d[i*4+j] = 1
                d[(i+2)*4+(j+2)] = 1
            elif m == 1:  # B00
                d[i*4+j] = 1
            elif m == 2:  # B01-B11
                d[i*4+(j+2)] = 1
                d[(i+2)*4+(j+2)] = -1
            elif m == 3:  # B10-B00
                d[(i+2)*4+j] = 1
                d[i*4+j] = -1
            elif m == 4:  # B11
                d[(i+2)*4+(j+2)] = 1
            elif m == 5:  # B00+B01
                d[i*4+j] = 1
                d[i*4+(j+2)] = 1
            elif m == 6:  # B10+B11
                d[(i+2)*4+j] = 1
                d[(i+2)*4+(j+2)] = 1
            return d
        
        # C block mapping: which Mx contributes to which C block with what sign.
        # C block (ci, cj) where ci, cj in 0,1 (block index), local (li, lj) in 0,1.
        # C[ci*2+li, cj*2+lj] = sum over m: sign[m, ci, cj] * Mx[m][li, lj]
        
        # C00 (ci=0,cj=0): M0(+1), M3(+1), M4(-1), M6(+1)
        # C01 (ci=0,cj=1): M2(+1), M4(+1)
        # C10 (ci=1,cj=0): M1(+1), M3(+1)
        # C11 (ci=1,cj=1): M0(+1), M1(-1), M2(+1), M5(+1)
        
        c_signs = {
            (0,0): {0:1, 3:1, 4:-1, 6:1},
            (0,1): {2:1, 4:1},
            (1,0): {1:1, 3:1},
            (1,1): {0:1, 1:-1, 2:1, 5:1},
        }
        
        # Now build U4, V4, W4.
        for m in range(7):
            for s in range(7):
                col = m * 7 + s
                
                # U4: linear form for X_m using sU
                # sU[s, x_idx] where x_idx=0->X[0,0], 1->X[0,1], 2->X[1,0], 3->X[1,1]
                for x_idx in range(4):
                    if sU[s, x_idx] != 0:
                        xi, xj = divmod(x_idx, 2)
                        coeff_x = sU[s, x_idx]
                        for a_idx, coeff_a in get_X(m, xi, xj).items():
                            U4[a_idx, col] += coeff_x * coeff_a
                
                # V4: linear form for Y_m using sV
                for y_idx in range(4):
                    if sV[s, y_idx] != 0:
                        yi, yj = divmod(y_idx, 2)
                        coeff_y = sV[s, y_idx]
                        for b_idx, coeff_b in get_Y(m, yi, yj).items():
                            V4[b_idx, col] += coeff_y * coeff_b
                
                # W4: contribution to C using sW
                # sW[s, c_idx] where c_idx=0->C[0,0], 1->C[1,0], 2->C[0,1], 3->C[1,1]
                # c_idx maps to (li, lj): c_idx=0->(0,0), 1->(1,0), 2->(0,1), 3->(1,1)
                for c_idx in range(4):
                    if sW[s, c_idx] != 0:
                        li, lj = (c_idx // 2, c_idx % 2) if False else ((c_idx % 2), (c_idx // 2))
                        # Wait, sW indexing: W[k*n+i] for 2x2x2 with n=2.
                        # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1]
                        # So c_idx=0: i=0,j=0; c_idx=1: i=1,j=0; c_idx=2: i=0,j=1; c_idx=3: i=1,j=1
                        li = c_idx % 2  # i
                        lj = c_idx // 2  # j
                        coeff_w = sW[s, c_idx]
                        # This contributes to C block (ci, cj) where the local position is (li, lj)
                        # But which C block? The Mx result is a 2x2 matrix that goes into
                        # specific C blocks. For Mx, the 2x2 result R has R[li, lj].
                        # R[li, lj] contributes to C[ci*2+li, cj*2+lj] for each (ci,cj) with sign.
                        for (ci, cj), signs in c_signs.items():
                            if m in signs:
                                c_global_i = ci * 2 + li
                                c_global_j = cj * 2 + lj
                                w_idx = c_global_j * 4 + c_global_i  # W[j*4+i]
                                W4[w_idx, col] += coeff_w * signs[m]
        
        return U4, V4, W4
    
    # Build the full 4x4x8 decomposition.
    U4, V4, W4 = strassen_4x4x4()
    
    # For 4x4x8: A is 4x4, B is 4x8, C is 4x8.
    # Split B into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # C_left = A * B_left (4x4), C_right = A * B_right (4x4).
    
    rank_total = 98
    U_full = np.zeros((16, rank_total), dtype=int)
    V_full = np.zeros((32, rank_total), dtype=int)
    W_full = np.zeros((32, rank_total), dtype=int)
    
    # Left half (cols 0-3 of B, cols 0-3 of C):
    U_full[:, 0:49] = U4
    V_full[0:16, 0:49] = V4  # B_left[k,j] = B[k,j] for j=0..3, index k*4+j
    # W for C_left: C[i,j] for j=0..3, W[j*4+i]
    W_full[0:16, 0:49] = W4
    
    # Right half (cols 4-7 of B, cols 4-7 of C):
    U_full[:, 49:98] = U4
    # B_right[k,j] = B[k,j] for j=4..7, index k*8+(j) in V_full
    # V_full index: k*8 + j for B[k,j]
    # B_right[k, j_local] = B[k, 4+j_local], V_full index = k*8 + (4+j_local)
    for k in range(4):
        for j in range(4):
            v4_idx = k*4 + j  # index in V4
            vf_idx = k*8 + (4 + j)  # index in V_full
            V_full[vf_idx, 49:98] = V4[v4_idx, :]
    
    # W for C_right: C[i,j] for j=4..7, W[j*4+i] in W_full
    # W_full index: j*4 + i for C[i,j]
    for j in range(4):
        for i in range(4):
            w4_idx = j*4 + i  # index in W4
            wf_idx = (4+j)*4 + i  # index in W_full
            W_full[wf_idx, 49:98] = W4[w4_idx, :]
    
    # Convert to lists
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if time.time() > deadline:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

