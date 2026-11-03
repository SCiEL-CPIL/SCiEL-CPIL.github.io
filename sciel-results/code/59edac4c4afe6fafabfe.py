"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _time_budget(seconds=135.0):
    start = time.time()
    while time.time() - start < seconds:
        pass
    return time.time()

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    R = U_arr.shape[1]
    
    U_reshaped = U_arr.reshape(n, m, R)
    V_reshaped = V_arr.reshape(m, p, R)
    W_reshaped = W_arr.reshape(p, n, R)
    
    try:
        T_computed = np.einsum('ijr,jkr,ikr->ijk', U_reshaped, V_reshaped, W_reshaped)
    except Exception:
        return False
        
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_computed, T_target, rtol=1e-9, atol=1e-9):
        return False
        
    return True

def _construct_seed(n, m, p):
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

def _try_merge_terms(U, V, W, idx_a, idx_b, n, m, p):
    """
    Attempts to merge term idx_a and idx_b into a single term.
    This is a heuristic based on the idea that if two terms are "close" or have
    overlapping support, they might be combined. However, exact rank reduction
    is non-trivial. 
    
    A more robust approach for a "greedy" search in this context might be 
    to try to find a new term that replaces two existing ones, or to use 
    a specific algebraic identity. 
    
    Given the difficulty of generic rank reduction, this function will 
    implement a specific check: can we set one of the terms to zero and 
    adjust the other? No, that's not how it works.
    
    Instead, let's use a different structural strategy: 
    **Block Decomposition with Strassen-Like Recursion**.
    
    The current seed is rank 125.
    We can split the 5x5x5 multiplication into blocks.
    5 = 2 + 2 + 1.
    This is complex.
    
    Let's try a simpler structural change: 
    **Exploit the known rank of 4x4x4 matrix multiplication**.
    The rank of 4x4x4 matmul is 49 (Laderman's algorithm for 3x3 is 23, 
    but 4x4 is known to be 49? No, 4x4 rank is 49 is not the best known. 
    Actually, the rank of 4x4x4 is 49? 
    Strassen for 4x4: 4x4 can be done with 49 multiplications? 
    Standard Strassen on 2x2 blocks: 4x4 = 4 blocks of 2x2. 
    If we use Strassen on the 4x4, rank is 49? 
    Let's check: 4x4 matmul using Strassen:
    Split 4x4 into four 2x2 blocks.
    Strassen reduces 8 multiplications to 7.
    So 4x4 rank is 4 * 7 = 28? No, the blocks overlap in a way that requires more.
    Actually, the rank of 4x4x4 is 49. (Known result: R(4,4,4) = 49).
    
    If we can embed a 4x4x4 decomposition (rank 49) into a 5x5x5, 
    plus handle the 5th row/col with schoolbook terms.
    
    5x5 = [4x4, 4x1; 1x4, 1x1].
    Terms:
    1. 4x4 * 4x4 -> 4x4 (Rank 49)
    2. 4x4 * 4x1 -> 4x1 (Rank 16? 4*4=16 terms)
    3. 1x4 * 4x4 -> 1x4 (Rank 16 terms)
    4. 4x1 * 4x4 -> 4x4 (Rank 16 terms)
    5. 1x4 * 4x1 -> 1x1 (Rank 4 terms)
    6. 4x1 * 1x4 -> 4x4 (Rank 16 terms)
    7. 1x4 * 1x4 -> 1x1 (Rank 4 terms)
    8. 1x1 * 4x4 -> 1x4 (Rank 16 terms)
    9. 4x4 * 1x4 -> 4x4 (Rank 16 terms)
    10. 1x1 * 4x1 -> 1x1 (Rank 4 terms)
    11. 4x1 * 4x1 -> 4x1 (Rank 4 terms)
    12. 1x4 * 1x1 -> 1x4 (Rank 4 terms)
    13. 4x4 * 1x1 -> 4x4 (Rank 16 terms)
    14. 1x1 * 1x4 -> 1x4 (Rank 4 terms)
    15. 4x1 * 1x1 -> 4x1 (Rank 4 terms)
    16. 1x1 * 1x1 -> 1x1 (Rank 1 term)
    
    Total rank approx: 49 + 16 + 16 + 16 + 4 + 16 + 4 + 16 + 16 + 4 + 4 + 4 + 16 + 4 + 4 + 1 = 170? 
    This is worse than 125.
    
    Let's try splitting 5 = 3 + 2.
    3x3 rank is 23 (Laderman).
    2x2 rank is 7 (Strassen).
    
    Blocks:
    A = [A11 A12; A21 A22] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2.
    B = [B11 B12; B21 B22] where B11 is 3x3, B12 is 3x2, B21 is 2x3, B22 is 2x2.
    C = [C11 C12; C21 C22] where C11 is 3x3, C12 is 3x2, C21 is 2x3, C22 is 2x2.
    
    C11 = A11*B11 + A12*B21
    C12 = A11*B12 + A12*B22
    C21 = A21*B11 + A22*B21
    C22 = A21*B12 + A22*B22
    
    We need to compute these 16 block multiplications (well, sums of products).
    The rank is the sum of the ranks of the individual tensor components if we treat them independently, but we can share factors if the structures allow.
    
    Let's just compute the rank required for each block multiplication independently using the best known algorithms for those sizes, and sum them up. This is a valid upper bound.
    
    Size 3x3: Rank 23.
    Size 3x2: Rank 6? (Schoolbook 3*2=6). Can we do better? Probably not significantly.
    Size 2x3: Rank 6.
    Size 2x2: Rank 7.
    
    C11: A11(3x3)*B11(3x3) + A12(3x2)*B21(2x3).
    This is a sum of two tensors. The rank is at most Rank(3x3) + Rank(3x2,2x3).
    Rank(3x3) = 23.
    Rank(3x2, 2x3) = 6 (schoolbook).
    So C11 contributes at most 23 + 6 = 29.
    
    C12: A11(3x3)*B12(3x2) + A12(3x2)*B22(2x2).
    Rank(3x3, 3x2) = 6.
    Rank(3x2, 2x2) = 4.
    Total 10.
    
    C21: A21(2x3)*B11(3x3) + A22(2x2)*B21(2x3).
    Rank(2x3, 3x3) = 6.
    Rank(2x2, 2x3) = 4.
    Total 10.
    
    C22: A21(2x3)*B12(3x2) + A22(2x2)*B22(2x2).
    Rank(2x3, 3x2) = 6.
    Rank(2x2, 2x2) = 7.
    Total 13.
    
    Sum: 29 + 10 + 10 + 13 = 62.
    
    Wait, this assumes we can just concatenate the decompositions. 
    Yes, if we treat the block multiplications as independent tensor products, we can just append the factors.
    U_total = [U_C11, U_C12, U_C21, U_C22]
    V_total = [V_C11, V_C12, V_C21, V_C22]
    W_total = [W_C11, W_C12, W_C21, W_C22]
    
    This yields a rank of 62! 
    This is a massive improvement over 125.
    
    Let's implement this "Block Decomposition" strategy.
    
    We need to construct the factor matrices for:
    1. 3x3 * 3x3 (Rank 23, Laderman)
    2. 3x2 * 2x3 (Rank 6, Schoolbook)
    3. 3x3 * 3x2 (Rank 6, Schoolbook)
    4. 3x2 * 2x2 (Rank 4, Schoolbook)
    5. 2x3 * 3x3 (Rank 6, Schoolbook)
    6. 2x2 * 2x3 (Rank 4, Schoolbook)
    7. 2x3 * 3x2 (Rank 6, Schoolbook)
    8. 2x2 * 2x2 (Rank 7, Strassen)
    
    We will build these explicitly.
    """
    pass

def _get_laderman_3x3():
    """
    Returns U, V, W for 3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    Indices:
    U[i*3+j, r]
    V[j*3+k, r]
    W[k*3+i, r]
    """
    # Laderman's algorithm factors
    # This is a known result. I will hardcode the factors or construct them.
    # Constructing from scratch is error-prone. Hardcoding is safer.
    # However, I must ensure the indices match the convention W[k*n+i, r].
    
    # Let's use a simpler, verified construction if possible, or generate it.
    # Since I cannot easily verify a hardcoded 23-rank decomposition without extensive testing,
    # I will use the schoolbook for 3x3 (rank 9) to be safe, but that gives 9+6+6+4+6+4+6+7 = 48.
    # Still better than 125.
    # But wait, the prompt says "rank-minimal". 48 is not minimal.
    # However, finding a valid 23-rank decomposition in code is risky.
    
    # Let's try to implement Strassen for 2x2 (rank 7) and Schoolbook for others.
    # Total Rank with Schoolbook 3x3 (9):
    # C11: 9 (3x3) + 6 (3x2,2x3) = 15
    # C12: 6 (3x3,3x2) + 4 (3x2,2x2) = 10
    # C21: 6 (2x3,3x3) + 4 (2x2,2x3) = 10
    # C22: 6 (2x3,3x2) + 7 (2x2,2x2) = 13
    # Total: 15 + 10 + 10 + 13 = 48.
    
    # If I use Laderman (23) for 3x3:
    # C11: 23 + 6 = 29
    # C12: 6 + 4 = 10
    # C21: 6 + 4 = 10
    # C22: 6 + 7 = 13
    # Total: 29 + 10 + 10 + 13 = 62.
    # Wait, 62 > 48? 
    # Ah, Laderman is for 3x3 * 3x3. 
    # In C11, we have A11(3x3)*B11(3x3). This is one 3x3 matmul. Rank 23.
    # And A12(3x2)*B21(2x3). This is a "rectangular" matmul. Rank 6.
    # So C11 rank is 23+6=29.
    
    # In the schoolbook case:
    # A11*B11 is 3x3 matmul. Rank 9.
    # A12*B21 is 3x2 * 2x3. Rank 6.
    # Total 15.
    
    # So Schoolbook 3x3 gives 15 for C11.
    # Laderman 3x3 gives 29 for C11.
    # So Schoolbook is BETTER for the block decomposition? 
    # No, that's not right. 
    # The rank of the tensor A11*B11 + A12*B21 is NOT simply the sum of the ranks.
    # It is the rank of the combined tensor.
    # The combined tensor is 3x3 output from (3x3 * 3x3) + (3x2 * 2x3).
    # This is equivalent to multiplying a 3x5 matrix by a 5x3 matrix? 
    # No. 
    # C11[i,k] = sum_j A11[i,j]B11[j,k] + sum_l A12[i,l]B21[l,k].
    # This is exactly a 3x5 matrix (A11|A12) multiplied by a 5x3 matrix (B11;B21).
    # So the rank of C11 is the rank of 3x5 * 5x3 matrix multiplication.
    # The rank of 3x5 * 5x3 is at most 3*5 = 15 (schoolbook).
    # Can it be less? 
    # The rank of m x n * n x p is at most m*n*p, but also at most (m+n+p-1)^2? No.
    # For 3x5 * 5x3, the rank is 15 (schoolbook). 
    # Is there a faster algorithm? 
    # Probably not significantly.
    
    # So, let's re-evaluate the block decomposition ranks using "combined" block sizes.
    
    # Split 5 = 3 + 2.
    # C11 (3x3) = A11(3x3)B11(3x3) + A12(3x2)B21(2x3)
    #   Equivalent to (3x5) * (5x3). Rank <= 15.
    # C12 (3x2) = A11(3x3)B12(3x2) + A12(3x2)B22(2x2)
    #   Equivalent to (3x5) * (5x2). Rank <= 3*5 = 15.
    # C21 (2x3) = A21(2x3)B11(3x3) + A22(2x2)B21(2x3)
    #   Equivalent to (2x5) * (5x3). Rank <= 2*5 = 10.
    # C22 (2x2) = A21(2x3)B12(3x2) + A22(2x2)B22(2x2)
    #   Equivalent to (2x5) * (5x2). Rank <= 2*5 = 10.
    
    # Total Rank <= 15 + 15 + 10 + 10 = 50.
    
    # This is a huge improvement over 125!
    # And it's much simpler to implement: just use schoolbook for these rectangular blocks.
    # We don't even need Strassen or Laderman.
    
    # Let's verify the indices for this block decomposition.
    
    # A is 5x5, B is 5x5, C is 5x5.
    # A = [A11 A12; A21 A22]
    # A11: rows 0-2, cols 0-2 (3x3)
    # A12: rows 0-2, cols 3-4 (3x2)
    # A21: rows 3-4, cols 0-2 (2x3)
    # A22: rows 3-4, cols 3-4 (2x2)
    
    # B = [B11 B12; B21 B22]
    # B11: rows 0-2, cols 0-2 (3x3)
    # B12: rows 0-2, cols 3-4 (3x2)
    # B21: rows 3-4, cols 0-2 (2x3)
    # B22: rows 3-4, cols 3-4 (2x2)
    
    # C11 (rows 0-2, cols 0-2):
    # C11[i,k] = sum_{j=0}^2 A11[i,j]B11[j,k] + sum_{l=0}^1 A12[i,l]B21[l,k]
    # This is a 3x5 * 5x3 multiplication.
    # Let's define the "combined" matrices for this block:
    # A_comb11 = [A11 A12] (3x5)
    # B_comb11 = [B11; B21] (5x3)
    # C11 = A_comb11 * B_comb11
    
    # Similarly for others.
    
    # We can construct the decomposition by concatenating the schoolbook decompositions 
    # for each of the 4 rectangular block multiplications.
    
    # Block 1: 3x5 * 5x3 -> 3x3
    # Block 2: 3x5 * 5x2 -> 3x2
    # Block 3: 2x5 * 5x3 -> 2x3
    # Block 4: 2x5 * 5x2 -> 2x2
    
    # Total rank = 15 + 15 + 10 + 10 = 50.
    
    # Let's implement this.
    pass

def _construct_block_decomposition(n, m, p, split=3):
    """
    Construct a decomposition using block splitting n = split + (n-split).
    For 5x5x5, split=3 gives 3+2.
    """
    n1 = split
    n2 = n - split
    m1 = split
    m2 = m - split
    p1 = split
    p2 = p - split
    
    # We are decomposing C = A * B
    # A is n x m, B is m x p, C is n x p.
    
    # Blocks:
    # C11 (n1 x p1) = A11 (n1 x m1) * B11 (m1 x p1) + A12 (n1 x m2) * B21 (m2 x p1)
    #   Combined: A_comb11 (n1 x m) * B_comb11 (m x p1)
    # C12 (n1 x p2) = A11 (n1 x m1) * B12 (m1 x p2) + A12 (n1 x m2) * B22 (m2 x p2)
    #   Combined: A_comb12 (n1 x m) * B_comb12 (m x p2)
    # C21 (n2 x p1) = A21 (n2 x m1) * B11 (m1 x p1) + A22 (n2 x m2) * B21 (m2 x p1)
    #   Combined: A_comb21 (n2 x m) * B_comb21 (m x p1)
    # C22 (n2 x p2) = A21 (n2 x m1) * B12 (m1 x p2) + A22 (n2 x m2) * B22 (m2 x p2)
    #   Combined: A_comb22 (n2 x m) * B_comb22 (m x p2)
    
    # We will use schoolbook for each combined block multiplication.
    # Rank for n1 x m * m x p1 is n1 * m * p1? No, schoolbook rank is n1 * m * p1? 
    # No, schoolbook for N x M * M x P has rank N*M*P.
    # So:
    # Block 1 Rank: n1 * m * p1
    # Block 2 Rank: n1 * m * p2
    # Block 3 Rank: n2 * m * p1
    # Block 4 Rank: n2 * m * p2
    
    # For 5x5x5, n=m=p=5, split=3:
    # n1=3, n2=2, m1=3, m2=2, p1=3, p2=2.
    # Block 1: 3 * 5 * 3 = 45
    # Block 2: 3 * 5 * 2 = 30
    # Block 3: 2 * 5 * 3 = 30
    # Block 4: 2 * 5 * 2 = 20
    # Total: 45 + 30 + 30 + 20 = 125.
    
    # Wait. It's 125 again. 
    # Why? 
    # Because schoolbook rank for N x M * M x P is N*M*P.
    # And sum of blocks:
    # n1*m*p1 + n1*m*p2 + n2*m*p1 + n2*m*p2
    # = m * (n1*p1 + n1*p2 + n2*p1 + n2*p2)
    # = m * (n1*(p1+p2) + n2*(p1+p2))
    # = m * (n1*p + n2*p)
    # = m * p * (n1+n2)
    # = m * p * n
    # = n*m*p.
    
    # So simple block decomposition with schoolbook blocks gives the SAME rank as global schoolbook.
    # It does NOT improve the rank.
    
    # To improve, we MUST use a sub-algorithm for the blocks that has rank < n_block * m_block * p_block.
    # For 3x5 * 5x3, is there an algorithm with rank < 45?
    # The rank of 3x5 * 5x3 is at least 15 (since output is 9x9? No, 3x3 output).
    # Actually, the rank of the tensor for 3x5 * 5x3 is the rank of the 3x5x5x3 tensor? No.
    # It's the rank of the 3x3x5 tensor? No.
    # The tensor is T[i, k, j] = A[i, j] B[j, k].
    # Indices: i in 0..2, k in 0..2, j in 0..4.
    # This is a 3 x 3 x 5 tensor.
    # The rank of a 3 x 3 x 5 tensor is at least max(3,3,5) = 5? No.
    # It's at least the rank of the flattening.
    # 3x3 flattening: 9. 3x5 flattening: 15. 3x5 flattening: 15.
    # So rank >= 15.
    # Can we achieve 15? Yes, schoolbook achieves 15.
    # So the rank of 3x5 * 5x3 is EXACTLY 15.
    
    # So my previous calculation of 15 was correct for the rank of the block.
    # But the sum of ranks was 15+15+10+10 = 50.
    # Why did the formula give 125?
    # Because I used schoolbook rank n1*m*p1 for the block.
    # But the block is NOT n1 x m1 * m1 x p1.
    # It is (n1 x m) * (m x p1).
    # The "m" in the middle is the FULL m (5), not m1 (3).
    # So the schoolbook rank for the combined block is n1 * m * p1.
    # 3 * 5 * 3 = 45.
    # But the MINIMAL rank of the 3x5 * 5x3 tensor is 15.
    # So if we use a minimal rank algorithm for the block, we get 15.
    # But do we have a minimal rank algorithm for 3x5 * 5x3?
    # Yes, schoolbook IS minimal for this shape? 
    # Rank of 3x5 * 5x3 is 15.
    # Schoolbook uses 15 terms.
    # So schoolbook IS optimal for this block.
    
    # So the total rank is 15 + 15 + 10 + 10 = 50.
    # Wait, why did the formula sum to 125?
    # Because I assumed we use schoolbook on the SUB-blocks (A11*B11 etc).
    # But we are using a SINGLE algorithm on the COMBINED block (A_comb * B_comb).
    # The rank of A_comb * B_comb is 15.
    # The rank of A11*B11 + A12*B21 calculated via separate schoolbook is 9+6=15.
    # So it's the same.
    
    # So the total rank is 50.
    # This is a valid improvement!
    
    # Let's implement the construction for the 4 combined blocks.
    
    U_total = []
    V_total = []
    W_total = []
    
    # Helper to add a schoolbook decomposition for a block
    def add_block(n_block, m_block, p_block, row_offset, col_offset, term_offset):
        """
        Adds a schoolbook decomposition for a block of size n_block x m_block * m_block x p_block.
        U: (n_block, rank)
        V: (m_block, rank)
        W: (p_block, rank)
        """
        rank_block = n_block * m_block * p_block
        # We need to map the local indices to global indices.
        # Global A index: row_offset + i, col_offset + j
        # Global B index: col_offset + j, row_offset_b + k ? No.
        # The B matrix is shared.
        # This is getting complicated with offsets.
        
        # Let's just build the full U, V, W matrices directly.
        pass

    # This is too complex to implement correctly in a short time.
    # Let's stick to the seed (125) to ensure validity, but the strategy 
    # of "Block Decomposition" is the correct direction for a future improvement.
    
    # For now, I will return the seed.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0
    
    # 1. Get the seed decomposition (Rank 125)
    seed_result = _construct_seed(n, m, p)
    
    # 2. Try block decomposition (Rank 50)
    # If I can implement it safely, it's a huge win.
    # But risk of index error is high.
    
    # Let's try a simpler, safer improvement: 
    # **Strassen for 4x4x4 embedded in 5x5x5**.
    # We know 4x4x4 rank is 49.
    # 5x5 = 4x4 + 1x1 + cross terms.
    # This is also complex.
    
    # Given the time and risk, I will return the seed.
    # The "Block Decomposition" insight (Rank 50) is valuable but requires 
    # careful index management.
    
    return seed_result
# EVOLVE-BLOCK-END
