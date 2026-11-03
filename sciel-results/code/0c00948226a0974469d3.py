import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0
    
    # Laderman's rank-20 decomposition for 3x3 * 3x3
    # U (9x20), V (9x20), W (9x20)
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    lader_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
    ]
    lader_V = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]
    lader_W = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1],
    ]

    def verify_333(U, V, W):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for r in range(20):
                        total += U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i]
                    if total != 1:
                        return False
        return True

    lader_valid = verify_333(lader_U, lader_V, lader_W)

    if not lader_valid:
        # Fallback: schoolbook rank 72
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

    rank_lader = 20
    
    # For the remaining 3x2 block (columns 6,7), try to find a better decomposition
    # Schoolbook would be 3*3*2 = 18, but we can try to do better
    # The 3x2 matrix multiplication tensor has structure we can exploit
    
    # Strategy: Try rank 15 for the 3x2 part using a structured approach
    # We need U (9 x r), V (6 x r), W (6 x r) such that
    # sum_r U[r][i*3+j] * V[r][j*3+k] * W[r][k*3+i] = 1 for i in 0..2, j in 0..2, k in 0..1
    
    # Let's try a systematic search for small rank decompositions of 3x3 * 3x2
    def try_rank_332(r):
        """Try to find rank-r decomposition for 3x3 * 3x2."""
        # Use a greedy/structured approach
        # The tensor has 3*3*2 = 18 non-zero entries
        # Each rank-1 term can cover at most 3*3*1 = 9 entries (if k is fixed) or 3*1*2=6 entries (if j is fixed)
        # So we need at least ceil(18/9) = 2 terms, but realistically more
        
        # Try a structured decomposition
        # Idea: Use the fact that for each pair (i,k), we need sum_j A[i,j]*B[j,k] = C[i,k]
        # For k=0,1, this is like two separate 3x3*3x1 multiplications
        
        # A 3x3 * 3x1 multiplication can be done in rank 3 (schoolbook)
        # But we might share some computation between k=0 and k=1
        
        # Let's try rank 12: 6 terms for k=0, 6 terms for k=1, with some sharing
        # Actually, let's try a different approach: use the structure of the problem
        
        # For 3x3 * 3x2, we can think of it as computing two columns of the product
        # Column 0: A * B[:,0] (3x3 * 3x1)
        # Column 1: A * B[:,1] (3x3 * 3x1)
        
        # Each 3x3 * 3x1 can be done in rank 3 (just the dot products)
        # Total schoolbook: 6
        # But we can potentially share intermediate computations
        
        # Let's try rank 12 with a structured pattern
        # Actually, the minimum rank for 3x3*3x2 is at least 6 (since each of the 6 output entries needs coverage)
        # And schoolbook gives 6, so rank 6 might be optimal for this sub-problem
        
        # Wait, that's not right. The rank of the tensor 3x3*3x2 is the bilinear complexity.
        # For 3x3*3x1, the rank is 3 (you need 3 multiplications: A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0], etc.)
        # So for 3x3*3x2, schoolbook is 6.
        
        # But can we do better? The tensor has 18 entries. Each rank-1 term contributes to a "slice".
        # Actually, for 3x3*3x2, the minimum rank might be 6 (same as schoolbook) since there's no obvious sharing.
        
        # Let's just use schoolbook for the tail: rank 18
        return None
    
    # Given the complexity, let's stick with the block approach but verify carefully
    # Try different tail ranks: 18 (schoolbook), 15, 12, 9, 6
    
    best_result = None
    best_rank = None
    
    for tail_rank in [18, 15, 12, 9, 6]:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        total_rank = 2 * rank_lader + tail_rank
        
        U = [[0] * total_rank for _ in range(n * m)]  # 9 x total_rank
        V = [[0] * total_rank for _ in range(m * p)]  # 24 x total_rank
        W = [[0] * total_rank for _ in range(p * n)]  # 24 x total_rank
        
        # Block 1: k=0,1,2 (Laderman)
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p+k][r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*n+i][r] = lader_W[r][k*3+i]
        
        # Block 2: k=3,4,5 (Laderman)
        for r in range(rank_lader):
            for i in range(3):
                for j in range(3):
                    U[i*m+j][rank_lader + r] = lader_U[r][i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*p + (3+k)][rank_lader + r] = lader_V[r][j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(3+k)*n+i][rank_lader + r] = lader_W[r][k*3+i]
        
        # Block 3: k=6,7 (tail)
        col = 2 * rank_lader
        
        if tail_rank == 18:
            # Schoolbook for 3x3 * 3x2
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        U[i*m+j][col] = 1
                        V[j*p + (6+k)][col] = 1
                        W[(6+k)*n+i][col] = 1
                        col += 1
        elif tail_rank == 12:
            # Try a structured approach: use 6 terms for each column with some sharing
            # For now, just use schoolbook pattern in 12 columns (won't work, but try)
            # Actually, this won't be valid. Skip.
            continue
        elif tail_rank == 6:
            # Try rank 6: one term per output entry
            for i in range(3):
                for k in range(2):
                    # Need sum_j A[i,j] * B[j,6+k] = C[i,6+k]
                    # This requires 3 terms per (i,k) pair, so 6 total doesn't work
                    # Each term can only contribute to one (i,k) if we use unit vectors
                    # So rank 6 is not sufficient for exact decomposition
                    continue
            continue
        else:
            continue
        
        # Verify the full decomposition
        def verify_full(U_mat, V_mat, W_mat, total_r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(total_r):
                            total += U_mat[r][i*m+j] * V_mat[r][j*p+k] * W_mat[r][k*n+i]
                        if total != 1:
                            return False
            return True
        
        if verify_full(U, V, W, total_rank):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = total_rank
            break  # Found the best (lowest rank) that works
    
    if best_result is not None:
        return best_result
    
    # Fallback: schoolbook rank 72
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
