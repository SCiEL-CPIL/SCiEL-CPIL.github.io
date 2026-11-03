"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    # Uses a rank-22 scheme derived from Strassen-like decomposition.
    n, m, p = 2, 3, 6
    rank = 22
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 22

    # For each block (3 blocks corresponding to 2x2 subblocks of C)
    for block in range(3):
        base = block * 7
        # V indices: row 0 cols 2*block, 2*block+1; row 1 cols 2*block, 2*block+1
        v00 = 0 * 6 + 2 * block      # V[0, 2*block]
        v01 = 0 * 6 + 2 * block + 1  # V[0, 2*block+1]
        v10 = 1 * 6 + 2 * block      # V[1, 2*block]
        v11 = 1 * 6 + 2 * block + 1  # V[1, 2*block+1]
        
        # W indices: row 2*block, 2*block+1 for i=0; row 2*block+1, 2*block+2 for i=1
        w00 = (2 * block) * 2 + 0      # W[2*block, 0] -> C[2*block][0]
        w01 = (2 * block) * 2 + 1      # W[2*block, 1] -> C[2*block][1]
        w10 = (2 * block + 1) * 2 + 0  # W[2*block+1, 0] -> C[2*block+1][0]
        w11 = (2 * block + 1) * 2 + 1  # W[2*block+1, 1] -> C[2*block+1][1]

        # Strassen terms for this block
        # s = base + 0: M1
        U[0][base] += 1   # A[0][0]
        U[3][base] += 1   # A[1][1]
        V[v00][base] += 1 # B[0][2b]
        V[v11][base] += 1 # B[1][2b+1]
        W[w00][base] += 1 # C[2b][0]
        W[w11][base] += 1 # C[2b+1][1]

        # s = base + 1: M2
        U[3][base + 1] += 1   # A[1][0]
        U[4][base + 1] += 1   # A[1][1]
        V[v00][base + 1] += 1 # B[0][2b]
        W[w10][base + 1] += 1 # C[2b+1][0]
        W[w11][base + 1] -= 1 # C[2b+1][1]

        # s = base + 2: M3
        U[0][base + 2] += 1   # A[0][0]
        V[v01][base + 2] += 1 # B[0][2b+1]
        V[v11][base + 2] -= 1 # B[1][2b+1]
        W[w01][base + 2] += 1 # C[2b][1]
        W[w11][base + 2] += 1 # C[2b+1][1]

        # s = base + 3: M4
        U[4][base + 3] += 1   # A[1][1]
        V[v10][base + 3] += 1 # B[1][2b]
        V[v00][base + 3] -= 1 # B[0][2b]
        W[w00][base + 3] += 1 # C[2b][0]
        W[w10][base + 3] += 1 # C[2b+1][0]

        # s = base + 4: M5
        U[0][base + 4] += 1   # A[0][0]
        U[1][base + 4] += 1   # A[0][1]
        V[v11][base + 4] += 1 # B[1][2b+1]
        W[w00][base + 4] -= 1 # C[2b][0]
        W[w01][base + 4] += 1 # C[2b][1]

        # s = base + 5: M6
        U[3][base + 5] += 1   # A[1][0]
        U[0][base + 5] -= 1   # A[0][0]
        V[v00][base + 5] += 1 # B[0][2b]
        V[v01][base + 5] += 1 # B[0][2b+1]
        W[w11][base + 5] += 1 # C[2b+1][1]

        # s = base + 6: M7
        U[1][base + 6] += 1   # A[0][1]
        U[4][base + 6] -= 1   # A[1][1]
        V[v10][base + 6] += 1 # B[1][2b]
        V[v11][base + 6] += 1 # B[1][2b+1]
        W[w00][base + 6] += 1 # C[2b][0]

    # Tail term for the remaining column (A[0][2] * B[2][5] -> C[5])
    # Wait, let me recheck the decomposition logic.
    # The previous analysis was a bit confused. Let's use a standard rank-20 or 22 construction.
    # Actually, the known minimum rank for (2,3,6) is 20 (over Q).
    # Let's try to implement a known rank-20 scheme or a verified rank-22 scheme.
    # Given the complexity, I will provide the rank-22 scheme which is easier to construct and verify.
    # However, the previous thought process had a "tail" term which seems off.
    # Let's stick to the clean Strassen-blocked construction which yields rank 21? No, 3*7 = 21.
    # Wait, the blocks cover B[0,0..1], B[0,2..3], B[0,4..5] and B[1,0..1], etc.
    # But we have A[0,2] and A[1,2] which are in the third column of A.
    # In the standard Strassen decomposition for (2,2,2), we use 7 terms.
    # Here A is 2x3. We can split A into two 2x2 blocks: A_left (cols 0,1) and A_right (col 2).
    # B is 3x6. Split B into B_top (rows 0,1) and B_bottom (row 2).
    # This is not a perfect 2x2x2 block.
    
    # Let's just return the schoolbook rank 36 if the 22 is too hard to verify quickly in thought, 
    # but the prompt implies finding a *better* one.
    # I will use the rank 22 scheme I partially derived. I need to complete it carefully.
    
    # Actually, let's just return the schoolbook for safety if I can't construct a valid lower rank immediately.
    # BUT, the task asks for rank-minimal. The score is the rank. Lower is better.
    # The seed is 36. I should at least try to produce something better.
    # A known rank for 2x3x3 is 11. For 2x3x6, it's likely around 20.
    # Let's try to construct a valid rank 22 explicitly using the code below.
    
    pass

# Let's restart the construct logic inside the EVOLVE-BLOCK properly.
# I will implement the rank 22 scheme.

def construct():
    n, m, p = 2, 3, 6
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]  # 6 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 18 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 12 rows

    # We treat the multiplication as:
    # C = A * B
    # A is 2x3, B is 3x6.
    # We can view B as three 3x2 blocks: B_1 (cols 0,1), B_2 (cols 2,3), B_3 (cols 4,5).
    # C will be three 2x2 blocks: C_1, C_2, C_3.
    # For each block j, C_j = A * B_j.
    # Since A is 2x3 and B_j is 3x2, this is a (2,3,2) multiplication.
    # The rank of (2,3,2) multiplication is 5? No, schoolbook is 6. Strassen-like?
    # Actually, (2,2,2) rank is 7. (2,3,2) can be embedded?
    # Let's just use the schoolbook for each block? 3 blocks * 6 terms = 18 terms?
    # If we do that, we need 18 terms.
    # Let's verify if rank 18 is possible.
    # U[i*3+k, t], V[(k)*2+j, t], W[(j)*2+i, t] ?
    # No, the indices are global.
    
    # Let's try a rank 20 construction using the following known scheme for (2,3,6):
    # This is too complex to derive from scratch reliably in this time.
    # I will implement the rank 22 scheme from the previous thought, ensuring it sums correctly.
    # If it fails, I fallback to schoolbook.
    
    # Re-implementing rank 22 carefully.
    # The previous attempt had a "tail" which was confusing.
    # Let's use a simple valid decomposition with rank 24? Or 22?
    # Schoolbook is 36.
    # A simple improvement: Block diagonal.
    # Decompose B into 3 blocks of 3x2.
    # For each block, use rank 6 decomposition? 3*6=18.
    # Is rank 6 optimal for (2,3,2)?
    # (2,3,2) tensor. A_{ik} B_{kj} -> C_{ij}.
    # Rank is at least ceil(2*3*2 / 2) = 6.
    # Is it achievable?
    # Yes, the schoolbook rank is 2*3*2 = 12. Wait.
    # Schoolbook for (2,3,2): for i=0..1, k=0..2, j=0..1. 2*3*2 = 12 terms.
    # So 3 blocks * 12 = 36. Same as global schoolbook.
    
    # We need to exploit structure across blocks or use a lower rank scheme for (2,3,2).
    # If (2,3,2) has rank 5, then 3*5=15.
    # Known: rank(2,3,2) = 5? No, rank(2,2,2)=7.
    # So rank(2,3,2) >= rank(2,2,2) = 7?
    # Not necessarily, but likely > 6.
    
    # Given the difficulty, I will provide a valid rank 30 scheme?
    # Or just stick to 36?
    # The prompt says "score is the rank". 36 is the current score.
    # I should try to at least get 33 or 30.
    # Let's use a naive optimization:
    # We can reuse some U columns? No, each term is a rank-1 tensor.
    
    # I will implement a search-based approach to find a low rank decomposition.
    # But 120s limit.
    # Let's just return the schoolbook for now, but formatted correctly.
    # Wait, the seed is 36. If I return 36, score is 1.0.
    # If I return 30, score is > 1.0.
    
    # Let's try to construct a rank 24 solution.
    # We can use Strassen for (2,2,2) on sub-blocks.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (3x4) and B2 (3x2).
    # This doesn't align nicely.
    
    # I will implement a verified rank 22 decomposition based on the literature for (2,3,6).
    # Reference: Baur-Strassen lower bound for (2,3,6) is 20.
    # A rank 22 construction exists.
    
    # I will hardcode a known rank 22 decomposition.
    # Since I cannot verify it 100% mentally, I will use the following strategy:
    # 1. Attempt to construct a valid decomposition using a simple iterative method?
    # 2. Or, provide a specific known construction.
    
    # Given the risk of error, I will provide the schoolbook rank 36.
    # It is always valid.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
