"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    n, m, p = 4, 4, 9
    
    # Schoolbook rank: 144
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a better decomposition using random initialization and iterative rank reduction
    # This is a heuristic search for lower rank decompositions
    
    # Strategy: Start with schoolbook, then try to merge terms that have similar structure
    # Or use a random search over low-rank decompositions
    
    # Let's try a structured approach:
    # 1. Start with a random low-rank decomposition
    # 2. Check if it's valid
    # 3. If not, try to adjust
    
    # Actually, let's use a more systematic approach:
    # Try rank values from known lower bounds up to schoolbook
    # For 4x4x9, the minimum rank is at least max(16, 36, 16) = 36 (by dimensional arguments)
    # But this is a weak lower bound. The actual minimum is likely higher.
    
    # Let's try to use the fact that 9 = 4 + 4 + 1
    # Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # Each A*B_i is a 4x4 multiplication (for B1, B2) or 4x1 (for B3)
    
    # For 4x4*4x4, we can use Strassen with rank 49
    # For 4x4*4x1, we can use rank 16 (schoolbook)
    # Total: 49 + 49 + 16 = 114
    
    # But we can do better by sharing some computations.
    # Let's implement the Strassen decomposition for 4x4*4x4
    
    def strassen_4x4():
        """Returns U, V, W for 4x4 * 4x4 with rank 49."""
        # Standard Strassen for 2x2 has rank 7
        # For 4x4, we can apply Strassen twice (block decomposition)
        # 4x4 = 2x(2x2), so we get rank 7^2 = 49
        
        # The Strassen algorithm for 2x2:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4, we treat it as 2x(2x2) blocks
        # This gives us a decomposition of rank 49
        
        U = []
        V = []
        W = []
        
        # We need to carefully construct the factor matrices
        # Let's use a known decomposition
        
        # Actually, let's just use the schoolbook for 4x4*4x4 (rank 16) 
        # and see if we can improve
        rank = 16
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    idx = i * 4 + j * 16 + k
                    # Wait, the indexing is different
                    # U has shape (4*4, rank) = (16, rank)
                    # V has shape (4*4, rank) = (16, rank)
                    # W has shape (4*4, rank) = (16, rank)
                    # U[i*4+j, c], V[j*4+k, c], W[k*4+i, c]
                    c = i * 4 + j * 4 + k
                    U[i * 4 + j][c] = 1
                    V[j * 4 + k][c] = 1
                    W[k * 4 + i][c] = 1
        
        return U, V, W
    
    # Let's just return the schoolbook decomposition for now
    # and add a time limit guard
    start_time = time.time()
    
    # Try to find a better decomposition
    # For now, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
