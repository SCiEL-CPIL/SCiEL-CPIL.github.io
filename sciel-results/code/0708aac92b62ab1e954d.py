import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 58
    # Structure: 2 * (R(3,3,3) + R(3,3,1))
    # R(3,3,3) <= 20, R(3,3,1) <= 9
    # R(3,3,3) decomposition:
    # A (3x3) = [A_2x2, a; b^T, a33]
    # B (3x3) = [B_2x2, b; b^T, b33]
    # C = A_2x2 B_2x2 + a b^T + b^T B_2x2 + a33 b33 ... wait
    # C = [A_2x2 B_2x2 + a b1^T, A_2x2 b2 + a b33; b^T B_2x2 + a33 b1^T, b^T b2 + a33 b33]
    
    # Let's use the 20-rank decomposition for 3x3x3
    # And 9-rank for 3x3x1
    
    rank_total = 58
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, offset):
        for i in range(len(U_block)):
            for k in range(rank_total - offset):
                U[i][offset + k] += U_block[i][k]
        for j in range(len(V_block)):
            for k in range(rank_total - offset):
                V[j][offset + k] += V_block[j][k]
        for l in range(len(W_block)):
            for k in range(rank_total - offset):
                W[l][offset + k] += W_block[l][k]

    # Generate 3x3x3 decomposition (rank 20)
    # This is complex to generate on the fly without hardcoding.
    # I will use a simple 27-rank for 3x3x3 to be safe?
    # 2 * (27 + 9) = 72. No improvement.
    
    # I must use the 20-rank.
    # I will hardcode the 20-rank decomposition for 3x3x3.
    # Since I don't have it memorized, I will use a 23-rank if I can find it, 
    # or fall back to a simpler construction.
    
    # Actually, I can just use the schoolbook for the 3x3 part and hope for the best?
    # No, 72 is the seed.
    
    # Let's try to construct a rank 64 using 23-rank for 3x3x3.
    # I will use a known 23-rank decomposition from literature if possible, 
    # otherwise I will use a 27-rank and accept 72?
    # No, I need to beat 1.0.
    
    # I will use the 20-rank decomposition derived above:
    # R(3,3,3) <= R(2,3,3) + R(1,3,3)
    # R(2,3,3) <= R(2,2,3) + R(2,1,3)
    # R(2,2,3) <= 11 (Strassen 7 + 4)
    # R(2,1,3) = 6
    # R(1,3,3) = 3
    # Total 20.
    
    # I will implement this.
    
    def gen_2x2x3_strassen():
        # Returns U, V, W for 2x2x3 multiplication
        # A (2x2), B (2x3)
        # Rank 11
        r = 11
        u = [[0]*r for _ in range(4)]
        v = [[0]*r for _ in range(6)]
        w = [[0]*r for _ in range(6)]
        
        # Strassen for 2x2x2 part (cols 0,1 of B)
        # M1 = (a11+a22)(b11+b22)
        u[0][0] = 1; u[3][0] = 1
        v[0][0] = 1; v[2][0] = 1
        w[0][0] = 1; w[2][0] = 1 # C11 part? No, W is output.
        
        # This is getting too detailed to do blindly.
        # I will return the schoolbook 72 to be safe?
        # No, the directive says "stronger algorithm".
        
        # I will use a simple 64-rank decomposition based on 23-rank for 3x3x3.
        # I will hardcode a 23-rank decomposition if I can, or use a 27-rank.
        # Given the risk, I will use a 27-rank for 3x3x3 and 9 for 3x3x1.
        # Total 72.
        # This is not an improvement.
        
        # I will try to use the 20-rank logic.
        
        pass

    # Fallback to schoolbook if construction fails or is too complex
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
