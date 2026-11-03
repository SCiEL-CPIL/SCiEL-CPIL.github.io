import time
import sys

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 4, 4, 7
    # Split the 7 columns of B into 4 + 3
    p1, p2 = 4, 3
    
    # 4x4x4 multiplication via 7x7x7 Strassen-like (using 2x2x2 Strassen recursively)
    # Actually, for 4x4x4, the best known is 49 (trivial) or 49 via Strassen on 2x2 blocks
    # Let's use the standard 4x4x4 tensor which has rank 49 (trivial)
    # But we can do better: 4x4x4 can be done with 49 multiplications (trivial)
    # Actually, the rank of 4x4x4 matrix multiplication is known to be 49 (no improvement over trivial for 4x4x4)
    # Wait, let me reconsider. The trivial rank is 4*4*4=64.
    # Using 2x2x2 Strassen recursively: 4x4x4 = 4*(2x2x2) -> 4*7=28? No, that's not how it works.
    # 4x4x4 can be decomposed into 16 2x2x2 multiplications -> 16*7=112. That's worse.
    # Actually, for 4x4x4, the trivial rank is 64. Let's just use trivial for 4x4x4.
    # For 4x4x3, trivial rank is 4*4*3=48.
    # Total: 64+48=112. Same as trivial.
    
    # Let me think differently. Use the full 4x4x7 trivial: 4*4*7=112.
    # Can we do better? 
    # 4x4x7: We can split as 4x4x4 + 4x4x3 = 64+48=112.
    # Or use Strassen-like on the 4x4 part: 4x4x4 has rank 49? No, it's 64.
    # Actually, the rank of matrix multiplication [4,4,4] is exactly 49? No, it's 64 (no known improvement).
    # Wait, I recall that for 2x2x2 it's 7 (Strassen). For 4x4x4, using Strassen recursively gives 4*7=28? No.
    # The 4x4x4 tensor can be decomposed by viewing it as 4 2x2x4 blocks? 
    # Let me just use the trivial decomposition for now: rank 112.
    
    # Actually, let me try a different split: 4x4x7 = 4x4x2 + 4x4x2 + 4x4x2 + 4x4x1
    # 4x4x2 trivial: 32 each, 4x4x1: 16. Total: 32*3+16=112. Same.
    
    # The trivial rank is 112. Let me just return that.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
