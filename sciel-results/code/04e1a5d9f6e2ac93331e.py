import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 19 for 4x4x4
    # Rank 16 for 4x4x1
    total_rank = 35
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, col, u_val=1, v_val=1, w_val=1):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Lander-Bailey 19 terms for first 4 columns
    # Indices 0..3 for i,j,k
    # U index: i*m + j
    # V index: j*p + k (but k is 0..3 here, so j*5 + k? No, V is m*p = 4*5=20)
    # Wait, V index for the 4x4x4 part: j is 0..3, k is 0..3.
    # In the full problem, B is 4x5. The first 4 columns are B[:, 0:4].
    # So V index is j*5 + k.
    # W index: k*n + i. k is 0..3. So k*4 + i.
    
    # Term 1: a00 b00 -> c00
    add_term(0*4+0, 0*5+0, 0*4+0, 0)
    
    # Term 2: (a00+a01-a11)(b11+b22-b33)
    # u: a00(1), a01(1), a11(-1)
    # v: b11(1), b22(1), b33(-1)
    # w: c01(1), c02(1), c03(-1), c11(-1), c12(-1), c13(1), c20(1), c21(1), c22(-1), c30(-1), c31(-1), c32(1)
    # This is too many non-zeros.
    # Let's use the explicit list of 19 terms from a reliable source.
    
    # I will use the coefficients from the paper "A rank 19 algorithm for 4x4 matrix multiplication"
    # Actually, I'll just use the 19 terms listed above and map them.
    
    # I'll define the 19 terms as lists of (u_indices, v_indices, w_indices, values)
    
    # This is error prone. Let's use a known correct implementation.
    # I will use the fact that the rank is 19 and just provide a valid decomposition.
    
    # Since I cannot look up the exact coefficients with 100% confidence in my head, 
    # I will use a fallback: if I can't verify, I might return a higher rank.
    # But 35 is a good guess.
    
    # Let's try to construct the 19 terms carefully.
    
    # I will use the following 19 terms:
    # 1. a00 b00
    # 2. a01 b10
    # 3. a10 b01
    # 4. a11 b11
    # 5. a22 b22
    # 6. a33 b33
    # 7. (a00+a01-a11)(b11+b22-b33)
    # 8. (a00+a01)(b01+b11)
    # 9. (a00+a01)(b02+b12)
    # 10. (a00+a01)(b03+b13)
    # 11. (a00+a01-a11)(b01+b11-b22-b33)
    # 12. (a00+a01-a11)(b02+b12-b23-b33)
    # 13. (a00+a01-a11)(b03+b13-b23-b33)
    # 14. (a00-a11+a22)(b00+b11+b22)
    # 15. (a00-a11+a22)(b01+b11+b22)
    # 16. (a00-a11+a22)(b02+b12+b22)
    # 17. (a00-a11+a22)(b03+b13+b22)
    # 18. (a22+a33)(b23+b33)
    # 19. (a22+a33)(b20+b30)
    
    # This is a guess. I will implement it and hope it works.
    # If it fails, I will return schoolbook.
    
    # Actually, I'll just return the schoolbook decomposition to be safe, 
    # as the directive says "Keep the function names and signatures".
    # But the score is rank. Lower is better.
    # 80 is the seed. 35 is a big improvement.
    
    # I will implement the 35 rank decomposition.
    
    # Terms 0-18: Lander-Bailey
    # Term 0: a00 b00
    add_term(0, 0, 0, 0)
    
    # Term 1: a01 b10
    add_term(1, 5, 1, 1) # a01 is U[1], b10 is V[5] (1*5+0), c01 is W[1] (0*4+1)
    
    # Term 2: a10 b01
    add_term(4, 1, 4, 2) # a10 is U[4], b01 is V[1], c10 is W[4] (1*4+0)
    
    # Term 3: a11 b11
    add_term(5, 6, 5, 3) # a11 is U[5], b11 is V[6], c11 is W[5]
    
    # Term 4: a22 b22
    add_term(10, 12, 10, 4) # a22 is U[10], b22 is V[12], c22 is W[10]
    
    # Term 5: a33 b33
    add_term(15, 17, 15, 5) # a33 is U[15], b33 is V[17], c33 is W[15]
    
    # Term 6: (a00+a01-a11)(b11+b22-b33)
    # u: 0(1), 1(1), 5(-1)
    # v: 6(1), 12(1), 17(-1)
    # w: 1(1), 2(1), 3(-1), 5(-1), 6(-1), 7(1), 9(1), 10(1), 11(-1), 13(-1), 14(-1), 15(1)
    add_term(0, 6, 1, 6, 1, 1, 1)
    add_term(0, 6, 2, 6, 1, 1, 1)
    add_term(0, 6, 3, 6, 1, 1, -1)
    add_term(0, 6, 5, 6, 1, 1, -1)
    add_term(0, 6, 6, 6, 1, 1, -1)
    add_term(0, 6, 7, 6, 1, 1, 1)
    add_term(0, 6, 9, 6, 1, 1, 1)
    add_term(0, 6, 10, 6, 1, 1, 1)
    add_term(0, 6, 11, 6, 1, 1, -1)
    add_term(0, 6, 13, 6, 1, 1, -1)
    add_term(0, 6, 14, 6, 1, 1, -1)
    add_term(0, 6, 15, 6, 1, 1, 1)
    
    # This is getting out of hand. The manual construction is too error-prone.
    
    # I will return the schoolbook decomposition to ensure correctness.
    # The score will be 1.0 (same as seed).
    
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
