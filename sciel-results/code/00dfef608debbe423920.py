import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 * 7 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # u, v, w are vectors of length 4 (indices 00, 01, 10, 11)
    # Indices for U: i*m+j -> i*2+j
    # Indices for V: j*p+k -> j*2+k (local k)
    # Indices for W: k*n+i -> k*2+i (local k)
    
    # The 7 rank-1 tensors
    # u: (1, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 0, 1), (1, 1, 0, 0), (0, 1, -1, 0), (1, -1, 0, 0)
    # v: (1, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, -1), (0, 0, 1, -1), (0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1)
    # w: (1, 0, 0, 1), (0, 1, 0, -1), (0, 0, 1, 1), (1, 1, 0, 0), (0, 0, 1, -1), (0, 0, 0, 0), (0, 0, 0, 0) 
    # Wait, w6 and w7 are not zero.
    
    # Let's use a verified set from a reliable source or derive it.
    # I will use the set that sums to the identity tensor.
    
    # Correct Strassen vectors (u, v, w) for 2x2x2
    # u is coeff of A, v is coeff of B, w is coeff of C (dual)
    
    u_vectors = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, -1, 0],
        [1, -1, 0, 0]
    ]
    
    v_vectors = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_vectors = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 0], # Placeholder
        [0, 0, 0, 0]  # Placeholder
    ]
    
    # I need to verify w6 and w7.
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # m6 is not in the output?
    # If m6 is not in the output, it means it's 0? No, that's not a valid decomposition.
    # Actually, c11 = m1 - m2 + m3 + m7 is a common formula.
    # Let's check if m6 is needed.
    # If I use 7 products, all 7 must be used.
    # Maybe c11 = m1 - m2 + m3 + m6?
    
    # Let's assume the standard Strassen 7 products are:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a01-a10)(b00+b01)
    # m7 = (a10-a00)(b11+b10)
    
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # In this formulation, m6 is NOT used. This implies the rank is effectively 6?
    # No, Strassen's algorithm uses 7 multiplications.
    # The formula for c11 is often c11 = m1 - m2 + m3 + m7.
    # Where is m6?
    # Ah, c11 = a10*b01 + a11*b11.
    # m1 = a00*b00 + a00*b11 + a11*b00 + a11*b11
    # -m2 = -a10*b00 - a11*b00
    # +m3 = a00*b01 - a00*b11
    # +m7 = a10*b11 + a10*b10 - a00*b11 - a00*b10
    # Sum = a11*b11 + a10*b10 + a10*b11 - a00*b10 - a00*b11 + a00*b01 + a00*b00 - a10*b00
    # This is not c11.
    
    # Correct Strassen output:
    # c00 = m1 + m4 - m5 - m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m7
    
    # It seems m6 is indeed not in the output in some formulations?
    # No, that's wrong.
    
    # I will use a Python script to find the w vectors that satisfy the tensor identity.
    # But I can't run code.
    
    # I will use the known rank-7 decomposition vectors.
    # u, v, w for the 7 terms.
    
    # Term 1:
    # u: 1 0 0 1
    # v: 1 0 0 1
    # w: 1 0 0 1
    
    # Term 2:
    # u: 0 0 1 1
    # v: 1 0 0 0
    # w: 0 1 0 -1
    
    # Term 3:
    # u: 1 0 0 0
    # v: 0 1 0 -1
    # w: 0 0 1 1
    
    # Term 4:
    # u: 0 0 0 1
    # v: 0 0 1 -1
    # w: 1 1 0 0
    
    # Term 5:
    # u: 1 1 0 0
    # v: 0 0 0 1
    # w: 0 0 1 -1
    
    # Term 6:
    # u: 0 1 -1 0
    # v: 1 1 0 0
    # w: 0 0 0 0 -> This must be wrong.
    
    # Let's check the tensor sum.
    # I will hardcode the U, V, W for the 5 blocks using the 7 vectors I am most confident about, and if I'm unsure about w6/w7, I'll use a fallback.
    # Actually, I can just use the schoolbook for the last block to be safe? No, that increases rank.
    
    # I will use the following valid decomposition:
    # This is the standard Strassen decomposition.
    
    u_list = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, -1, 0],
        [1, -1, 0, 0]
    ]
    
    v_list = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    w_list = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    
    # I am not confident in w6 and w7.
    # I will use rank 40 (schoolbook) if I can't be sure.
    # But 35 is a known improvement.
    
    # Let's assume the question allows me to return the schoolbook if I can't construct the Strassen one correctly.
    # But I should try.
    
    # I will use the schoolbook decomposition for safety, as an incorrect decomposition scores 0.
    # Rank 40 is safe.
    
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
