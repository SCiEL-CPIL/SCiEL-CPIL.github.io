import time
import sys

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # We decompose into 5 blocks of M_{2,3,2}
    # Each block handles 2 output columns.
    # Rank of M_{2,3,2} is 11.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-11 decomposition for M_{2,3,2}
    # A is 2x3, B is 3x2, C is 2x2
    # A indices: 0..5 (row-major: 00,01,02,10,11,12)
    # B indices: 0..5 (col-major? No, standard matrix index j*p+k. Here p=2. 00,10,20,01,11,21)
    # C indices: 0..3 (Trace convention W[k*n+i]. n=2. 00,10,01,11 -> indices 0, 2, 1, 3)
    # Wait, trace convention W[k*n+i].
    # k=0, i=0 -> 0
    # k=1, i=0 -> 2
    # k=0, i=1 -> 1
    # k=1, i=1 -> 3
    # So C[0] is c_{00}, C[1] is c_{10}, C[2] is c_{01}, C[3] is c_{11}.
    # Let's verify: W[k*n+i].
    # k is column of C (0 or 1). i is row of C (0 or 1).
    # k=0, i=0 -> 0*2+0 = 0. (c_{0,0})
    # k=0, i=1 -> 0*2+1 = 1. (c_{1,0})
    # k=1, i=0 -> 1*2+0 = 2. (c_{0,1})
    # k=1, i=1 -> 1*2+1 = 3. (c_{1,1})
    # Yes.
    
    # The 11 factors for one block:
    # u (length 6), v (length 6), w (length 4)
    factors = [
        # 1. a00 * b00 -> c00
        ([1,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0]),
        # 2. a01 * b10 -> c01 (c_{0,1} is index 2)
        # Wait, b10 is j=1, k=0. Index 1*2+0 = 2.
        # a01 is i=0, j=1. Index 0*3+1 = 1.
        # c01 is i=0, k=1. Index 1*2+0 = 2.
        ([0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,1,0]),
        # 3. a02 * b20 -> c02? No, c_{0,2} doesn't exist.
        # a02 is i=0, j=2. Index 2.
        # b20 is j=2, k=0. Index 2*2+0 = 4.
        # Product contributes to c_{0,0}. Index 0.
        ([0,0,1,0,0,0], [0,0,0,0,1,0], [1,0,0,0]),
        # 4. a10 * b01 -> c10 (c_{1,0} is index 1)
        # a10 is i=1, j=0. Index 3.
        # b01 is j=0, k=1. Index 0*2+1 = 1.
        # c10 is i=1, k=0. Index 0*2+1 = 1.
        ([0,0,0,1,0,0], [0,1,0,0,0,0], [0,1,0,0]),
        # 5. a11 * b11 -> c11 (c_{1,1} is index 3)
        # a11 is i=1, j=1. Index 4.
        # b11 is j=1, k=1. Index 1*2+1 = 3.
        # c11 is i=1, k=1. Index 1*2+1 = 3.
        ([0,0,0,0,1,0], [0,0,0,1,0,0], [0,0,0,1]),
        # 6. a12 * b21 -> c12? No.
        # a12 is i=1, j=2. Index 5.
        # b21 is j=2, k=1. Index 2*2+1 = 5.
        # Product contributes to c_{1,1}. Index 3.
        ([0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,1]),
        # 7. (a00 + a11) * (b10 + b01)
        # u: 1 at 0, 1 at 4.
        # v: 1 at 2 (b10), 1 at 1 (b01).
        # w: contributes to c01 (idx 2) and c10 (idx 1) and c00 (idx 0) and c11 (idx 3)?
        # (a00+a11)(b10+b01) = a00b10 + a00b01 + a11b10 + a11b01
        # a00b10 -> c00
        # a00b01 -> c01
        # a11b10 -> c10
        # a11b01 -> c11
        # So w has 1 at 0, 1 at 2, 1 at 1, 1 at 3.
        ([1,0,0,0,1,0], [0,1,1,0,0,0], [1,1,1,1]),
        # 8. (a01 + a12) * (b20 + b11)
        # u: 1 at 1, 1 at 5.
        # v: 1 at 4 (b20), 1 at 3 (b11).
        # w: a01b20->c00, a01b11->c01, a12b20->c10, a12b11->c11.
        # w: 1 at 0, 1 at 2, 1 at 1, 1 at 3.
        ([0,1,0,0,0,1], [0,0,0,1,1,0], [1,1,1,1]),
        # 9. (a02 + a10) * (b00 + b21)
        # u: 1 at 2, 1 at 3.
        # v: 1 at 0 (b00), 1 at 5 (b21).
        # w: a02b00->c00, a02b21->c01, a10b00->c10, a10b21->c11.
        # w: 1 at 0, 1 at 2, 1 at 1, 1 at 3.
        ([0,0,1,1,0,0], [1,0,0,0,0,1], [1,1,1,1]),
        # 10. (a00 + a01 + a02) * (b10 + b11 + b21)
        # u: 1 at 0,1,2.
        # v: 1 at 2 (b10), 1 at 3 (b11), 1 at 5 (b21).
        # w: 
        # a00(b10+b11+b21) -> c00, c01, c01? No.
        # a00b10 -> c00
        # a00b11 -> c01
        # a00b21 -> c01
        # a01b10 -> c00
        # a01b11 -> c01
        # a01b21 -> c01
        # a02b10 -> c00
        # a02b11 -> c01
        # a02b21 -> c01
        # So w: 1 at 0, 1 at 2 (for c01).
        # Wait, c01 is index 2.
        # So w: [1, 0, 1, 0]?
        # Let's check a00b11. a00 is i=0. b11 is j=1, k=1. Product is c_{0,1}. Index 2.
        # a00b21. a00 is i=0. b21 is j=2, k=1. Product is c_{0,1}. Index 2.
        # So all terms with k=1 contribute to index 2.
        # All terms with k=0 contribute to index 0.
        # So w: [1, 0, 1, 0].
        ([1,1,1,0,0,0], [0,0,1,1,0,1], [1,0,1,0]),
        # 11. (a10 + a11 + a12) * (b00 + b01 + b20)
        # u: 1 at 3,4,5.
        # v: 1 at 0 (b00), 1 at 1 (b01), 1 at 4 (b20).
        # w:
        # a10b00 -> c10 (idx 1)
        # a10b01 -> c11 (idx 3)
        # a10b20 -> c10 (idx 1)
        # a11b00 -> c10 (idx 1)
        # a11b01 -> c11 (idx 3)
        # a11b20 -> c10 (idx 1)
        # a12b00 -> c10 (idx 1)
        # a12b01 -> c11 (idx 3)
        # a12b20 -> c10 (idx 1)
        # So w: [0, 1, 0, 1].
        ([0,0,0,1,1,1], [1,1,0,0,1,0], [0,1,0,1]),
    ]
    
    # Apply the 5 blocks
    for block_idx in range(5):
        # Block b handles output columns 2*b and 2*b+1
        # A indices are global: i*m + j
        # B indices are global: j*p + k. Here p=10.
        # C indices are global: k*p + i. Here p=10.
        
        for k in range(11):
            col = block_idx * 11 + k
            u_local, v_local, w_local = factors[k]
            
            # Map U
            for idx_a, val in enumerate(u_local):
                if val != 0:
                    # idx_a is 0..5. i = idx_a // 3, j = idx_a % 3
                    i = idx_a // 3
                    j = idx_a % 3
                    U[i * m + j][col] += val
            
            # Map V
            for idx_b, val in enumerate(v_local):
                if val != 0:
                    # idx_b is 0..5. j = idx_b // 2, k_block = idx_b % 2
                    # Global k = 2 * block_idx + k_block
                    j = idx_b // 2
                    k_block = idx_b % 2
                    k_global = 2 * block_idx + k_block
                    V[j * p + k_global][col] += val
            
            # Map W
            for idx_c, val in enumerate(w_local):
                if val != 0:
                    # idx_c is 0..3. 
                    # Trace convention: W[k*n+i].
                    # idx_c = k_block * n + i. n=2.
                    # k_block = idx_c // 2
                    # i = idx_c % 2
                    k_block = idx_c // 2
                    i = idx_c % 2
                    k_global = 2 * block_idx + k_block
                    W[k_global * n + i][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
