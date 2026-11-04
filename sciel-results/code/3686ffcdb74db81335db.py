import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    
    # 1. Strassen for 2x2x2
    # U1: (4, 7), V1: (4, 7), W1: (4, 7)
    # Indices: U[i,j], V[j,k], W[k,i]
    U1 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    V1 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1]
    ]
    W1 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    R1 = 7
    
    # 2. Schoolbook for 2x2x3
    # A: 2x2, B: 2x3, C: 2x3
    # U2: (4, 12), V2: (6, 12), W2: (6, 12)
    # U2[i*2+j, col], V2[j*3+k, col], W2[k*2+i, col]
    R2 = 2 * 2 * 3
    U2 = [[0] * R2 for _ in range(4)]
    V2 = [[0] * R2 for _ in range(6)]
    W2 = [[0] * R2 for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U2[i * 2 + j][col] = 1
                V2[j * 3 + k][col] = 1
                W2[k * 2 + i][col] = 1
                col += 1
                
    R = R1 * R2
    
    # Total dimensions
    # A: 4x4 -> (rA1, rA2, cA1, cA2)
    # B: 4x9 -> (rB1, rB2, cB1, cB2)
    # C: 4x9 -> (rC1, rC2, cC1, cC2)
    
    U = [[0] * R for _ in range(16)]
    V = [[0] * R for _ in range(36)]
    W = [[0] * R for _ in range(36)]
    
    # Fill U
    # Index in U: (rA, cA) = (2*rA1 + rA2, 2*cA1 + cA2)
    # Col in U: k1 * R2 + k2
    for rA1 in range(2):
        for rA2 in range(2):
            for cA1 in range(2):
                for cA2 in range(2):
                    i_U = rA1 * 2 + rA2
                    j_U = cA1 * 2 + cA2
                    u_idx = i_U * 4 + j_U
                    
                    # U1 index: (rA1, cA1) -> rA1*2 + cA1
                    u1_idx = rA1 * 2 + cA1
                    # U2 index: (rA2, cA2) -> rA2*2 + cA2
                    u2_idx = rA2 * 2 + cA2
                    
                    for k1 in range(R1):
                        val1 = U1[u1_idx][k1]
                        if val1 == 0: continue
                        for k2 in range(R2):
                            val2 = U2[u2_idx][k2]
                            if val2 == 0: continue
                            U[u_idx][k1 * R2 + k2] = val1 * val2

    # Fill V
    # Index in V: (rB, cB) = (2*rB1 + rB2, 3*cB1 + cB2)
    # Note: V in tensor is indexed by (j, k) of the multiplication.
    # Here j is col of A / row of B.
    # Block level: j1 = cA1, k1 = rB1.
    # Inner level: j2 = cA2, k2 = rB2.
    # Wait, V1 index is (cA1, rB1).
    # V2 index is (cA2, cB2)? No.
    # In inner multiplication 2x2 * 2x3:
    # A is 2x2. B is 2x3.
    # V2 index is (j, k) where j is row of B (0..1), k is col of B (0..2).
    # So V2 index is (rB2, cB2).
    # And it must match with U2's j which is cA2.
    # So V2[rB2 * 3 + cB2, col].
    
    for rB1 in range(2):
        for rB2 in range(2):
            for cB1 in range(3):
                for cB2 in range(3):
                    i_V = rB1 * 2 + rB2
                    j_V = cB1 * 3 + cB2
                    v_idx = i_V * 9 + j_V
                    
                    # V1 index: (cA1, rB1). But cA1 is not in V's index!
                    # V is defined on B's indices.
                    # The tensor link is that cA1 (col of A block) == rB1 (row of B block).
                    # In the decomposition, U1 has cA1, V1 has rB1.
                    # They are linked by the rank index k1.
                    # So V1 is indexed by (rB1, cB1)?
                    # No. In matrix mult C = AB, V is indexed by (j, k) where j is col of A, k is col of B.
                    # So V1 should be indexed by (j_block, k_block) = (cA1, cB1).
                    # But V's physical index is (rB, cB).
                    # rB = 2*rB1 + rB2.
                    # cB = 3*cB1 + cB2.
                    # The "j" in V1 is the block column of A, which is the block row of B.
                    # So j1 = rB1.
                    # The "k" in V1 is the block column of B.
                    # So k1 = cB1.
                    # Thus V1 index is (rB1, cB1).
                    
                    v1_idx = rB1 * 2 + cB1
                    v2_idx = rB2 * 3 + cB2
                    
                    for k1 in range(R1):
                        val1 = V1[v1_idx][k1]
                        if val1 == 0: continue
                        for k2 in range(R2):
                            val2 = V2[v2_idx][k2]
                            if val2 == 0: continue
                            V[v_idx][k1 * R2 + k2] = val1 * val2

    # Fill W
    # Index in W: (k, i) = (cC, rC) = (3*cC1 + cC2, 2*rC1 + rC2)
    # W1 index: (k_block, i_block) = (cC1, rC1).
    # W2 index: (k_inner, i_inner) = (cC2, rC2).
    
    for rC1 in range(2):
        for rC2 in range(2):
            for cC1 in range(3):
                for cC2 in range(3):
                    k_W = cC1 * 3 + cC2
                    i_W = rC1 * 2 + rC2
                    w_idx = k_W * 4 + i_W
                    
                    w1_idx = cC1 * 2 + rC1
                    w2_idx = cC2 * 2 + rC2
                    
                    for k1 in range(R1):
                        val1 = W1[w1_idx][k1]
                        if val1 == 0: continue
                        for k2 in range(R2):
                            val2 = W2[w2_idx][k2]
                            if val2 == 0: continue
                            W[w_idx][k1 * R2 + k2] = val1 * val2

    return {"U": U, "V": V, "W": W}
