import time

def construct():
    deadline = time.time() + 120
    
    # Dimensions
    n, m, p = 6, 6, 6
    
    # Strassen 2x2 decomposition
    # U2, V2, W2 of shape (4, 7)
    # Local indices: 0:00, 1:01, 2:10, 3:11
    U2 = [
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, -1],
        [0, 0, 1, 1, -1, 0, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    V2 = [
        [1, 0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, -1],
        [0, 0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, -1, -1]
    ]
    W2 = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, -1, 0, 0, 1]
    ]
    
    # Verify Strassen 2x2
    def verify_strassen():
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    s = 0
                    for s_idx in range(7):
                        s += U2[i*2+k][s_idx] * V2[k*2+j][s_idx] * W2[j*2+i][s_idx]
                    if s != 1:
                        return False
        return True
    
    if not verify_strassen():
        return _build_schoolbook()
    
    # Partition 6x6 into 3x3 grid of 2x2 blocks
    # i = 2*i3 + i2, k = 2*k3 + k2, j = 2*j3 + j2
    # i3, j3, k3 in {0,1,2}, i2, j2, k2 in {0,1}
    # For each (i3, k3, j3), the 2x2 block product uses Strassen (rank 7)
    # Total rank = 3*3*3*7 = 189? No wait.
    # Actually for each fixed (i3, j3), we sum over k3=0,1,2
    # Each term A[i3,k3]*B[k3,j3] is a 2x2 matrix product with Strassen rank 7
    # So for each (i3, j3, k3) we have 7 terms
    # Total rank = 3*3*3*7 = 189
    
    # Wait, that's the same as before. Let me reconsider.
    # Actually the correct block decomposition:
    # C[i3, j3] (2x2 block) = sum_{k3=0}^{2} A[i3, k3] (2x2) * B[k3, j3] (2x2)
    # Each 2x2*2x2 product has Strassen rank 7
    # So total rank = 3 (i3) * 3 (j3) * 3 (k3) * 7 = 189
    
    # Alternative: 2x2 grid of 3x3 blocks
    # i = 2*i3 + i2 where i3 in {0,1}, i2 in {0,1,2,3,4,5}... no that doesn't work.
    # i = 2*i3 + i2, i3 in {0,1}, i2 in {0,1} gives only 4 values.
    # For 3x3 blocks: i = 3*i3 + i2, i3 in {0,1}, i2 in {0,1,2}
    # Then C[i3,j3] (3x3) = sum_{k3=0}^{1} A[i3,k3] (3x3) * B[k3,j3] (3x3)
    # Each 3x3*3x3 product has schoolbook rank 27
    # Total rank = 2*2*2*27 = 216. Worse.
    
    # Let me try: use Strassen recursively.
    # 6x6 = 2x2 blocks of 3x3. Use Strassen on the 2x2 block structure.
    # The 2x2 block multiplication has rank 7 (Strassen).
    # Each "scalar" in the 2x2 block algebra is a 3x3 matrix, multiplied with schoolbook rank 27.
    # Total rank = 7 * 27 = 189. Same again.
    
    # The fundamental issue: any block decomposition gives product of ranks.
    # To beat 189, we need a non-block decomposition or a better 3x3 decomposition.
    
    # Best known: rank(3x3) <= 19 (Landsberg). But explicit construction is complex.
    # Let's use the known rank-21 decomposition for 3x3 (Bini et al.) if we can find it.
    # Actually, let me just use the schoolbook 3x3 (rank 27) with Strassen 2x2 outer:
    # 6x6 as 2x2 blocks of 3x3: rank = 7 * 27 = 189
    
    # Or 6x6 as 3x3 blocks of 2x2: rank = 27 * 7 = 189
    
    # Both give 189. The current code claims rank 189 but may have bugs.
    # Let me just implement the 3x3 blocks of 2x2 approach cleanly.
    
    # 6x6 partitioned as 3x3 grid of 2x2 blocks:
    # i = 2*i3 + i2, j = 2*j3 + j2, k = 2*k3 + k2
    # i3,j3,k3 in {0,1,2}, i2,j2,k2 in {0,1}
    # C[i,j] = sum_k A[i,k] B[k,j]
    # For each (i3,j3,k3), the 2x2 block: C[i3,j3] = A[i3,k3] * B[k3,j3] (2x2 mat mult)
    # Using Strassen: sum_s U2[i2,k2,s] * A[2*i3+i2, 2*k3+k2] * V2[k2,j2,s] * B[2*k3+k2, 2*j3+j2] * W2[j2,i2,s]
    
    rank = 27 * 7  # 3*3*3 * 7 = 189
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i3 in range(3):
        for k3 in range(3):
            for j3 in range(3):
                for s in range(7):
                    # U factor: for all i2, k2
                    for i2 in range(2):
                        for k2 in range(2):
                            U[(2*i3+i2)*6 + (2*k3+k2)][col] = U2[i2*2+k2][s]
                    # V factor: for all k2, j2
                    for k2 in range(2):
                        for j2 in range(2):
                            V[(2*k3+k2)*6 + (2*j3+j2)][col] = V2[k2*2+j2][s]
                    # W factor: for all j2, i2
                    for j2 in range(2):
                        for i2 in range(2):
                            W[(2*j3+j2)*6 + (2*i3+i2)][col] = W2[j2*2+i2][s]
                    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    # Verify
    def verify():
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    total = 0
                    for s in range(rank):
                        total += U[i*m+k][s] * V[k*p+j][s] * W[j*n+i][s]
                    if total != 1:
                        return False
        return True
    
    if not verify():
        return _build_schoolbook()
    
    return {"U": U, "V": V, "W": W}

def _build_schoolbook():
    n, m, p = 6, 6, 6
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
