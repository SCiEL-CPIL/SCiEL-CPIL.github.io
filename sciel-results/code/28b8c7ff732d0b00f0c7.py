import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    
    # Strassen 2x2
    U2 = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    # U2[i*2+k, r]
    # r=0: M1=(A11+A22)(B11+B22). A: A11(0,0), A22(1,1). U2[0,0]=1, U2[3,0]=1.
    # Wait, my manual array above might be wrong. Let's use the builder function from the prompt which is verified.
    
    # I will reuse the logic from the prompt but optimized.
    
    def _build_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    U2, V2, W2 = _build_strassen_2x2()

    def _build_4x4x4_strassen():
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        big_mults = [
            {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
            {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
            {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
            {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
            {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
            {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
            {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
        ]
        
        for big_idx, big in enumerate(big_mults):
            base_r = big_idx * 7
            for small_idx in range(7):
                r = base_r + small_idx
                for (bi, bk), sign_A in big['A']:
                    for i2 in range(2):
                        for k2 in range(2):
                            i = 2 * bi + i2
                            k = 2 * bk + k2
                            U[i * 4 + k, r] += sign_A * U2[i2 * 2 + k2, small_idx]
                for (bk, bj), sign_B in big['B']:
                    for k2 in range(2):
                        for j2 in range(2):
                            k = 2 * bk + k2
                            j = 2 * bj + j2
                            V[k * 4 + j, r] += sign_B * V2[k2 * 2 + j2, small_idx]
                for (bj, bi), sign_C in big['C']:
                    for j2 in range(2):
                        for i2 in range(2):
                            j = 2 * bj + j2
                            i = 2 * bi + i2
                            W[j * 4 + i, r] += sign_C * W2[j2 * 2 + i2, small_idx]
        return U, V, W

    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Build 4x4x1 schoolbook
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16))
    W41 = np.zeros((4, 16))
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U41[i * 4 + k, r] = 1
            V41[k, r] = 1
            W41[i, r] = 1

    total_rank = 49 + 49 + 16
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))

    # Part 1: 4x4x4 (cols 0-3)
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:49] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:49] = W44[j * 4 + i, :]

    # Part 2: 4x4x4 (cols 4-7)
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 49:98] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 49:98] = W44[j * 4 + i, :]

    # Part 3: 4x4x1 (col 8)
    U[:, 98:114] = U41
    for k in range(4):
        V[k * 9 + 8, 98:114] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 98:114] = W41[i, :]

    # Quick validation
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 9))
    for r in range(total_rank):
        a_vec = np.zeros(4)
        b_vec = np.zeros(9)
        for i in range(4):
            a_vec[i] = sum(U[i * 4 + k, r] * A_test[i, k] for k in range(4))
        for j in range(9):
            b_vec[j] = sum(V[k * 9 + j, r] * B_test[k, j] for k in range(4))
        for i in range(4):
            for j in range(9):
                C_computed[i, j] += W[j * 4 + i, r] * a_vec[i] * b_vec[j]
                
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback
        rank = 144
        U = np.zeros((16, rank))
        V = np.zeros((36, rank))
        W = np.zeros((36, rank))
        idx = 0
        for i in range(4):
            for j in range(9):
                for k in range(4):
                    U[i * 4 + k, idx] = 1
                    V[k * 9 + j, idx] = 1
                    W[j * 4 + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
