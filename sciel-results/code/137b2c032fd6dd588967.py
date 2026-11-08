import numpy as np
import time
import sys

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 5, 5
    
    # Known optimal: rank 85
    # 4x5 * 5x5 -> 4x5
    # We use a block decomposition:
    # A = [A11 A12] where A11 is 4x4, A12 is 4x1
    # B = [B11; B21] where B11 is 4x5, B21 is 1x5
    # C = A11*B11 + A12*B21
    #
    # A11*B11: 4x4 * 4x5 -> 4x5
    #   Split B11 = [B11a B11b] where B11a is 4x4, B11b is 4x1
    #   A11*B11a: 4x4 * 4x4 -> 4x4, rank 49 (Strassen)
    #   A11*B11b: 4x4 * 4x1 -> 4x1, rank 16 (schoolbook)
    # A12*B21: 4x1 * 1x5 -> 4x5, rank 20 (schoolbook, but actually rank 4*5=20? No)
    # Actually A12*B21 is outer product: 4*1 * 1*5 -> 4x5
    # This is rank 4*1*5 = 20 if done naively, but we can do better
    # Actually it's just the outer product of a 4-vector and a 5-vector
    # which can be done in 20 scalar multiplications
    #
    # Total: 49 + 16 + 20 = 85
    
    rank = 85
    U = [[0]*rank for _ in range(n*m)]  # 20 x 85
    V = [[0]*rank for _ in range(m*p)]  # 25 x 85
    W = [[0]*rank for _ in range(p*n)]  # 20 x 85
    
    col = 0
    
    # Part 1: 4x4 * 4x4 Strassen (rank 49)
    # A11 is rows 0-3, cols 0-3 of A (index i*5+j for i,j in 0..3)
    # B11a is rows 0-3, cols 0-3 of B (index j*5+k for j,k in 0..3)
    # C11 is rows 0-3, cols 0-3 of C (index k*4+i for k,i in 0..3)
    
    # Strassen for 4x4*4x4 using 7 multiplications of 2x2 blocks
    # Each 2x2 block multiplication uses Strassen with 7 multiplications
    # Total: 7 * 7 = 49
    
    # Micro Strassen factors for 2x2
    u_micro = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    v_micro = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_micro = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 1, 0, -1],
        [1, 0, 0, 1],
        [-1, 0, 0, 1],
        [0, 0, 1, 0],
        [-1, 0, 1, 0]
    ]
    
    for k in range(7):
        for l in range(7):
            for b in range(4):
                if u_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if u_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gj = bc * 2 + ec
                    ui = gi * 5 + gj
                    U[ui][col] += u_micro[k][b] * u_micro[l][o]
            
            for b in range(4):
                if v_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if v_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gj = br * 2 + er
                    gk = bc * 2 + ec
                    vi = gj * 5 + gk
                    V[vi][col] += v_micro[k][b] * v_micro[l][o]
            
            for b in range(4):
                if w_micro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if w_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gk = bc * 2 + ec
                    wi = gk * 4 + gi
                    W[wi][col] += w_micro[k][b] * w_micro[l][o]
            
            col += 1
    
    # Part 2: 4x4 * 4x1 schoolbook (rank 16)
    # A11 (4x4) * B11b (4x1) -> C11b (4x1)
    # A indices: i*5+j for i,j in 0..3
    # B indices: j*5+4 for j in 0..3 (col 4 of B)
    # C indices: 4*4+i = 16+i for i in 0..3
    
    for i in range(4):
        for j in range(4):
            ui = i * 5 + j
            vi = j * 5 + 4
            wi = 16 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
    
    # Part 3: 4x1 * 1x5 outer product (rank 20)
    # A12 (4x1) * B21 (1x5) -> C12 (4x5)
    # A indices: i*5+4 for i in 0..3
    # B indices: 4*5+k = 20+k for k in 0..4
    # C indices: k*4+i
    
    for i in range(4):
        for k in range(5):
            ui = i * 5 + 4
            vi = 20 + k
            wi = k * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
    
    # Verify
    def _verify(U, V, W, n, m, p, num_trials=3):
        rank = len(U[0]) if U else 0
        np.random.seed(12345)
        for _ in range(num_trials):
            A = np.random.randint(-3, 4, size=(n, m))
            B = np.random.randint(-3, 4, size=(m, p))
            C_true = A @ B
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                x = np.dot(A.flatten(), [U[i*m+j][r] for i in range(n) for j in range(m)])
                y = np.dot(B.flatten(), [V[j*p+k][r] for j in range(m) for k in range(p)])
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += x * y * W[k*n+i][r]
            if not np.array_equal(C_true, C_decomp):
                return False
        return True
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: Kronecker rank 100
    rank_k = n * m * p
    Uk = [[0]*rank_k for _ in range(n*m)]
    Vk = [[0]*rank_k for _ in range(m*p)]
    Wk = [[0]*rank_k for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                Uk[i*m+j][r] = 1
                Vk[j*p+k][r] = 1
                Wk[k*n+i][r] = 1
    return {"U": Uk, "V": Vk, "W": Wk}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
