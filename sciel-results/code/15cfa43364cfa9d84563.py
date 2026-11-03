import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2x2
    # u, v, w are length 2
    strassen = [
        ((1, 1), (1, 1), (1, 1)), # P1
        ((0, 1), (1, 0), (1, 0)), # P2? No, need correct vectors
    ]
    # Actually, let's just use the explicit rank-1 tensors for M_2,2,2
    # T_ij_k = sum_r u_r_i v_r_j w_r_k
    # The 7 terms are:
    # 1: u=(1,1), v=(1,1), w=(1,1) -> (A11+A22)(B11+B22)
    # 2: u=(0,1), v=(1,0), w=(1,0) -> A21 B11
    # 3: u=(1,0), v=(0,1), w=(1,-1) -> A11 (B12-B22)
    # 4: u=(0,1), v=(0,1), w=(-1,1) -> A22 (B21-B11)
    # 5: u=(1,0), v=(1,1), w=(0,1) -> A11 B22
    # 6: u=(1,-1), v=(1,0), w=(1,1) -> (A21-A11)(B11+B12)
    # 7: u=(0,1), v=(0,1), w=(1,-1) -> (A12-A22)(B21+B22)
    
    # Wait, P2 is (A21+A22)B11.
    # My vectors above:
    # 2: u=(0,1), v=(1,0), w=(1,0) -> A21 B11. Missing A22 B11.
    # Correct Strassen vectors:
    # P1: u=(1,1), v=(1,1), w=(1,1)
    # P2: u=(0,1), v=(1,1), w=(1,0) -> (A21+A22)B11
    # P3: u=(1,0), v=(0,1), w=(1,-1) -> A11(B12-B22)
    # P4: u=(0,1), v=(0,1), w=(-1,1) -> A22(B21-B11)
    # P5: u=(1,0), v=(1,1), w=(0,1) -> A11 B22 (Wait, P5 is (A11+A12)B22)
    #    u=(1,1), v=(1,1), w=(0,1) -> (A11+A12)B22
    # P6: u=(1,-1), v=(1,0), w=(1,1) -> (A21-A11)(B11+B12)
    # P7: u=(0,1), v=(0,1), w=(1,-1) -> (A12-A22)(B21+B22)
    
    s_vecs = [
        ((1, 1), (1, 1), (1, 1)),
        ((0, 1), (1, 1), (1, 0)),
        ((1, 0), (0, 1), (1, -1)),
        ((0, 1), (0, 1), (-1, 1)),
        ((1, 1), (1, 1), (0, 1)),
        ((1, -1), (1, 0), (1, 1)),
        ((0, 1), (0, 1), (1, -1))
    ]
    
    col = 0
    for r in range(7):
        u_r, v_r, w_r = s_vecs[r]
        for j_prime in range(2):
            for k_prime in range(3):
                # U index: i (0..1)
                for i in range(2):
                    U[i * 4 + 0][col] = u_r[i] # Inner index 0
                # V index: j (0..3) = 2*J + j'
                for J in range(2):
                    V[J * 2 + j_prime][col] = v_r[J]
                # W index: k (0..5) = 3*K + k'
                for K in range(2):
                    W[K * 3 + k_prime][col] = w_r[K]
                col += 1
    return {"U": U, "V": V, "W": W}
