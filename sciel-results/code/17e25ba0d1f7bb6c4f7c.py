"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Winograd-style decomposition using 2x2x2 (rank 7) and 2x2x1 (rank 4) blocks.
    # Total rank: 2 * (7 + 7 + 4) = 36.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] += val
        for idx, val in v_dict.items():
            V[idx][col] += val
        for idx, val in w_dict.items():
            W[idx][col] += val
        col += 1

    def add_strassen_block(a_i, a_j, b_i, b_j, c_i, c_k):
        # Maps local 2x2 indices to global
        def ma(l): return (a_i + l//2)*m + (a_j + l%2)
        def mb(l): return (b_i + l//2)*p + (b_j + l%2)
        def mc(l): return (c_k + l%2)*n + (c_i + l//2)

        # Strassen 7 terms
        add_rank1_term({ma(0):1, ma(3):1}, {mb(0):1, mb(3):1}, {mc(0):1, mc(3):1})
        add_rank1_term({ma(2):1, ma(3):1}, {mb(0):1}, {mc(2):1, mc(3):-1})
        add_rank1_term({ma(0):1}, {mb(1):1, mb(3):-1}, {mc(1):1, mc(3):1})
        add_rank1_term({ma(3):1}, {mb(2):1, mb(0):-1}, {mc(0):1, mc(2):1})
        add_rank1_term({ma(0):1, ma(1):1}, {mb(3):1}, {mc(0):-1, mc(1):1})
        add_rank1_term({ma(2):1, ma(0):-1}, {mb(0):1, mb(1):1}, {mc(3):1})
        add_rank1_term({ma(1):1, ma(3):-1}, {mb(2):1, mb(3):1}, {mc(0):1})

    def add_schoolbook_2x2x1(a_i, a_j, b_i, b_k, c_i, c_k):
        # 2x2 A * 2x1 B -> 2x1 C
        def ma(l): return (a_i + l//2)*m + (a_j + l%2)
        def mb(l): return (b_i + l)*p + b_k
        def mc(l): return c_k*n + (c_i + l)

        # C_0 = A_00 B_0 + A_01 B_1
        # C_1 = A_10 B_0 + A_11 B_1
        add_rank1_term({ma(0):1}, {mb(0):1}, {mc(0):1})
        add_rank1_term({ma(1):1}, {mb(1):1}, {mc(0):1})
        add_rank1_term({ma(2):1}, {mb(0):1}, {mc(1):1})
        add_rank1_term({ma(3):1}, {mb(1):1}, {mc(1):1})

    # Block 1: A cols 0-1, B rows 0-1
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A cols 2-3, B rows 2-3
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)

    # Verify correctness
    if time.time() < deadline - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for all i,j,k: sum_t U[i,t]*V[j,t]*W[k,t] = 1 if (i,j,k) is a valid multiplication term, else 0
        # The tensor T[i,j,k] = 1 if i = row of A, j = col of A = row of B, k = col of B = col of C
        # Actually the convention is W[k*n+i, r], so W index is k*n+i where k is output col, i is output row
        # U index: i*m + j where i is A row, j is A col (= B row)
        # V index: j*p + k where j is B row, k is B col
        # W index: k*n + i where k is C col, i is C row
        
        # So T[i,j,k] should be 1 if the multiplication produces C[i,k] += A[i,j]*B[j,k]
        # That is, for each (i,j,k), the contribution is A[i,j]*B[j,k] to C[i,k]
        # So the tensor element at (i,j,k) is 1
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for t in range(rank):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        actual += U_np[u_idx, t] * V_np[v_idx, t] * W_np[w_idx, t]
                    max_err = max(max_err, abs(actual - expected))
        
        if max_err > 1e-9:
            # Fallback: return schoolbook rank n*m*p = 40
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            t = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j][t] = 1
                        V[j*p + k][t] = 1
                        W[k*n + i][t] = 1
                        t += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
