"""Find a rank-minimal exact tensor decomposition for 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    dim_u = n * m  # 10
    dim_v = m * p  # 25
    dim_w = p * n  # 10
    
    # Build the target tensor
    # T[a, b, c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                target[u_idx, v_idx, w_idx] = 1.0
    
    def get_schoolbook():
        """Return schoolbook decomposition as lists."""
        rank = n * m * p  # 50
        U = [[0] * rank for _ in range(dim_u)]
        V = [[0] * rank for _ in range(dim_v)]
        W = [[0] * rank for _ in range(dim_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def get_block_decomposition():
        """
        Returns a rank 49 decomposition using block structure:
        Split A (2x5) into A_L (2x2) and A_R (2x3).
        Split B (5x5) into B_TL (2x2), B_TR (2x3), B_BL (3x2), B_BR (3x3).
        
        Ranks:
        Block 1: A_L (2x2) * B_TL (2x2) -> C_TL (2x2): rank 7 (Strassen)
        Block 2: A_L (2x2) * B_TR (2x3) -> C_TR (2x3): rank 6 (schoolbook)
        Block 3: A_R (2x3) * B_BL (3x2) -> C_BL (2x2): rank 6 (schoolbook)
        Block 4: A_R (2x3) * B_BR (3x3) -> C_BR (2x3): rank 9 (schoolbook)
        Total = 7 + 6 + 6 + 9 = 28
        
        Wait, that's not right. Let me recalculate.
        For schoolbook n*m*p:
        Block 1: 2*2*2 = 8
        Block 2: 2*2*3 = 12
        Block 3: 2*3*2 = 12
        Block 4: 2*3*3 = 18
        Total with schoolbook = 50
        
        With Strassen for block 1: 7 + 12 + 12 + 18 = 49
        """
        total_rank = 49
        U = [[0] * total_rank for _ in range(dim_u)]
        V = [[0] * total_rank for _ in range(dim_v)]
        W = [[0] * total_rank for _ in range(dim_w)]
        
        col = 0
        
        # Helper to add a Strassen term (multiple non-zero entries in U, V, W)
        def add_strassen_term(u_dict, v_dict, w_dict):
            nonlocal col
            for idx, val in u_dict.items():
                U[idx][col] += val
            for idx, val in v_dict.items():
                V[idx][col] += val
            for idx, val in w_dict.items():
                W[idx][col] += val
            col += 1

        # Helper to add a schoolbook term (single non-zero entry in each)
        def add_schoolbook_term(u_idx, v_idx, w_idx):
            nonlocal col
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

        # Index mappings:
        # U (2x5): i in 0..1, j in 0..4. idx = i*5+j
        #   A_L: i=0,1; j=0,1 -> indices 0,1,5,6
        #   A_R: i=0,1; j=2,3,4 -> indices 2,3,4,7,8,9
        # V (5x5): j in 0..4, k in 0..4. idx = j*5+k
        #   B_TL: j=0,1; k=0,1 -> indices 0,1,5,6
        #   B_TR: j=0,1; k=2,3,4 -> indices 2,3,4,7,8,9
        #   B_BL: j=2,3,4; k=0,1 -> indices 10,11,12,15,16,17,20,21,22
        #   B_BR: j=2,3,4; k=2,3,4 -> indices 12,13,14,17,18,19,22,23,24
        # W (5x2): k in 0..4, i in 0..1. idx = k*2+i
        #   C_TL: k=0,1; i=0,1 -> indices 0,1,2,3
        #   C_TR: k=2,3,4; i=0,1 -> indices 4,5,6,7,8,9
        #   C_BL: k=0,1; i=0,1 -> indices 0,1,2,3
        #   C_BR: k=2,3,4; i=0,1 -> indices 4,5,6,7,8,9

        # Block 1: A_L (2x2) * B_TL (2x2) -> C_TL (2x2) using Strassen
        # Local indices:
        # A_L[0,0]=U[0], A_L[0,1]=U[1], A_L[1,0]=U[5], A_L[1,1]=U[6]
        # B_TL[0,0]=V[0], B_TL[0,1]=V[1], B_TL[1,0]=V[5], B_TL[1,1]=V[6]
        # C_TL[0,0]=W[0], C_TL[0,1]=W[1], C_TL[1,0]=W[2], C_TL[1,1]=W[3]
        
        # Strassen for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        add_strassen_term({0:1, 6:1}, {0:1, 6:1}, {0:1, 3:1})
        # M2 = (A10+A11)B00
        add_strassen_term({5:1, 6:1}, {0:1}, {2:1})
        # M3 = A00(B10-B11)
        add_strassen_term({0:1}, {5:1, 6:-1}, {1:1})
        # M4 = A11(B01-B00)
        add_strassen_term({6:1}, {1:1, 0:-1}, {3:1})
        # M5 = (A00+A01)B11
        add_strassen_term({0:1, 1:1}, {6:1}, {1:-1})
        # M6 = (A10-A00)(B00+B10)
        add_strassen_term({5:1, 0:-1}, {0:1, 5:1}, {3:1})
        # M7 = (A01-A11)(B01+B11)
        add_strassen_term({1:1, 6:-1}, {1:1, 6:1}, {0:1})
        # col is now 7

        # Block 2: A_L (2x2) * B_TR (2x3) -> C_TR (2x3) using schoolbook
        # A_L: i=0,1; j=0,1. U indices: 0,1,5,6
        # B_TR: j=0,1; k=2,3,4. V indices: 2,3,4,7,8,9
        # C_TR: k=2,3,4; i=0,1. W indices: 4,5,6,7,8,9
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    u_idx = i * 5 + j
                    v_idx = j * 5 + (2 + k)
                    w_idx = (2 + k) * 2 + i
                    add_schoolbook_term(u_idx, v_idx, w_idx)
        # col is now 7 + 6 = 13

        # Block 3: A_R (2x3) * B_BL (3x2) -> C_BL (2x2) using schoolbook
        # A_R: i=0,1; j=2,3,4. U indices: 2,3,4,7,8,9
        # B_BL: j=2,3,4; k=0,1. V indices: 10,11,12,15,16,17,20,21,22
        # C_BL: k=0,1; i=0,1. W indices: 0,1,2,3
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    u_idx = i * 5 + (2 + j)
                    v_idx = (2 + j) * 5 + k
                    w_idx = k * 2 + i
                    add_schoolbook_term(u_idx, v_idx, w_idx)
        # col is now 13 + 6 = 19

        # Block 4: A_R (2x3) * B_BR (3x3) -> C_BR (2x3) using schoolbook
        # A_R: i=0,1; j=2,3,4. U indices: 2,3,4,7,8,9
        # B_BR: j=2,3,4; k=2,3,4. V indices: 12,13,14,17,18,19,22,23,24
        # C_BR: k=2,3,4; i=0,1. W indices: 4,5,6,7,8,9
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    u_idx = i * 5 + (2 + j)
                    v_idx = (2 + j) * 5 + (2 + k)
                    w_idx = (2 + k) * 2 + i
                    add_schoolbook_term(u_idx, v_idx, w_idx)
        # col is now 19 + 9 = 28

        # Verify: col should be 28, but we allocated 49. 
        # This means we have 21 unused columns. Let me check the rank calculation.
        # Actually, I think I made an error. The schoolbook rank for each block should be:
        # Block 2: 2*2*3 = 12 terms (not 6)
        # Block 3: 2*3*2 = 12 terms (not 6)
        # Block 4: 2*3*3 = 18 terms (not 9)
        # Total = 7 + 12 + 12 + 18 = 49. That's correct.
        # But in my loops, I'm iterating over i in range(2), j in range(2), k in range(3) for block 2,
        # which gives 2*2*3 = 12 terms. Similarly for others. So col should be 49.
        
        # Let me verify by checking col at the end
        assert col == 49, f"Expected col=49, got {col}"
        
        return {"U": U, "V": V, "W": W}
    
    def compute_tensor(U, V, W):
        """Compute the tensor from factor matrices."""
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct."""
        T = compute_tensor(np.array(U), np.array(V), np.array(W))
        return np.allclose(T, target, atol=1e-9)
    
    # Try the block decomposition first
    block_result = get_block_decomposition()
    
    # Verify the block decomposition
    if verify_decomposition(block_result["U"], block_result["V"], block_result["W"]):
        return block_result
    
    # If verification fails, fall back to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
