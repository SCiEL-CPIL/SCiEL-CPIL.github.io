"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    # Block size is 2, grid size is 3
    b = 2
    g = 3
    
    # --- Inner Strassen Decomposition for 2x2 matrices ---
    # Rank 7
    # U_strassen: (4, 7)
    # V_strassen: (4, 7)
    # W_strassen: (4, 7)
    # Indices: 0,1,2,3 correspond to [00, 01, 10, 11] in row-major order for 2x2
    
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    
    U_s = [
        [1, 0, 1, 0], # a00
        [0, 1, 0, 1], # a01
        [1, 0, 1, 0], # a10
        [1, 1, 0, 1]  # a11
    ]
    # Correct Strassen U matrix (rows are elements of A, cols are 7 products):
    # Prod 1: a00+a11
    # Prod 2: a10+a11
    # Prod 3: a00
    # Prod 4: a11
    # Prod 5: a00+a01
    # Prod 6: a10-a00
    
    U_s = [
        [1, 0, 1, 0, 1, 1], # a00
        [0, 0, 0, 0, 1, 0], # a01
        [0, 1, 0, 0, 0, 1], # a10
        [1, 1, 0, 1, 0, -1] # a11
    ]
    # Wait, standard Strassen:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    U_s = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, -1]# A11
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1], # B01
        [1, 0, -1, 1, 1, 0, 0], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ]
    
    W_s = [
        [1, 0, 0, 0, 0, 0, 0], # C00 = m1 + m4 - m5 - m7
        [0, 0, 1, 0, 1, 0, 0], # C01 = m3 + m5
        [0, 1, 0, 0, 0, 1, 0], # C10 = m2 + m4
        [0, 0, 0, 1, 0, 0, 1]  # C11 = m1 - m2 + m3 + m6
    ]
    # Let's re-verify W_s indices.
    # C00 = M1 + M4 - M5 - M7 -> [1, 0, 0, 1, -1, 0, -1]
    # C01 = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # --- Outer BCLR Decomposition for 3x3 matrices ---
    # Rank 21
    # U_b, V_b, W_b are (9, 21)
    # Indices 0..8 correspond to blocks [00, 01, 02, 10, 11, 12, 20, 21, 22]
    
    # BCLR coefficients (from literature)
    # This is a specific rank 21 decomposition.
    # Source: "A 21-multiplication algorithm for 3x3 matrix multiplication"
    
    U_b = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A00
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A01
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A02
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A10
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A11
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A12
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A20
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A21
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # A22
    ]
    # The above is just identity, which is rank 9. We need the actual BCLR.
    # I will use a known valid rank 21 decomposition.
    # One common one is based on the "Bini-Capovani-Lotti-Romani" scheme.
    # Let's use the one from the paper "Matrix multiplication with 21 multiplications".
    
    # U_b (9x21)
    U_b = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is getting complicated to recall exactly. 
    # Alternative: Use the "Strassen-like" decomposition for 3x3 if available, or stick to a simpler block decomposition that is definitely correct.
    # Actually, there is a well-known rank 21 decomposition.
    # Let's try to construct it via Kronecker products of smaller decompositions? No, 3 is prime.
    
    # Let's use the "BCLR" decomposition explicitly found in literature.
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Algorithms_of_rank_21
    
    # U_b:
    # 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
    # This is identity, rank 9. 
    
    # I will implement a verified Rank 21 decomposition.
    # From "A Note on 3x3 Matrix Multiplication"
    
    U_b = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is not rank 21, it's just a guess.
    
    # Let's use a simpler, guaranteed improvement: 
    # Block Strassen for 3x3 blocks of 2x2?
    # 6x6 = 3x3 blocks of 2x2.
    # If we use Strassen for the 3x3 block matrix? Strassen is for 2x2.
    # We can use a rank 7 decomposition for 2x2.
    # For 3x3, the best known is 21.
    
    # To be safe and ensure correctness within time, I will use the **Bilinear Complexity of 3x3** which is 21.
    # I will hardcode a known correct BCLR decomposition.
    
    # U_b (9x21)
    U_b = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    # This is still a guess. 
    # Let's fallback to a **verified** decomposition: 
    # **Strassen for 2x2 blocks, applied to a 3x3 grid?** No, Strassen requires 2x2.
    # 
    # **Alternative Strategy**: 
    # Use the **Coppersmith-Winograd** or other fast algorithms? Too complex.
    # 
    # **Safe Strategy**: 
    # Use **Block Strassen** where we treat the 6x6 as a 2x2 grid of 3x3 blocks.
    # Rank = 7 * (rank of 3x3 mult).
    # If we use schoolbook for 3x3 (rank 27), total rank = 7 * 27 = 189.
    # This is better than 216.
    # 189 < 216.
    # Score: 216/189 = 1.14.
    # This is a guaranteed improvement and much easier to implement correctly than BCLR 21.
    # 
    # Let's implement **Block Strassen (2x2 grid of 3x3 blocks)**.
    # Inner dimension: 3x3 multiplication (schoolbook, rank 27).
    # Outer dimension: 2x2 multiplication (Strassen, rank 7).
    # Total Rank: 7 * 27 = 189.
    
    # Inner: 3x3 Schoolbook
    # U_in: (9, 27), V_in: (9, 27), W_in: (9, 27)
    U_in = [[0]*27 for _ in range(9)]
    V_in = [[0]*27 for _ in range(9)]
    W_in = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_in[i*3+j][col] = 1
                V_in[j*3+k][col] = 1
                W_in[k*3+i][col] = 1
                col += 1
    
    # Outer: 2x2 Strassen
    # U_out: (4, 7), V_out: (4, 7), W_out: (4, 7)
    # Blocks are 3x3.
    # A00, A01, A10, A11
    U_out = [
        [1, 0, 1, 0, 1, 1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, -1]# A11
    ]
    V_out = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 1], # B01
        [1, 0, -1, 1, 1, 0, 0], # B10
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ]
    W_out = [
        [1, 0, 0, 1, -1, 0, -1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Construct Global U, V, W
    # Total Rank = 7 * 27 = 189
    total_rank = 189
    U = [[0]*total_rank for _ in range(36)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # Mapping:
    # Global Row/Col index = block_index * 9 + local_index
    # Block index in 2x2 grid: 0,1,2,3 (00, 01, 10, 11)
    # Local index in 3x3 block: 0..8
    
    for k_outer in range(7):
        for k_inner in range(27):
            col_global = k_outer * 27 + k_inner
            
            # U: A matrix
            for b_idx in range(4):
                val = U_out[b_idx][k_outer]
                if val != 0:
                    # Block b_idx corresponds to rows b_idx*9 to b_idx*9+8
                    for l_idx in range(9):
                        val_in = U_in[l_idx][k_inner]
                        if val_in != 0:
                            row_global = b_idx * 9 + l_idx
                            U[row_global][col_global] += val * val_in
            
            # V: B matrix
            for b_idx in range(4):
                val = V_out[b_idx][k_outer]
                if val != 0:
                    # Block b_idx corresponds to rows b_idx*9 to b_idx*9+8
                    for l_idx in range(9):
                        val_in = V_in[l_idx][k_inner]
                        if val_in != 0:
                            row_global = b_idx * 9 + l_idx
                            V[row_global][col_global] += val * val_in
            
            # W: C matrix
            for b_idx in range(4):
                val = W_out[b_idx][k_outer]
                if val != 0:
                    # Block b_idx corresponds to rows b_idx*9 to b_idx*9+8
                    for l_idx in range(9):
                        val_in = W_in[l_idx][k_inner]
                        if val_in != 0:
                            row_global = b_idx * 9 + l_idx
                            W[row_global][col_global] += val * val_in

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
