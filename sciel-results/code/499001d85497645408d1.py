import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    # Rank is 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 decomposition patterns
    # Indices for 2x2: i,j,k in {0,1}
    # We store them as lists of (index, value) for non-zero entries
    
    # r=0: M1 = A[0,0]*(B[0,0]+B[1,1])
    U0 = [(0, 1), (3, 1)]   # i=0,j=0; i=1,j=1
    V0 = [(0, 1), (3, 1)]   # j=0,k=0; j=1,k=1
    W0 = [(0, 1), (3, 1)]   # k=0,i=0; k=1,i=1
    
    # r=1: M2 = (A[1,0]+A[1,1])*B[0,0]
    U1 = [(2, 1), (3, 1)]   # i=1,j=0; i=1,j=1
    V1 = [(0, 1)]           # j=0,k=0
    W1 = [(1, 1), (3, -1)]  # k=0,i=1; k=1,i=1
    
    # r=2: M3 = A[0,0]*(B[0,1]+B[1,1])
    U2 = [(0, 1)]           # i=0,j=0
    V2 = [(1, 1), (3, -1)]  # j=0,k=1; j=1,k=1
    W2 = [(2, 1), (3, 1)]   # k=1,i=0; k=1,i=1
    
    # r=3: M4 = A[1,1]*(B[1,0]+B[0,0])
    U3 = [(3, 1)]           # i=1,j=1
    V3 = [(2, 1), (0, -1)]  # j=1,k=0; j=0,k=0
    W3 = [(0, 1), (1, 1)]   # k=0,i=0; k=0,i=1
    
    # r=4: M5 = (A[0,0]+A[0,1])*B[1,1]
    U4 = [(0, 1), (1, 1)]   # i=0,j=0; i=0,j=1
    V4 = [(3, 1)]           # j=1,k=1
    W4 = [(0, -1), (2, 1)]  # k=0,i=0; k=1,i=0
    
    # r=5: M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U5 = [(2, 1), (0, -1)]  # i=1,j=0; i=0,j=0
    V5 = [(0, 1), (1, 1)]   # j=0,k=0; j=0,k=1
    W5 = [(3, 1)]           # k=1,i=1
    
    # r=6: M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U6 = [(1, 1), (3, -1)]  # i=0,j=1; i=1,j=1
    V6 = [(2, 1), (3, 1)]   # j=1,k=0; j=1,k=1
    W6 = [(0, 1)]           # k=0,i=0
    
    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]
    
    # Helper to add a block
    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    # 8 blocks of 2x2x2:
    # A has rows i=0,1 and cols j=0,1,2,3
    # B has rows j=0,1,2,3 and cols k=0..7
    # C has rows k=0..7 and cols i=0,1
    
    # Split into 8 sub-problems: (i_offset, j_offset, k_offset)
    # j_offset in {0, 2}, k_offset in {0, 2, 4, 6}
    blocks = []
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            blocks.append((j_off, k_off))
    
    for idx, (j_off, k_off) in enumerate(blocks):
        base_rank = idx * 7
        add_block(base_rank, 0, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
