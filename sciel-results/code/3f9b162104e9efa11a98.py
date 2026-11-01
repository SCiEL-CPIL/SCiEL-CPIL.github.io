"""Rank-44 decomposition for 2x3 * 3x8 matrix multiplication via Strassen and Outer Product blocks."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We decompose M_{2,3,8} into 4 blocks of M_{2,3,2}.
    # Each M_{2,3,2} block has rank 11 (7 for Strassen 2x2x2 + 4 for Outer 2x1x1x2).
    # Total rank = 4 * 11 = 44.
    
    total_rank = 44
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Define the Strassen decomposition for 2x2 * 2x2
    # Indices for local 2x2:
    # A: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 in local U
    # B: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 in local V
    # C: (0,0), (0,1), (1,0), (1,1) -> indices 0, 1, 2, 3 in local W (k*n+i)
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1], # C00 (k=0, i=0)
        [0, 1, 1, 0, 0, 0, 0],  # C01 (k=1, i=0)
        [0, 1, 0, 1, 0, 0, 0],  # C10 (k=0, i=1)
        [1, -1, 1, 0, 0, 1, 0]  # C11 (k=1, i=1)
    ]
    
    # Outer product decomposition for 2x1 * 1x2
    # A: (0,0), (1,0) -> indices 0, 1
    # B: (0,0), (0,1) -> indices 0, 1
    # C: (0,0), (0,1), (1,0), (1,1)
    
    outer_U = [
        [1, 1, 0, 0], # A0
        [0, 0, 1, 1]  # A1
    ]
    
    outer_V = [
        [1, 0, 1, 0], # B0
        [0, 1, 0, 1]  # B1
    ]
    
    outer_W = [
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C01
        [0, 0, 1, 0], # C10
        [0, 0, 0, 1]  # C11
    ]
    
    for b in range(4):
        # k indices for this block: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A1 (2x2) * B1 (2x2) ---
        # A1 uses A[0,0], A[0,1], A[1,0], A[1,1]
        # B1 uses B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        
        # Map local Strassen indices to global
        # A global: i*m + j. Local A: i*2 + j.
        # For j=0,1: Global A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4.
        # Local A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11.
        
        a_map = [0, 1, 3, 4] # Local 0->0, 1->1, 2->3, 3->4
        
        # B global: j*p + k. Local B: j*2 + k.
        # For j=0,1:
        # Local B[0]=B0k0, B[1]=B0k1, B[2]=B1k0, B[3]=B1k1.
        # Global B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=p+k0, B[1,k1]=p+k1.
        
        b_map = [k0, k1, p + k0, p + k1]
        
        # W global: k*n + i. Local W: k*2 + i.
        # For k=0,1:
        # Local W[0]=C0k0, W[1]=C0k1, W[2]=C1k0, W[3]=C1k1.
        # Global W[k0*n+0], W[k0*n+1], W[k1*n+0], W[k1*n+1].
        
        w_map = [k0 * n + 0, k0 * n + 1, k1 * n + 0, k1 * n + 1]
        
        for t in range(7):
            for local_a in range(4):
                if strassen_U[local_a][t] != 0:
                    U[a_map[local_a]][col] = strassen_U[local_a][t]
            for local_b in range(4):
                if strassen_V[local_b][t] != 0:
                    V[b_map[local_b]][col] = strassen_V[local_b][t]
            for local_w in range(4):
                if strassen_W[local_w][t] != 0:
                    W[w_map[local_w]][col] = strassen_W[local_w][t]
            col += 1
            
        if time.time() > deadline:
            # Fallback to schoolbook if running out of time (should not happen)
            return _schoolbook_decomp(n, m, p)
        
        # --- Outer product part for A2 (2x1) * B2 (1x2) ---
        # A2 uses A[0,2], A[1,2]
        # B2 uses B[2,k0], B[2,k1]
        
        # A global: A[0,2]=2, A[1,2]=5.
        a_map_outer = [2, 5]
        
        # B global: B[2,k0]=2p+k0, B[2,k1]=2p+k1.
        b_map_outer = [2 * p + k0, 2 * p + k1]
        
        # W global: Same as Strassen W map.
        
        for t in range(4):
            for local_a in range(2):
                if outer_U[local_a][t] != 0:
                    U[a_map_outer[local_a]][col] = outer_U[local_a][t]
            for local_b in range(2):
                if outer_V[local_b][t] != 0:
                    V[b_map_outer[local_b]][col] = outer_V[local_b][t]
            for local_w in range(4):
                if outer_W[local_w][t] != 0:
                    W[w_map[local_w]][col] = outer_W[local_w][t]
            col += 1

    return {"U": U, "V": V, "W": W}

def _schoolbook_decomp(n, m, p):
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
# EVOLVE-BLOCK-END
