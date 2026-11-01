import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56  # 8 blocks * 7 Strassen steps
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen local vectors
    # u_s, v_s, w_s for s in 0..6
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Note: My local C index mapping above was:
    # c00 -> C_{0,0}, c01 -> C_{0,1}, c10 -> C_{1,0}, c11 -> C_{1,1}
    # But in the global mapping:
    # c00 (local 0) -> C_{0, 2r} -> W_{4r}
    # c01 (local 1) -> C_{0, 2r+1} -> W_{4r+2}
    # c10 (local 2) -> C_{1, 2r} -> W_{4r+1}
    # c11 (local 3) -> C_{1, 2r+1} -> W_{4r+3}
    
    # Let's re-verify the w vectors based on this mapping.
    # Step 1: w1 = [1, 0, 0, 1] -> c00, c11
    # c00 is local 0. c11 is local 3.
    # w1[0]=1, w1[3]=1.
    # Step 2: w2 = [0, 0, 1, -1] -> c10, c11
    # c10 is local 2. c11 is local 3.
    # w2[2]=1, w2[3]=-1.
    # Step 3: w3 = [0, 1, 0, 1] -> c01, c11
    # c01 is local 1. c11 is local 3.
    # w3[1]=1, w3[3]=1.
    # Step 4: w4 = [1, 0, 1, 0] -> c00, c10
    # c00 is local 0. c10 is local 2.
    # w4[0]=1, w4[2]=1.
    # Step 5: w5 = [-1, 1, 0, 0] -> c00, c01
    # c00 is local 0. c01 is local 1.
    # w5[0]=-1, w5[1]=1.
    # Step 6: w6 = [0, 0, 0, 1] -> c11
    # c11 is local 3.
    # w6[3]=1.
    # Step 7: w7 = [1, 0, 0, 0] -> c00
    # c00 is local 0.
    # w7[0]=1.
    
    u_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    v_vecs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    w_vecs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V
            # Local B index li -> Global V index
            # li=0 -> B_{0, 2r} -> V_{2r}
            # li=1 -> B_{0, 2r+1} -> V_{2r+1}
            # li=2 -> B_{1, 2r} -> V_{16+2r}
            # li=3 -> B_{1, 2r+1} -> V_{16+2r+1}
            v_indices = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_indices[li]][col] = v_vecs[s][li]
                    
            # W
            # Local C index li -> Global W index
            # li=0 -> C_{0, 2r} -> W_{4r}
            # li=1 -> C_{0, 2r+1} -> W_{4r+2}
            # li=2 -> C_{1, 2r} -> W_{4r+1}
            # li=3 -> C_{1, 2r+1} -> W_{4r+3}
            w_indices = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_indices[li]][col] = w_vecs[s][li]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
