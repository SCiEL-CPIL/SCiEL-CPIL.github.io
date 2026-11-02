import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # 2x2 Strassen decomposition vectors
    # u, v, w are 4x7 matrices
    # c=0..6
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Check 2x2 decomposition
    # T_{i,j,k} = sum_c U2[i,j,c] V2[j,k,c] W2[k,i,c]
    # i,j,k in 0..1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"2x2 decomposition failed at {i},{j},{k} with sum {s}")
                    
    # 4x4 construction
    # Blocks: 11, 12, 21, 22
    # A blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1)
    # B blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1)
    # C blocks: 11 (0,0), 12 (0,1), 21 (1,0), 22 (1,1)
    
    # Products P1..P7
    # Each product has A-blocks, B-blocks, C-blocks with coefficients
    # Format: list of (block, coeff)
    products = [
        # P1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # P2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # P3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # P4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # P5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # P6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # P7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    col = 0
    for m in range(7):
        a_blocks, b_blocks, c_blocks = products[m]
        for c in range(7):
            # Construct u
            u = np.zeros(16)
            for (br, bc, coeff) in a_blocks:
                # br, bc are block row, col (0..1)
                # Block is 2x2
                # Indices in u: i in 2*br..2*br+1, j in 2*bc..2*bc+1
                # u index is i*4 + j
                for i in range(2):
                    for j in range(2):
                        u[(2*br + i)*4 + (2*bc + j)] += coeff * U2[i*2+j, c]
            
            # Construct v
            v = np.zeros(16)
            for (br, bc, coeff) in b_blocks:
                for i in range(2):
                    for j in range(2):
                        v[(2*br + i)*4 + (2*bc + j)] += coeff * V2[i*2+j, c]
                        
            # Construct w
            w = np.zeros(16)
            for (br, bc, coeff) in c_blocks:
                # w index is k*4 + i
                # Block is C_{br, bc}
                # i in 2*br..2*br+1, k in 2*bc..2*bc+1
                for i in range(2):
                    for k in range(2):
                        w[(2*bc + k)*4 + (2*br + i)] += coeff * W2[k*2+i, c]
                        
            U4[:, col] = u
            V4[:, col] = v
            W4[:, col] = w
            col += 1
            
    if time.time() > deadline:
        # Return current best (which is the full 49)
        pass
        
    # Verify 4x4 decomposition
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for c in range(49):
                    s += U4[i*4+j, c] * V4[j*4+k, c] * W4[k*4+i, c]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"4x4 decomposition failed at {i},{j},{k} with sum {s}")
                    
    # Convert to lists of lists for the required format
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U4.tolist()]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V4.tolist()]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W4.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
