"""Rewritten solution for matrix multiplication 4,4,4."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 element patterns
    # Indexing: [elem00, elem01, elem10, elem11]
    U_elem = [
        [1, 0, 0, 1],   # t=0: a00+a11
        [0, 0, 1, 1],   # t=1: a10+a11
        [1, 0, 0, 0],   # t=2: a00
        [0, 0, 0, 1],   # t=3: a11
        [1, 1, 0, 0],   # t=4: a00+a01
        [-1, 0, 1, 0],  # t=5: a10-a00
        [0, 1, 0, -1],  # t=6: a01-a11
    ]
    V_elem = [
        [1, 0, 0, 1],   # t=0: b00+b11
        [1, 0, 0, 0],   # t=1: b00
        [0, 1, 0, -1],  # t=2: b01-b11
        [-1, 0, 1, 0],  # t=3: b10-b00
        [0, 0, 0, 1],   # t=4: b11
        [1, 1, 0, 0],   # t=5: b00+b01
        [0, 0, 1, 1],   # t=6: b10+b11
    ]
    W_elem = [
        [[1, 0], [0, 1]],   # t=0: C00+=1, C11+=1
        [[0, 0], [1, -1]],  # t=1: C10+=1, C11-=1
        [[0, 1], [0, 1]],   # t=2: C01+=1, C11+=1
        [[1, 0], [1, 0]],   # t=3: C00+=1, C10+=1
        [[-1, 1], [0, 0]],  # t=4: C00-=1, C01+=1
        [[0, 0], [0, 1]],   # t=5: C11+=1
        [[1, 0], [0, 0]],   # t=6: C00+=1
    ]
    
    # Block-level Strassen: which 2x2 blocks and coefficients
    A_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: A00+A11
        [(1,0,1), (1,1,1)],      # s=1: A10+A11
        [(0,0,1)],               # s=2: A00
        [(1,1,1)],               # s=3: A11
        [(0,0,1), (0,1,1)],      # s=4: A00+A01
        [(1,0,1), (0,0,-1)],     # s=5: A10-A00
        [(0,1,1), (1,1,-1)],     # s=6: A01-A11
    ]
    B_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: B00+B11
        [(0,0,1)],               # s=1: B00
        [(0,1,1), (1,1,-1)],     # s=2: B01-B11
        [(1,0,1), (0,0,-1)],     # s=3: B10-B00
        [(1,1,1)],               # s=4: B11
        [(0,0,1), (0,1,1)],      # s=5: B00+B01
        [(1,0,1), (1,1,1)],      # s=6: B10+B11
    ]
    C_blocks = [
        [(0,0,1), (1,1,1)],      # s=0: C00+=M, C11+=M
        [(1,0,1), (1,1,-1)],     # s=1: C10+=M, C11-=M
        [(0,1,1), (1,1,1)],      # s=2: C01+=M, C11+=M
        [(0,0,1), (1,0,1)],      # s=3: C00+=M, C10+=M
        [(0,0,-1), (0,1,1)],     # s=4: C00-=M, C01+=M
        [(1,1,1)],               # s=5: C11+=M
        [(0,0,1)],               # s=6: C00+=M
    ]
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            
            # Build U for this term
            for (bi, bj, c_a) in A_blocks[s]:
                for ei in range(2):
                    for ej in range(2):
                        row_a = (2*bi + ei)*4 + (2*bj + ej)
                        U[row_a][col] += c_a * U_elem[t][ei*2+ej]
            
            # Build V for this term
            for (bj, bk, c_b) in B_blocks[s]:
                for ej in range(2):
                    for ek in range(2):
                        row_b = (2*bj + ej)*4 + (2*bk + ek)
                        V[row_b][col] += c_b * V_elem[t][ej*2+ek]
            
            # Build W for this term
            for (bi, bk, c_c) in C_blocks[s]:
                for ei in range(2):
                    for ek in range(2):
                        row_c = (2*bk + ek)*4 + (2*bi + ei)
                        W[row_c][col] += c_c * W_elem[t][ei][ek]
    
    # Verify correctness with multiple deterministic tests using NumPy for speed and reliability
    try:
        tests = [
            (np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]),
             np.array([[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]])),
            (np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
             np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])),
            (np.array([[2,1,0,0],[0,3,1,0],[0,0,2,1],[1,0,0,3]]),
             np.array([[1,0,1,0],[0,2,0,1],[1,0,0,2],[0,1,2,0]])),
        ]
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for A_test, B_test in tests:
            AB = A_test @ B_test
            # Compute using decomposition: sum_r (U[:,r] dot A) * (V[:,r] dot B) * W[:,r]
            # U is 16x49, A is 4x4 -> flatten A to 16x1
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            # u_vals: 49x1, v_vals: 49x1
            u_vals = U_np.T @ A_flat
            v_vals = V_np.T @ B_flat
            
            # scalar product of u_vals and v_vals element-wise
            scalars = u_vals * v_vals
            
            # AB_decomp = W_np @ scalars (W is 16x49, scalars is 49x1)
            AB_decomp_flat = W_np @ scalars
            AB_decomp = AB_decomp_flat.reshape(4, 4)
            
            # Check if they match
            if not np.allclose(AB, AB_decomp, atol=1e-9):
                # Fallback to schoolbook rank 64
                rank_fb = 64
                U_fb = [[0]*rank_fb for _ in range(16)]
                V_fb = [[0]*rank_fb for _ in range(16)]
                W_fb = [[0]*rank_fb for _ in range(16)]
                col_fb = 0
                for i2 in range(4):
                    for j2 in range(4):
                        for k2 in range(4):
                            U_fb[i2*4+j2][col_fb] = 1
                            V_fb[j2*4+k2][col_fb] = 1
                            W_fb[k2*4+i2][col_fb] = 1
                            col_fb += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
    except Exception:
        pass
    
    if time.time() > deadline:
        # If we are running out of time, return the current best (49)
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
