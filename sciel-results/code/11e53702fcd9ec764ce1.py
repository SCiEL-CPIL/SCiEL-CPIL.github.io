import numpy as np
def construct():
        n, m, p = 2, 4, 6
        rank = 42
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen local factors
        # U_local[i, t] where i is local A index (0..3), t is term (0..6)
        # V_local[j, t] where j is local B index (0..3), t is term (0..6)
        # W_local[k, t] where k is local C index (0..3), t is term (0..6)
        
        # Local indices:
        # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U_local = np.array([
            [1, 0, 1, 0, 1, -1, 0], # a00
            [0, 0, 0, 0, 1, 0, 1],  # a01
            [0, 1, 0, 0, 0, 1, 0],  # a10
            [1, 1, 0, 1, 0, 0, -1]  # a11
        ])
        
        V_local = np.array([
            [1, 1, 0, -1, 0, 1, 0], # b00
            [0, 0, 1, 0, 0, 1, 0],  # b01
            [0, 0, 0, 1, 0, 0, 1],  # b10
            [1, 0, -1, 0, 1, 0, 1]  # b11
        ])
        
        W_local = np.array([
            [1, 0, 0, 1, -1, 0, 1], # c00
            [0, 0, 1, 0, 1, 0, 0],  # c01
            [0, 1, 0, 1, 0, 0, 0],  # c10
            [1, -1, 1, 0, 0, 1, 0]  # c11
        ])
        
        # Verify W_local against formulas:
        # c00 = M1 + M4 - M5 + M7 -> [1, 0, 0, 1, -1, 0, 1] Correct.
        # c01 = M3 + M5 -> [0, 0, 1, 0, 1, 0, 0] Correct.
        # c10 = M2 + M4 -> [0, 1, 0, 1, 0, 0, 0] Correct.
        # c11 = M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 0, 1, 0] Correct.
        
        # Define the 6 blocks
        # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
        # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
        # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
        
        blocks = [
            (0, 0, 0), # a_j_start, b_j_start, b_k_start
            (2, 2, 0),
            (0, 0, 2),
            (2, 2, 2),
            (0, 0, 4),
            (2, 2, 4)
        ]
        
        for b_idx, (a_js, b_js, b_ks) in enumerate(blocks):
            # a_js: start col in A (0 or 2)
            # b_js: start row in B (0 or 2)
            # b_ks: start col in B (0, 2, or 4)
            
            # Map local indices to global
            # A global index: i * m + j
            # i is 0 or 1. j is a_js or a_js+1.
            # Local A indices:
            # 0: (0, a_js) -> 0 * 4 + a_js
            # 1: (0, a_js+1) -> 0 * 4 + a_js + 1
            # 2: (1, a_js) -> 1 * 4 + a_js
            # 3: (1, a_js+1) -> 1 * 4 + a_js + 1
            
            a_map = [
                0 * m + a_js,
                0 * m + a_js + 1,
                1 * m + a_js,
                1 * m + a_js + 1
            ]
            
            # B global index: j * p + k
            # j is b_js or b_js+1. k is b_ks or b_ks+1.
            # Local B indices:
            # 0: (b_js, b_ks)
            # 1: (b_js, b_ks+1)
            # 2: (b_js+1, b_ks)
            # 3: (b_js+1, b_ks+1)
            
            b_map = [
                b_js * p + b_ks,
                b_js * p + b_ks + 1,
                (b_js + 1) * p + b_ks,
                (b_js + 1) * p + b_ks + 1
            ]
            
            # C global index: k * n + i
            # k is b_ks or b_ks+1. i is 0 or 1.
            # Local C indices:
            # 0: (b_ks, 0)
            # 1: (b_ks+1, 0)
            # 2: (b_ks, 1)
            # 3: (b_ks+1, 1)
            
            c_map = [
                b_ks * n + 0,
                (b_ks + 1) * n + 0,
                b_ks * n + 1,
                (b_ks + 1) * n + 1
            ]
            
            # Fill U, V, W
            for t in range(7):
                col = b_idx * 7 + t
                for l in range(4):
                    U[a_map[l], col] += U_local[l, t]
                    V[b_map[l], col] += V_local[l, t]
                    W[c_map[l], col] += W_local[l, t]
                    
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
