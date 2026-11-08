import numpy as np
import time

def construct():
    start_time = time.time()
    n, m, p = 4, 4, 8
    
    # Rank 98 decomposition: two independent 4x4x4 multiplications
    # Each 4x4x4 uses recursive Strassen (rank 49)
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 2x2x2 Strassen terms
    # Each term is (u_dict, v_dict, w_dict)
    # Keys are (row, col)
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Block products for 4x4x4
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # P0 = (A00+A11)*(B00+B11) -> C00+C11
    # P1 = (A10+A11)*B00 -> C10+C11
    # P2 = A00*(B01-B11) -> C01+C11
    # P3 = A11*(B10-B00) -> C00+C10
    # P4 = (A00+A01)*B11 -> C00+C01
    # P5 = (A10-A00)*(B00+B01) -> C11
    # P6 = (A01-A11)*(B10+B11) -> C00
    
    # We need to map these to the 2x2 Strassen structure.
    # For each block product, we define the linear combination of 2x2 blocks in A, B, C.
    # A_blocks: list of (block_r, block_c, coeff)
    # B_blocks: list of (block_r, block_c, coeff)
    # C_blocks: list of (block_r, block_c, coeff)
    
    block_products = [
        # P0
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        # P1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        # P2
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        # P3
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        # P4
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        # P5
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        # P6
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ]),
    ]
    
    col = 0
    for half in range(2): # 0: left, 1: right
        b_offset = half * 4
        
        for bp_idx in range(7):
            A_blocks, B_blocks, C_blocks = block_products[bp_idx]
            
            # For each 2x2x2 Strassen term
            for s_idx in range(7):
                u_pat, v_pat, w_pat = strassen_2x2[s_idx]
                
                # Accumulate U, V, W contributions
                u_vec = np.zeros(16)
                v_vec = np.zeros(32)
                w_vec = np.zeros(32)
                
                # Process U (A)
                # A is 4x4. Index i*4+k.
                # Block (br, bc) corresponds to A[br*2 : br*2+2, bc*2 : bc*2+2]
                for (ar, ac, a_coeff) in A_blocks:
                    for (lr, ls, u_coeff) in u_pat.items():
                        global_i = ar * 2 + lr
                        global_k = ac * 2 + ls
                        global_A_idx = global_i * 4 + global_k
                        u_vec[global_A_idx] += a_coeff * u_coeff
                
                # Process V (B)
                # B is 4x8. Index k*8+j.
                # Block (br, bc) corresponds to B[br*2 : br*2+2, b_offset + bc*2 : b_offset + bc*2+2]
                for (br, bc, b_coeff) in B_blocks:
                    for (lr, ls, v_coeff) in v_pat.items():
                        global_k = br * 2 + lr
                        local_j = bc * 2 + ls
                        global_j = b_offset + local_j
                        global_B_idx = global_k * 8 + global_j
                        v_vec[global_B_idx] += b_coeff * v_coeff
                
                # Process W (C)
                # C is 4x8. Index j*4+i (trace convention W[k*n+i,r] -> W[j*n+i] for n=4? No, W is p*n = 8*4=32. Index is j*4+i? 
                # Task says: W[k*n+i,r]. Here n=4. So index is k*4+i? 
                # Wait, "output factor follows the official trace convention W[k*n+i,r]".
                # Usually C = AB. C is p x n. 
                # Tensor T_{i,k,j} = C_{i,j} = sum_k A_{i,k} B_{k,j}.
                # W vector for a term r is indexed by (i,j).
                # The task says W shape is (p*n, rank). 
                # If W is flattened as row-major of C (p rows, n cols), index is j*n + i? Or i*p + j?
                # "W[k*n+i,r]" suggests the index is $k \cdot n + i$.
                # In C (p x n), if we flatten row-major, element (j, i) is at $j \cdot n + i$.
                # So $k$ in the formula corresponds to $j$ (row of C) and $i$ corresponds to $i$ (col of C).
                # So global_W_idx = global_j * 4 + global_i.
                
                for (cr, cc, c_coeff) in C_blocks:
                    for (lr, ls, w_coeff) in w_pat.items():
                        global_i = cr * 2 + lr
                        local_j = cc * 2 + ls
                        global_j = b_offset + local_j
                        global_W_idx = global_j * 4 + global_i
                        w_vec[global_W_idx] += c_coeff * w_coeff
                
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
                
                if col >= rank:
                    break
            if col >= rank:
                break
        if col >= rank:
            break
            
    elapsed = time.time() - start_time
    if elapsed > 140:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
