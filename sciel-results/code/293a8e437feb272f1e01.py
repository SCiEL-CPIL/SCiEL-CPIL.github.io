import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    
    # Target: Rank 85 decomposition
    # Core 4x4x4 rank 49
    # Extra 4x4x1 rank 16
    # Extra 4x1x5 rank 20
    # Total 85
    
    # 1. Define 2x2x2 Strassen factors
    # U2, V2, W2 shape (4, 7)
    # Indexing: row*2 + col
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # M2 = (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[2,1]=1
    
    # M3 = A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[1,2]=1
    
    # M4 = A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[2,3]=1
    
    # M5 = (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[1,4]=1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1; W2[3,6]=1

    # 2. Construct 4x4x4 Strassen factors
    # U4, V4, W4 shape (16, 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen terms for 4x4
    # Each term l defines A_in, B_in, C_out as linear combos of 2x2 blocks
    # Block indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # A_in = sum coeff * A_block
    # The 2x2 Strassen operates on the "virtual" 2x2 matrix A_in.
    # We need to map the virtual elements a00, a01, a10, a11 to the actual 4x4 elements.
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
    ]
    
    for l, (A_blocks, B_blocks, C_blocks) in enumerate(terms):
        for s in range(7):
            col = l * 7 + s
            
            # Determine the linear form for U (A side)
            # u_s(a) = sum_{i in 0..3} U2[i,s] * a_i
            # a_i is the i-th element of the virtual 2x2 matrix A_in.
            # A_in = sum_{(br, bc, coeff) in A_blocks} coeff * A_{br, bc}
            # So a_i = sum_{(br, bc, coeff) in A_blocks} coeff * (A_{br, bc})[i//2, i%2]
            # Thus U4[r, col] += sum_{(br, bc, coeff) in A_blocks} coeff * U2[i,s] * delta(r, br, bc, i)
            
            # We iterate over the virtual elements i (0..3)
            for i in range(4):
                val_u = U2[i, s]
                if val_u == 0: continue
                # i corresponds to row i//2, col i%2 in the 2x2 block
                vi_r = i // 2
                vi_c = i % 2
                for (br, bc, coeff) in A_blocks:
                    # The element A_{br, bc}[vi_r, vi_c] is at global index (br*2+vi_r)*4 + (bc*2+vi_c)
                    gr = br * 2 + vi_r
                    gc = bc * 2 + vi_c
                    U4[gr * 4 + gc, col] += coeff * val_u
            
            # Similarly for V (B side)
            for i in range(4):
                val_v = V2[i, s]
                if val_v == 0: continue
                vi_r = i // 2
                vi_c = i % 2
                for (br, bc, coeff) in B_blocks:
                    gr = br * 2 + vi_r
                    gc = bc * 2 + vi_c
                    V4[gr * 4 + gc, col] += coeff * val_v
            
            # Similarly for W (C side)
            # W index is k*n + i. For 4x4x4, n=4. Index k*4+i.
            # The virtual C_out is 2x2.
            # c_i is the i-th element of C_out.
            # C_out = sum_{(br, bc, coeff) in C_blocks} coeff * C_{br, bc}
            # So w_s(c) = sum_{i} W2[i,s] * c_i
            # c_i = sum_{(br, bc, coeff) in C_blocks} coeff * (C_{br, bc})[i//2, i%2]
            # W4[global_idx, col] += sum ...
            for i in range(4):
                val_w = W2[i, s]
                if val_w == 0: continue
                vi_r = i // 2
                vi_c = i % 2
                for (br, bc, coeff) in C_blocks:
                    # C_{br, bc} is a 2x2 block of the 4x4 result matrix.
                    # br, bc are block row/col in 2x2 grid of blocks.
                    # Global row = br*2 + vi_r, Global col = bc*2 + vi_c.
                    # W index convention: W[k*n + i, r] where k is col, i is row?
                    # Task says: W[k*n+i, r].
                    # In matrix mult C[i,k], i is row, k is col.
                    # So W index = k*n + i.
                    # Here n=4.
                    gr = br * 2 + vi_r
                    gc = bc * 2 + vi_c
                    W4[gc * 4 + gr, col] += coeff * val_w

    # 3. Combine into full 4x5x5
    # Core rank 49
    # Part 2: j in 0..3, k=4. A[i,j] B[j,4] -> C[i,4]. Rank 16.
    # Part 3: j=4, k in 0..4. A[i,4] B[4,k] -> C[i,k]. Rank 20.
    # Total rank 85.
    
    total_rank = 49 + 16 + 20
    
    U_full = np.zeros((20, total_rank))
    V_full = np.zeros((25, total_rank))
    W_full = np.zeros((20, total_rank))
    
    # Copy Core (4x4x4)
    # U_full[i*5+j, r] for i,j in 0..3
    for i in range(4):
        for j in range(4):
            U_full[i * 5 + j, :49] = U4[i * 4 + j, :]
            
    # V_full[j*5+k, r] for j,k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j * 5 + k, :49] = V4[j * 4 + k, :]
            
    # W_full[k*4+i, r] for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k * 4 + i, :49] = W4[k * 4 + i, :]
            
    # Part 2: C[i,4] += A[i,j] B[j,4] for j=0..3
    # 16 terms
    for i in range(4):
        for j in range(4):
            r = 49 + i * 4 + j
            U_full[i * 5 + j, r] = 1
            V_full[j * 5 + 4, r] = 1
            # C[i,4] -> W index 4*4 + i = 16 + i
            W_full[16 + i, r] = 1
            
    # Part 3: C[i,k] += A[i,4] B[4,k] for k=0..4
    # 20 terms
    for i in range(4):
        for k in range(5):
            r = 49 + 16 + i * 5 + k
            U_full[i * 5 + 4, r] = 1
            V_full[4 * 5 + k, r] = 1
            # C[i,k] -> W index k*4 + i
            W_full[k * 4 + i, r] = 1

    # Verify
    def verify():
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    val = np.dot(U_full[i * 5 + j, :], V_full[j * 5 + k, :] * W_full[k * 4 + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if verify():
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback to schoolbook 100
        rank = 100
        U = np.zeros((20, rank))
        V = np.zeros((25, rank))
        W = np.zeros((20, rank))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i * 5 + j, col] = 1
                    V[j * 5 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
