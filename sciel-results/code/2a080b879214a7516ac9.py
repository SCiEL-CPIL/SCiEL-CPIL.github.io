import numpy as np
import time

DIMENSIONS = [4, 4, 8]
TIME_LIMIT = 150  # seconds

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Strassen decomposition for 2x2x2
    U2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    V2 = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # Generate 49 terms for 4x4x4
    U4_list = []
    V4_list = []
    W4_list = []
    
    for s in range(7):
        for t in range(7):
            U4 = np.kron(U2[s].reshape(2,2), U2[t].reshape(2,2)).flatten()
            V4 = np.kron(V2[s].reshape(2,2), V2[t].reshape(2,2)).flatten()
            W4 = np.kron(W2[s].reshape(2,2), W2[t].reshape(2,2)).flatten()
            
            U4_list.append(U4)
            V4_list.append(V4)
            W4_list.append(W4)
            
            if time.time() - start_time > TIME_LIMIT - 30:
                break
        if time.time() - start_time > TIME_LIMIT - 30:
            break

    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Precompute index mappings
    # V4 index v4 = k*4 + j (k=0..3, j=0..3)
    # V Block 1 index v = k*8 + j (k=0..3, j=0..3)
    # V Block 2 index v = k*8 + (j+4) (k=0..3, j=0..3)
    
    v4_to_v1 = []
    v4_to_v2 = []
    w4_to_w1 = []
    w4_to_w2 = []
    
    for k in range(4):
        for j in range(4):
            v4 = k * 4 + j
            v1 = k * 8 + j
            v2 = k * 8 + (j + 4)
            v4_to_v1.append((v4, v1))
            v4_to_v2.append((v4, v2))
            
            w4 = k * 4 + j  # Assuming W4 index is i*4+j? 
            # Wait, W4 is from kron(W2, W2).
            # W2[s] is 4 elements. W2 is (7,4).
            # The flatten order of kron(W2[s].reshape(2,2), W2[t].reshape(2,2))
            # is row-major.
            # W2[s].reshape(2,2) is [[a,b],[c,d]].
            # kron gives [[a*A, b*A], [c*A, d*A]].
            # Flatten: a*A[0,0], a*A[0,1], b*A[0,0], b*A[0,1], ...
            # This corresponds to i=0,1 and j=0,1?
            # Let's check 2x2 W2 convention.
            # W2[s] = [w00, w01, w10, w11]?
            # If W2[s] is [1,0,0,1] for s=0.
            # C00 = A00*B00 + A01*B10?
            # Standard Strassen:
            # M1 = (A11 - B22)(B22) ...
            # The W matrix entries are the coefficients for C.
            # If W2[s] is [1,0,0,1], it means C00 += 1*M_s, C11 += 1*M_s?
            # Usually C = [[C00, C01], [C10, C11]].
            # If W2[s] = [1,0,0,1], it adds to C00 and C11.
            # So index 0->C00, 1->C01, 2->C10, 3->C11.
            # So w2 = i*2 + j.
            # For 4x4, w4 = i*4 + j.
            # So my assumption w4 = k*4+j was wrong variable name. It is i*4+j.
            # In the loop above, I used k and j.
            # Let's rename to i and j for W.
            pass

    # Recalculate W mappings with i, j
    for i in range(4):
        for j in range(4):
            w4 = i * 4 + j
            w1 = i * 8 + j
            w2 = i * 8 + (j + 4)
            w4_to_w1.append((w4, w1))
            w4_to_w2.append((w4, w2))

    for r in range(49):
        r2 = 49 + r
        
        # U is same for both blocks
        U[:, r] = U4_list[r]
        U[:, r2] = U4_list[r]
        
        # Scatter V
        V4 = V4_list[r]
        for src_idx, dst_idx in v4_to_v1:
            V[dst_idx, r] = V4[src_idx]
        for src_idx, dst_idx in v4_to_v2:
            V[dst_idx, r2] = V4[src_idx]
            
        # Scatter W
        W4 = W4_list[r]
        for src_idx, dst_idx in w4_to_w1:
            W[dst_idx, r] = W4[src_idx]
        for src_idx, dst_idx in w4_to_w2:
            W[dst_idx, r2] = W4[src_idx]
            
        if time.time() - start_time > TIME_LIMIT - 10:
            break

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
