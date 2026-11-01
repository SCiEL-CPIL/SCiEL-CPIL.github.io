import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 2, 4, 6
    
    # Kronecker product: (2x2x2 Strassen) ⊗ (1x2x3)
    # T1: 2x2x2, rank 7 (Strassen)
    # T2: 1x2x3, rank 3
    # Total: 2x4x6, rank 21
    
    rank = 21
    
    # Strassen factors for 2x2x2
    # Local indices: 0:00, 1:01, 2:10, 3:11
    sU = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    sV = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(1, 1), (3, -1)],
        [(2, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    sW = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (3, 1)],
        [(0, 1), (2, 1)],
        [(0, -1), (1, 1)],
        [(3, 1)],
        [(0, 1)]
    ]
    
    # T2 factors for 1x2x3: rank 3
    # n2=1, m2=2, p2=3
    # U2 has shape (1*2, 3) = (2, 3)
    # V2 has shape (2*3, 3) = (6, 3)
    # W2 has shape (3*1, 3) = (3, 3)
    # For r2 in {0,1,2} (corresponding to k2=r2):
    #   U2[i2, r2] = 1 for i2 in {0,1} (both rows of A contribute)
    #   V2[j2*3+k2, r2] = 1 if k2==r2 else 0
    #   W2[k2*1+i2, r2] = 1 if k2==r2 else 0
    # Wait, W2 index: W[k*n+i, r] so for n2=1: W2[k2, r2]
    
    U2 = np.zeros((2, 3), dtype=np.int64)
    V2 = np.zeros((6, 3), dtype=np.int64)
    W2 = np.zeros((3, 3), dtype=np.int64)
    
    for r2 in range(3):
        for i2 in range(2):  # i2 in {0,1} for the 1x2 matrix A
            U2[i2, r2] = 1
        for j2 in range(2):  # j2 in {0,1}
            k2 = r2
            V2[j2 * 3 + k2, r2] = 1
        k2 = r2
        W2[k2, r2] = 1
    
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for r in range(rank):
        r1 = r // 3  # Strassen index 0-6
        r2 = r % 3   # T2 index 0-2
        
        # U[i1*2 + i2, r] = U1[i1, r1] * U2[i2, r2]
        # U1[i1, r1] from sU[r1]
        for li, val in sU[r1]:
            i1 = li // 2
            # U2[i2, r2] = 1 for i2 in {0,1}
            for i2 in range(2):
                U[i1 * 2 + i2, r] += val * U2[i2, r2]
        
        # V[j1*3 + j2, r] = V1[j1, r1] * V2[j2, r2]
        # Wait, V index: V[j*m*p... no. V has shape (m*p, rank) = (4*6, rank)
        # For Kronecker: V[(j1*m2 + j2) * (p1*p2) + ... 
        # Actually: the B matrix is m x p = 4 x 6
        # B[j, k] where j in {0..3}, k in {0..5}
        # j = j1*2 + j2, k = k1*3 + k2
        # V index: j * p + k = (j1*2+j2)*6 + (k1*3+k2)
        # V1[j1, r1] from sV[r1], where j1 in {0,1} maps to local indices 0-3
        # V2[j2*3+k2, r2]
        
        for li, val in sV[r1]:
            j1_local = li  # 0-3
            j1 = j1_local // 2
            # Hmm, sV uses local indices 0-3 for 2x2, where 0:00, 1:01, 2:10, 3:11
            # So j1 = li // 2 gives the row in the 2x2 block
            # But we need to map this to the Kronecker structure
            # In the 2x2x2 tensor, the B index is j1 (0 or 1)
            # In the 4x6 matrix, B index is (j1, j2) where j1 in {0,1}, j2 in {0,1}
            # So global j = j1*2 + j2
            # And k = k1*3 + k2
            # V global index = j*6 + k = (j1*2+j2)*6 + (k1*3+k2)
            
            # But sV[r1] gives contributions to local B indices 0-3
            # Local index li: row = li//2, col = li%2 in the 2x2 B block
            # In Kronecker, B row j1 corresponds to li//2
            # B col k1 corresponds to li%2
            # So for each j2 and k2:
            #   V[(li//2)*2 + j2, ...] no wait
            # V index = (j1*2+j2)*6 + (k1*3+k2)
            # where j1 = li//2, k1 = li%2 from sV
            for j2 in range(2):
                for k2 in range(3):
                    v2_val = V2[j2 * 3 + k2, r2]
                    if v2_val != 0:
                        V[(li // 2) * 2 * 6 + j2 * 6 + (li % 2) * 3 + k2, r] += val * v2_val
        
        # W[k1*1 + k2, r] = W1[k1, r1] * W2[k2, r2]
        # W has shape (p*n, rank) = (6*2, rank) = (12, rank)
        # W index = k*n + i = (k1*3+k2)*2 + (i1*1+i2) = (k1*3+k2)*2 + i1 + i2
        # Wait, n=2, so i = i1*1 + i2 where i1 in {0,1}, i2 in {0,1}
        # Actually for Kronecker: C is 2x6, C[i,k] where i in {0,1}, k in {0..5}
        # i = i1*1 + i2, k = k1*3 + k2
        # W index = k*n + i = (k1*3+k2)*2 + (i1+i2)
        
        for li, val in sW[r1]:
            i1 = li // 2
            k1 = li % 2
            for k2 in range(3):
                for i2 in range(2):
                    w2_val = W2[k2, r2]
                    if w2_val != 0:
                        W[(k1 * 3 + k2) * 2 + (i1 + i2), r] += val * w2_val
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
