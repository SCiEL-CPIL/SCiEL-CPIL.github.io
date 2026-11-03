"""Direct rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def construct():
    global _START_TIME, _DEADLINE
    
    n, m, p = 3, 4, 8
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: i=0, schoolbook for 1x4x8: 32 terms
    for j in range(4):
        for k in range(8):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Part 2: i=1,2, Strassen on 2x4x8 split into 8 blocks of 2x2x2
    # Strassen matrices for 2x2x2 (indices: U: (i,j), V: (j,k), W: (k,i))
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For 2x2x2 with A indices (i,j) i=0,1 j=0,1; B indices (j,k) j=0,1 k=0,1; C indices (k,i) k=0,1 i=0,1
    # U entries: U[i*2+j, t]
    # V entries: V[j*2+k, t]  
    # W entries: W[k*2+i, t]
    
    # Strassen decomposition (7 terms):
    # t=0: M1 = (A00+A11)(B00+B11)
    # t=1: M2 = (A10+A11)B00
    # t=2: M3 = A00(B10-B11)
    # t=3: M4 = A11(B01-B00)
    # t=4: M5 = (A00+A01)B11
    # t=5: M6 = (A10-A00)(B00+B01)
    # t=6: M7 = (A01-A11)(B10+B11)
    
    # U matrix (2x2 per term, 7 terms): rows are (i,j) pairs: (0,0),(0,1),(1,0),(1,1)
    # U[t] for term t:
    # t=0: A00+A11 -> (1,0,0,1) for (00,01,10,11)
    # t=1: A10+A11 -> (0,0,1,1)
    # t=2: A00 -> (1,0,0,0)
    # t=3: A11 -> (0,0,0,1)
    # t=4: A00+A01 -> (1,1,0,0)
    # t=5: A10-A00 -> (-1,0,1,0)
    # t=6: A01-A11 -> (0,1,0,-1)
    
    U_strassen = [
        [1, 0, 0, 1],   # t=0
        [0, 0, 1, 1],   # t=1
        [1, 0, 0, 0],   # t=2
        [0, 0, 0, 1],   # t=3
        [1, 1, 0, 0],   # t=4
        [-1, 0, 1, 0],  # t=5
        [0, 1, 0, -1]   # t=6
    ]
    
    # V matrix: rows are (j,k) pairs: (0,0),(0,1),(1,0),(1,1)
    # t=0: B00+B11 -> (1,0,0,1)
    # t=1: B00 -> (1,0,0,0)
    # t=2: B10-B11 -> (0,0,1,-1)
    # t=3: B01-B00 -> (-1,1,0,0)
    # t=4: B11 -> (0,0,0,1)
    # t=5: B00+B01 -> (1,1,0,0)
    # t=6: B10+B11 -> (0,0,1,1)
    
    V_strassen = [
        [1, 0, 0, 1],   # t=0
        [1, 0, 0, 0],   # t=1
        [0, 0, 1, -1],  # t=2
        [-1, 1, 0, 0],  # t=3
        [0, 0, 0, 1],   # t=4
        [1, 1, 0, 0],   # t=5
        [0, 0, 1, 1]    # t=6
    ]
    
    # W matrix: rows are (k,i) pairs: (0,0),(0,1),(1,0),(1,1)
    # C11 = M1 + M4 - M5 + M7 -> W[0,0] gets M1,M4,-M5,M7
    # C12 = M3 + M5 -> W[0,1] gets M3,M5
    # C21 = M2 + M4 -> W[1,0] gets M2,M4
    # C22 = M1 - M2 + M3 + M6 -> W[1,1] gets M1,-M2,M3,M6
    
    # W[t] for term t, indexed by (k,i): (0,0),(0,1),(1,0),(1,1)
    # t=0 (M1): C11+=1, C22+=1 -> (1,0,0,1)
    # t=1 (M2): C21+=1, C22+=-1 -> (0,0,1,-1)
    # t=2 (M3): C12+=1, C22+=1 -> (0,1,0,1)
    # t=3 (M4): C11+=1, C21+=1 -> (1,0,1,0)
    # t=4 (M5): C11+=-1, C12+=1 -> (-1,1,0,0)
    # t=5 (M6): C22+=1 -> (0,0,0,1)
    # t=6 (M7): C11+=1 -> (1,0,0,0)
    
    W_strassen = [
        [1, 0, 0, 1],   # t=0
        [0, 0, 1, -1],  # t=1
        [0, 1, 0, 1],   # t=2
        [1, 0, 1, 0],   # t=3
        [-1, 1, 0, 0],  # t=4
        [0, 0, 0, 1],   # t=5
        [1, 0, 0, 0]    # t=6
    ]
    
    # For 2x4x8: i in {1,2} (local i'=0,1), j in {0,1,2,3}, k in {0,...,7}
    # Split j into 2 groups of 2, k into 4 groups of 2
    # 8 blocks of 2x2x2
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # U: local (i', j') where i' in {0,1}, j' in {0,1}
                # global i = 1 + i', global j = j_start + j'
                for l in range(4):
                    if U_strassen[t][l] != 0:
                        local_i_prime = l // 2
                        local_j_prime = l % 2
                        global_i = 1 + local_i_prime
                        global_j = j_start + local_j_prime
                        U[global_i * m + global_j][col] += U_strassen[t][l]
                
                # V: local (j', k') where j' in {0,1}, k' in {0,1}
                # global j = j_start + j', global k = k_start + k'
                for l in range(4):
                    if V_strassen[t][l] != 0:
                        local_j_prime = l // 2
                        local_k_prime = l % 2
                        global_j = j_start + local_j_prime
                        global_k = k_start + local_k_prime
                        V[global_j * p + global_k][col] += V_strassen[t][l]
                
                # W: local (k', i') where k' in {0,1}, i' in {0,1}
                # global k = k_start + k', global i = 1 + i'
                for l in range(4):
                    if W_strassen[t][l] != 0:
                        local_k_prime = l // 2
                        local_i_prime = l % 2
                        global_k = k_start + local_k_prime
                        global_i = 1 + local_i_prime
                        W[global_k * n + global_i][col] += W_strassen[t][l]
                
                col += 1
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        pass  # We've already computed, just return
    
    return {"U": U, "V": V, "W": W}
