import time
import sys

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=5
    
    # Total schoolbook rank
    rank_schoolbook = n * m * p
    
    # We will construct a list of (u_vec, v_vec, w_vec) tuples
    # u_vec is length n*m
    # v_vec is length m*p
    # w_vec is length p*n
    
    terms = []
    
    # Define Strassen terms for 2x2x2
    # Indices in 2x2: 0,1
    # U indices (A): 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices (B): 00, 01, 10, 11 -> 0, 1, 2, 3
    # W indices (C): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [1, 0, 1, 0], # M6 (A10 - A00) -> [ -1, 0, 1, 0 ]? 
        # Wait, M6 = (A10 - A00)(B00 + B01). 
        # A10 is index 2, A00 is index 0. So [-1, 0, 1, 0].
        [0, 1, 0, -1]  # M7 = (A01 - A11)(B10 + B11). A01 is 1, A11 is 3. [0, 1, 0, -1].
    ]
    
    strassen_v = [
        [1, 1, 0, 0], # M1 (B00+B11) -> B00 is 0, B11 is 3. [1, 0, 0, 1]? 
        # B indices: 00(0), 01(1), 10(2), 11(3).
        # M1: B00+B11 -> [1, 0, 0, 1]
        [1, 0, 0, 0], # M2 (B00)
        [0, 1, 0, -1],# M3 (B01-B11)
        [1, 0, 1, 0], # M4 (B10-B00) -> B10 is 2, B00 is 0. [-1, 0, 1, 0]? 
        # M4 = A11(B10-B00). V part is B10-B00.
        [0, 0, 0, 1], # M5 (B11)
        [1, 1, 0, 0], # M6 (B00+B01)
        [0, 0, 1, 1]  # M7 (B10+B11)
    ]
    
    # Correcting Strassen V and W based on standard formula:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U (A):
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, 1]
    # M3: [1, 0, 0, 0]
    # M4: [0, 0, 0, 1]
    # M5: [1, 1, 0, 0]
    # M6: [-1, 0, 1, 0]
    # M7: [0, 1, 0, -1]
    
    # V (B):
    # M1: [1, 0, 0, 1]
    # M2: [1, 0, 0, 0]
    # M3: [0, 1, 0, -1]
    # M4: [-1, 0, 1, 0]
    # M5: [0, 0, 0, 1]
    # M6: [1, 1, 0, 0]
    # M7: [0, 0, 1, 1]
    
    # W (C):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # W vectors for each M:
    # M1: [1, 0, 0, 1]
    # M2: [0, 0, 1, -1]
    # M3: [0, 1, 0, 1]
    # M4: [1, 0, 1, 0]
    # M5: [-1, 1, 0, 0]
    # M6: [0, 0, 0, 1]
    # M7: [1, 0, 0, 0]
    
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    def add_strassen_block(k_offset):
        """Add 7 Strassen terms for A[0:2,0:2] * B[0:2, k_offset:k_offset+2] -> C[0:2, k_offset:k_offset+2]"""
        for r in range(7):
            u_full = [0] * (n * m)
            v_full = [0] * (m * p)
            w_full = [0] * (p * n)
            
            # Map 2x2 indices to global
            # A indices: i in 0..1, j in 0..1
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_u_idx = i * m + j
                    u_full[global_u_idx] = strassen_u[r][local_idx]
                    
            # B indices: i (row) in 0..1, j (col) in 0..1
            # Global B index: row * p + col
            # Local B index: row * 2 + col
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    global_v_idx = i * p + (k_offset + j)
                    v_full[global_v_idx] = strassen_v[r][local_idx]
                    
            # C indices: i (row) in 0..1, j (col) in 0..1
            # Global C index: col * n + row
            # Local C index: col * 2 + row
            for i in range(2):
                for j in range(2):
                    local_idx = j * 2 + i
                    global_w_idx = (k_offset + j) * n + i
                    w_full[global_w_idx] = strassen_w[r][local_idx]
                    
            terms.append((u_full, v_full, w_full))

    # Add Strassen blocks for k=0 and k=2
    add_strassen_block(0)
    add_strassen_block(2)
    
    # Add remaining schoolbook terms
    # Skip terms where i in 0..1, j in 0..1, k in 0..1 (block 1)
    # Skip terms where i in 0..1, j in 0..1, k in 2..3 (block 2)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in replaced blocks
                if i < 2 and j < 2:
                    if k < 2:
                        continue
                    if 2 <= k < 4:
                        continue
                
                u_full = [0] * (n * m)
                v_full = [0] * (m * p)
                w_full = [0] * (p * n)
                
                u_full[i * m + j] = 1
                v_full[j * p + k] = 1
                w_full[k * n + i] = 1
                
                terms.append((u_full, v_full, w_full))
                
    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
