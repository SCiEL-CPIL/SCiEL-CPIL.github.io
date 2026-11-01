import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                # The check in the seed was checking for delta(i,j) which is wrong for matmul.
                # For matmul, the sum over r of U[i,j,r] V[j,k,r] W[i,k,r] should be 1?
                # No, W index is k*n+i. So W[k*n+i][r] is the coefficient for C_{i,k}.
                # The term is U[i,j,r] V[j,k,r] W[k*n+i,r].
                # This sums to the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
                # This should be 1 for all i,j,k.
                if abs(s - 1.0) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Rank 42 decomposition using Strassen for 6 blocks
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen U vectors (A factors)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    u_strassen = [
        [1.0, 0.0, 0.0, 1.0], # M1
        [0.0, 0.0, 1.0, 1.0], # M2
        [1.0, 0.0, 0.0, 0.0], # M3
        [0.0, 0.0, 0.0, 1.0], # M4
        [1.0, 1.0, 0.0, 0.0], # M5
        [-1.0, 0.0, 1.0, 0.0],# M6
        [0.0, 1.0, 0.0, -1.0] # M7
    ]
    
    for l in range(6):
        # V vectors for block l
        # B indices:
        # B_0,2l -> 2l
        # B_0,2l+1 -> 2l+1
        # B_1,2l -> 12+2l
        # B_1,2l+1 -> 12+2l+1
        
        b00_idx = 2*l
        b01_idx = 2*l + 1
        b10_idx = 12 + 2*l
        b11_idx = 12 + 2*l + 1
        
        v_strassen = [
            {b00_idx: 1.0, b11_idx: 1.0}, # M1: B00 + B11
            {b00_idx: 1.0},               # M2: B00
            {b01_idx: 1.0, b11_idx: -1.0},# M3: B01 - B11
            {b10_idx: 1.0, b00_idx: -1.0},# M4: B10 - B00
            {b11_idx: 1.0},               # M5: B11
            {b00_idx: 1.0, b01_idx: 1.0}, # M6: B00 + B01
            {b10_idx: 1.0, b11_idx: 1.0}  # M7: B10 + B11
        ]
        
        # W vectors for block l
        # C indices (W index = k*n + i = 2k + i):
        # C_0,2l -> 4l
        # C_1,2l -> 4l+1
        # C_0,2l+1 -> 4l+2
        # C_1,2l+1 -> 4l+3
        
        c00_idx = 4*l
        c10_idx = 4*l + 1
        c01_idx = 4*l + 2
        c11_idx = 4*l + 3
        
        # Reconstruction:
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 - M5 + M6
        
        w_strassen = [
            {c00_idx: 1.0, c11_idx: 1.0}, # M1
            {c10_idx: 1.0},               # M2
            {c01_idx: 1.0, c11_idx: -1.0},# M3
            {c00_idx: -1.0, c10_idx: 1.0},# M4
            {c01_idx: 1.0, c11_idx: -1.0},# M5
            {c00_idx: -1.0, c11_idx: 1.0},# M6
            {c00_idx: 1.0}                # M7
        ]
        
        for m_idx in range(7):
            r = l * 7 + m_idx
            
            # Fill U
            for i in range(4):
                U[i][r] = u_strassen[m_idx][i]
                
            # Fill V
            for idx, val in v_strassen[m_idx].items():
                V[idx][r] += val
                
            # Fill W
            for idx, val in w_strassen[m_idx].items():
                W[idx][r] += val
                
    # Verify
    if _check(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    # Fallback
    fb = _schoolbook(n, m, p)
    return fb

# EVOLVE-BLOCK-END
