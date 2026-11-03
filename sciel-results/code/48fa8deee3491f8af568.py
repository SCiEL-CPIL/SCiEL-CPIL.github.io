import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 125.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook(n, m, p):
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x2_2x2():
    """
    Returns U, V, W for 2x2 x 2x2 multiplication with rank 7.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U: i*m+j -> i in 0..1, j in 0..1.
    V: j*p+k -> j in 0..1, k in 0..1.
    W: k*n+i -> k in 0..1, i in 0..1.
    """
    n, m, p = 2, 2, 2
    r = 7
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Mapping to tensor factors
    # U[i*m+j, c]
    # V[j*p+k, c]
    # W[k*n+i, c]
    
    # c=0: M1
    # A00 (0,0), A11 (1,1)
    U[0][0] = 1; U[3][0] = 1
    # B00 (0,0), B11 (1,1)
    V[0][0] = 1; V[3][0] = 1
    # C00 (0,0), C11 (1,1)
    W[0][0] = 1; W[3][0] = 1
    
    # c=1: M2
    # A10 (1,0), A11 (1,1)
    U[2][1] = 1; U[3][1] = 1
    # B00 (0,0)
    V[0][1] = 1
    # C10 (0,1), C11 (1,1) -- Wait, C10 is i=1, k=0. W[k*n+i] = W[0*2+1] = W[1].
    # C10 corresponds to W[1]. C11 corresponds to W[3].
    # M2 contributes to C10 and C11?
    # C10 = M2 + M4. So M2 is in C10.
    # C11 = M1 - M2 + M3 + M6. So M2 is in C11.
    # So W[1] and W[3] get M2.
    W[1][1] = 1; W[3][1] = 1
    
    # c=2: M3
    # A00 (0,0)
    U[0][2] = 1
    # B11 (1,1), B00 (0,0)
    V[3][2] = 1; V[0][2] = -1
    # C01 (0,1) -> W[1*2+0] = W[2]. C11 (1,1) -> W[3].
    # C01 = M3 + M5.
    # C11 = ... + M3 ...
    W[2][2] = 1; W[3][2] = 1
    
    # c=3: M4
    # A11 (1,1)
    U[3][3] = 1
    # B00 (0,0), B10 (1,0)
    V[0][3] = 1; V[2][3] = -1
    # C00 (0,0) -> W[0]. C10 (0,1) -> W[1].
    # C00 = ... + M4 ...
    # C10 = M2 + M4.
    W[0][3] = 1; W[1][3] = 1
    
    # c=4: M5
    # A00 (0,0), A01 (0,1)
    U[0][4] = 1; U[1][4] = 1
    # B11 (1,1)
    V[3][4] = 1
    # C00 (0,0) -> W[0]. C01 (0,1) -> W[2].
    # C00 = ... - M5 ...
    # C01 = M3 + M5.
    W[0][4] = -1; W[2][4] = 1
    
    # c=5: M6
    # A10 (1,0), A00 (0,0)
    U[2][5] = 1; U[0][5] = -1
    # B00 (0,0), B01 (0,1)
    V[0][5] = 1; V[1][5] = 1
    # C11 (1,1) -> W[3].
    # C11 = ... + M6.
    W[3][5] = 1
    
    # c=6: M7
    # A01 (0,1), A11 (1,1)
    U[1][6] = 1; U[3][6] = -1
    # B10 (1,0), B11 (1,1)
    V[2][6] = 1; V[3][6] = 1
    # C00 (0,0) -> W[0].
    # C00 = ... + M7.
    W[0][6] = 1
    
    return U, V, W

def _build_3x3x2():
    """
    Builds decomposition for 3x3 x 3x2 with rank 17.
    """
    n, m, p = 3, 3, 2
    r = 17
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Part 1: Strassen for A[0:2,0:2] x B[0:2,0:2]
    U_s, V_s, W_s = _strassen_2x2_2x2_2x2()
    for i in range(4):
        for c in range(7):
            U[i][c] += U_s[i][c]
            V[i][c] += V_s[i][c]
            # W index: k*n+i. In 3x3, k in 0..1, i in 0..1.
            # W_s index is k*2+i.
            # We need to map to W[k*3+i].
            # W_s has shape (4, 7). Indices 0..3 correspond to k*2+i.
            # k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 2. k=1, i=1 -> 3.
            # In 3x3, k=0, i=0 -> 0. k=0, i=1 -> 1. k=1, i=0 -> 3. k=1, i=1 -> 4.
            # So we need to map W_s indices to W indices.
            # W_s[0] (k=0,i=0) -> W[0]
            # W_s[1] (k=0,i=1) -> W[1]
            # W_s[2] (k=1,i=0) -> W[3]
            # W_s[3] (k=1,i=1) -> W[4]
            
    # Mapping W_s to W
    w_map = [0, 1, 3, 4]
    for ws_idx in range(4):
        w_idx = w_map[ws_idx]
        for c in range(7):
            W[w_idx][c] += W_s[ws_idx][c]
            
    # Part 2: Schoolbook for A[0:2,2] x B[2,0:2]
    # A[0,2], A[1,2]
    # B[2,0], B[2,1]
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    
    c_offset = 7
    # C[0,0]: U[0*3+2]=U[2], V[2*2+0]=V[4], W[0*3+0]=W[0]
    U[2][c_offset] = 1; V[4][c_offset] = 1; W[0][c_offset] = 1; c_offset += 1
    # C[0,1]: U[2], V[2*2+1]=V[5], W[1*3+0]=W[3]
    U[2][c_offset] = 1; V[5][c_offset] = 1; W[3][c_offset] = 1; c_offset += 1
    # C[1,0]: U[1*3+2]=U[5], V[4], W[0]
    U[5][c_offset] = 1; V[4][c_offset] = 1; W[0][c_offset] = 1; c_offset += 1
    # C[1,1]: U[5], V[5], W[3]
    U[5][c_offset] = 1; V[5][c_offset] = 1; W[3][c_offset] = 1; c_offset += 1
    
    # Part 3: Schoolbook for A[2,0:2] x B[0:2,0:2]
    # A[2,0], A[2,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[2,0] += A[2,0]B[0,0] + A[2,1]B[1,0]
    # C[2,1] += A[2,0]B[0,1] + A[2,1]B[1,1]
    
    # C[2,0]: U[2*3+0]=U[6], V[0*2+0]=V[0], W[0*3+2]=W[2]
    U[6][c_offset] = 1; V[0][c_offset] = 1; W[2][c_offset] = 1; c_offset += 1
    # C[2,0]: U[6], V[1*2+0]=V[2], W[2]
    U[6][c_offset] = 1; V[2][c_offset] = 1; W[2][c_offset] = 1; c_offset += 1
    # C[2,1]: U[6], V[0*2+1]=V[1], W[1*3+2]=W[5]
    U[6][c_offset] = 1; V[1][c_offset] = 1; W[5][c_offset] = 1; c_offset += 1
    # C[2,1]: U[6], V[1*2+1]=V[3], W[5]
    U[6][c_offset] = 1; V[3][c_offset] = 1; W[5][c_offset] = 1; c_offset += 1
    
    # Part 4: Schoolbook for A[2,2] x B[2,0:2]
    # A[2,2]
    # B[2,0], B[2,1]
    # C[2,0] += A[2,2]B[2,0]
    # C[2,1] += A[2,2]B[2,1]
    
    # C[2,0]: U[2*3+2]=U[8], V[4], W[2]
    U[8][c_offset] = 1; V[4][c_offset] = 1; W[2][c_offset] = 1; c_offset += 1
    # C[2,1]: U[8], V[5], W[5]
    U[8][c_offset] = 1; V[5][c_offset] = 1; W[5][c_offset] = 1; c_offset += 1
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    # Split B into two 3x4 blocks
    # Each 3x4 block is split into two 3x2 blocks
    # Total 4 blocks of 3x3x2
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for block in range(4):
        U_b, V_b, W_b = _build_3x3x2()
        U_blocks.append(U_b)
        V_blocks.append(V_b)
        W_blocks.append(W_b)
        
    # Concatenate
    # U shape (9, 17*4)
    # V shape (6, 17*4) -- Wait, V is (m*p, rank). m=3, p=8. So V is (24, rank).
    # In _build_3x3x2, V is (3*2, 17) = (6, 17).
    # We need to map the 3x2 blocks to the 3x8 B.
    # Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5. Block 3: cols 6,7.
    
    r_total = 17 * 4
    U = [[0]*r_total for _ in range(n*m)]
    V = [[0]*r_total for _ in range(m*p)]
    W = [[0]*r_total for _ in range(p*n)]
    
    for b in range(4):
        U_b, V_b, W_b = U_blocks[b], V_blocks[b], W_blocks[b]
        offset = b * 17
        
        # U is same for all blocks (A is same)
        for i in range(n*m):
            for c in range(17):
                U[i][offset + c] += U_b[i][c]
                
        # V mapping
        # Block b covers columns 2*b and 2*b+1 of B.
        # V_b indices: j*p+k. j in 0..2, k in 0..1.
        # V index: j*2+k.
        # Target V index: j*8 + (2*b + k).
        for j in range(m):
            for k in range(2):
                vb_idx = j*2 + k
                v_target_idx = j*8 + (2*b + k)
                for c in range(17):
                    V[v_target_idx][offset + c] += V_b[vb_idx][c]
                    
        # W mapping
        # W_b indices: k*n+i. k in 0..1, i in 0..2.
        # W index: k*3+i.
        # Target W index: (2*b+k)*3 + i.
        for k in range(2):
            for i in range(n):
                wb_idx = k*3 + i
                w_target_idx = (2*b + k)*3 + i
                for c in range(17):
                    W[w_target_idx][offset + c] += W_b[wb_idx][c]
                    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    # print(_verify(res["U"], res["V"], res["W"], 3, 3, 8))
