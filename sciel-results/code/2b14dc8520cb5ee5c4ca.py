import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the 44-rank decomposition using numpy for efficiency
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8) -> 2x8
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0, col] = 1
            V[0 * 8 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8) -> 2x8
    # Strassen decomposition for each 2x2 block of B2
    for block in range(4):
        # 1. (a00+a11)(b00+b11)
        U[1, col] = 1; U[5, col] = 1
        V[1*8 + block*2, col] = 1; V[2*8 + block*2+1, col] = 1
        W[(block*2)*2 + 0, col] = 1; W[(block*2+1)*2 + 1, col] = 1
        col += 1
        
        # 2. (a10+a11)b00
        U[4, col] = 1; U[5, col] = 1
        V[1*8 + block*2, col] = 1
        W[(block*2)*2 + 1, col] = 1; W[(block*2+1)*2 + 1, col] = -1
        col += 1
        
        # 3. a00(b01-b11)
        U[1, col] = 1
        V[1*8 + block*2+1, col] = 1; V[2*8 + block*2+1, col] = -1
        W[(block*2+1)*2 + 0, col] = 1; W[(block*2+1)*2 + 1, col] = 1
        col += 1
        
        # 4. a11(b10-b00)
        U[5, col] = 1
        V[2*8 + block*2, col] = 1; V[1*8 + block*2, col] = -1
        W[(block*2)*2 + 0, col] = 1; W[(block*2)*2 + 1, col] = 1
        col += 1
        
        # 5. (a00+a01)b11
        U[1, col] = 1; U[2, col] = 1
        V[2*8 + block*2+1, col] = 1
        W[(block*2)*2 + 0, col] = -1; W[(block*2+1)*2 + 0, col] = 1
        col += 1
        
        # 6. (a10-a00)(b00+b01)
        U[4, col] = 1; U[1, col] = -1
        V[1*8 + block*2, col] = 1; V[1*8 + block*2+1, col] = 1
        W[(block*2+1)*2 + 1, col] = 1
        col += 1
        
        # 7. (a01-a11)(b10+b11)
        U[2, col] = 1; U[5, col] = -1
        V[2*8 + block*2, col] = 1; V[2*8 + block*2+1, col] = 1
        W[(block*2)*2 + 0, col] = 1
        col += 1
    
    # Convert to list format with integer/half-integer entries
    def to_list(mat):
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if abs(val - round(val)) < 1e-10:
                    row.append(int(round(val)))
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
