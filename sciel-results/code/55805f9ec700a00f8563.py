import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def strassen_222():
    # Returns U, V, W for 2x2x2
    # U is 4x7, V is 4x7, W is 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Indices:
    # U: 00, 01, 10, 11
    # V: 00, 01, 10, 11
    # W: 00, 01, 10, 11 (k*n+i)
    
    # c0 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # c1 = (a10+a11)(b00)
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = 1; W[3][1] = 1
    
    # c2 = (a00)(b10+b11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = 1
    W[0][2] = 1; W[3][2] = 1
    
    # c3 = (a11)(b01-b11)
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[3][3] = 1
    
    # c4 = (a00+a01)(b11)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[3][4] = 1
    
    # c5 = (a10-a00)(b00+b01)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[0][5] = 1; W[3][5] = 1
    
    # c6 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # Split m into 2+1
    # Block 1: 2x2x7
    # Block 2: 2x1x7
    
    # Block 1: 2x2x7
    # Split p into 2+2+2+1
    # B1.1: 2x2x2 (Strassen)
    # B1.2: 2x2x2 (Strassen)
    # B1.3: 2x2x2 (Strassen)
    # B1.4: 2x2x1 (Schoolbook)
    
    # Block 2: 2x1x7
    # Schoolbook rank 7? No, rank 7.
    # T_{i,2,k} = 1.
    # U_{i,2,r} V_{2,k,r} W_{k,i,r} = 1.
    # For each k, we can have a rank 1 term?
    # If we use 1 term for all k?
    # U_{i,2,r}=1, V_{2,k,r}=1, W_{k,i,r}=1.
    # Sum = 1.
    # So rank 1?
    # Let's check.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # For j=2, C_{ik} += A_{i2} B_{2k}.
    # Decomposition:
    # C_{ik} = sum_r W_{ki,r} (sum_j A_{ij} U_{ij,r}) (sum_j B_{jk} V_{jk,r}).
    # If we set r=0:
    # U_{i2,0} = 1 for all i.
    # V_{2k,0} = 1 for all k.
    # W_{ki,0} = 1 for all k,i.
    # Then term is 1 * A_{i2} * B_{2k} * 1 = A_{i2} B_{2k}.
    # This is exactly the contribution of j=2.
    # So rank 1 is sufficient for the 2x1x7 block?
    # Wait, U has shape 6x1. V has shape 21x1. W has shape 14x1.
    # U[i*3+2, 0] = 1 for i=0,1.
    # V[2*7+k, 0] = 1 for k=0..6.
    # W[k*2+i, 0] = 1 for k=0..6, i=0,1.
    # This works. Rank 1.
    
    # So Block 2 rank is 1.
    
    # Total Rank = 25 (from Block 1) + 1 (from Block 2) = 26.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: 2x2x7
    # Sub-blocks for p=0,1; p=2,3; p=4,5; p=6
    
    # 1.1: 2x2x2 (Strassen) for k=0,1
    U_s, V_s, W_s = strassen_222()
    U_block = [[0]*7 for _ in range(6)]
    V_block = [[0]*7 for _ in range(21)]
    W_block = [[0]*7 for _ in range(14)]
    
    for r in range(4):
        i = r // 2
        j = r % 2
        U_block[i*3 + j] = U_s[r]
        
    for r in range(4):
        j = r // 2
        k = r % 2
        V_block[j*7 + k] = V_s[r]
        
    for r in range(4):
        k = r // 2
        i = r % 2
        W_block[k*2 + i] = W_s[r]
        
    U_list.append(U_block)
    V_list.append(V_block)
    W_list.append(W_block)
    
    # 1.2: 2x2x2 (Strassen) for k=2,3
    U_s, V_s, W_s = strassen_222()
    U_block = [[0]*7 for _ in range(6)]
    V_block = [[0]*7 for _ in range(21)]
    W_block = [[0]*7 for _ in range(14)]
    
    for r in range(4):
        i = r // 2
        j = r % 2
        U_block[i*3 + j] = U_s[r]
        
    for r in range(4):
        j = r // 2
        k = r % 2
        V_block[j*7 + (2+k)] = V_s[r]
        
    for r in range(4):
        k = r // 2
        i = r % 2
        W_block[(2+k)*2 + i] = W_s[r]
        
    U_list.append(U_block)
    V_list.append(V_block)
    W_list.append(W_block)
    
    # 1.3: 2x2x2 (Strassen) for k=4,5
    U_s, V_s, W_s = strassen_222()
    U_block = [[0]*7 for _ in range(6)]
    V_block = [[0]*7 for _ in range(21)]
    W_block = [[0]*7 for _ in range(14)]
    
    for r in range(4):
        i = r // 2
        j = r % 2
        U_block[i*3 + j] = U_s[r]
        
    for r in range(4):
        j = r // 2
        k = r % 2
        V_block[j*7 + (4+k)] = V_s[r]
        
    for r in range(4):
        k = r // 2
        i = r % 2
        W_block[(4+k)*2 + i] = W_s[r]
        
    U_list.append(U_block)
    V_list.append(V_block)
    W_list.append(W_block)
    
    # 1.4: 2x2x1 (Schoolbook) for k=6
    U_block = [[0]*4 for _ in range(6)]
    V_block = [[0]*4 for _ in range(21)]
    W_block = [[0]*4 for _ in range(14)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            U_block[i*3+j][col] = 1
            V_block[j*7+6][col] = 1
            W_block[6*2+i][col] = 1
            col += 1
            
    U_list.append(U_block)
    V_list.append(V_block)
    W_list.append(W_block)
    
    # Block 2: 2x1x7 (Rank 1)
    U_block = [[0]*1 for _ in range(6)]
    V_block = [[0]*1 for _ in range(21)]
    W_block = [[0]*1 for _ in range(14)]
    
    for i in range(2):
        U_block[i*3+2][0] = 1
        
    for k in range(7):
        V_block[2*7+k][0] = 1
        
    for k in range(7):
        for i in range(2):
            W_block[k*2+i][0] = 1
            
    U_list.append(U_block)
    V_list.append(V_block)
    W_list.append(W_block)
    
    # Combine
    total_rank = sum(len(block[0]) for block in U_list)
    U = [[0]*total_rank for _ in range(6)]
    V = [[0]*total_rank for _ in range(21)]
    W = [[0]*total_rank for _ in range(14)]
    
    offset = 0
    for i, block in enumerate(U_list):
        for r in range(6):
            for c in range(len(block[0])):
                U[r][offset+c] = block[r][c]
                V[r][offset+c] = V_list[i][r][c]
                W[r][offset+c] = W_list[i][r][c]
        offset += len(block[0])
        
    return {"U": U, "V": V, "W": W}

