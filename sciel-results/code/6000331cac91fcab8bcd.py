import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank = 189
    
    # Initialize
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 factors
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    # r=1
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[2][1]=1; W2[3][1]=-1
    # r=2
    U2[0][2]=1
    V2[2][2]=1; V2[3][2]=-1
    W2[1][2]=1; W2[3][2]=1
    # r=3
    U2[3][3]=1
    V2[1][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[2][3]=1
    # r=4
    U2[0][4]=1; U2[2][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[1][4]=1
    # r=5
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1
    # r=6
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1
    
    # Schoolbook 3x3 factors
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
                
    # Combine
    for r3 in range(27):
        for r2 in range(7):
            r = r3 * 7 + r2
            for i3 in range(3):
                for j3 in range(3):
                    for k3 in range(3):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    i = i3 * 2 + i2
                                    j = j3 * 2 + j2
                                    k = k3 * 2 + k2
                                    
                                    u_val = U3[i3*3+j3][r3] * U2[i2*2+j2][r2]
                                    if u_val != 0:
                                        U[i*6+j][r] += u_val
                                        
                                    v_val = V3[j3*3+k3][r3] * V2[j2*2+k2][r2]
                                    if v_val != 0:
                                        V[j*6+k][r] += v_val
                                        
                                    w_val = W3[k3*3+i3][r3] * W2[k2*2+i2][r2]
                                    if w_val != 0:
                                        W[k*6+i][r] += w_val
                                                    
    return {"U": U, "V": V, "W": W}
