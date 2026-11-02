"""Rank-minimal exact tensor decomposition for 2x3 x 3x8 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 8]

def _strassen_block():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    U: i*2+j
    V: j*2+k
    W: k*2+i
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _outer_block():
    """
    Returns U, V, W for 2x1x2 Outer Product decomposition.
    U: 2x4, V: 2x4, W: 4x4
    Indices:
    U: i
    V: k
    W: k*2+i
    """
    U = [[0]*4 for _ in range(2)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(4)]
    
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
    U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
    U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
    
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 110
    
    n, m, p = DIMENSIONS
    R = 44
    
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    col = 0
    
    # 4 Strassen blocks (j in {0,1}, k in {0,1}, {2,3}, {4,5}, {6,7})
    for k_off in range(0, 8, 2):
        j_off = 0
        u_b, v_b, w_b = _strassen_block()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + (j_off + j)
                    U[u_idx][col + r] = u_b[i*2+j][r]
            
            for j in range(2):
                for k in range(2):
                    v_idx = (j_off + j) * p + (k_off + k)
                    V[v_idx][col + r] = v_b[j*2+k][r]
            
            for k in range(2):
                for i in range(2):
                    w_idx = (k_off + k) * n + i
                    W[w_idx][col + r] = w_b[k*2+i][r]
        
        col += 7
    
    # 4 Outer blocks (j=2, k in {0,1}, {2,3}, {4,5}, {6,7})
    for k_off in range(0, 8, 2):
        j_off = 2
        u_b, v_b, w_b = _outer_block()
        for r in range(4):
            for i in range(2):
                u_idx = i * m + (j_off + 0)
                U[u_idx][col + r] = u_b[i][r]
            
            for k in range(2):
                v_idx = (j_off + 0) * p + (k_off + k)
                V[v_idx][col + r] = v_b[k][r]
            
            for k in range(2):
                for i in range(2):
                    w_idx = (k_off + k) * n + i
                    W[w_idx][col + r] = w_b[k*2+i][r]
        
        col += 4
    
    return {"U": U, "V": V, "W": W}
