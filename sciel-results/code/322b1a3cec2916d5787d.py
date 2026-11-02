import numpy as np
import time

DIMENSIONS = [2, 2, 14]
_START_TIME = time.time()
_TIMEOUT = 120.0

def _time_remaining():
    return _TIMEOUT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    strassen_terms = np.array([
        [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1],
        [1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1],
        [0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0],
        [-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0]
    ], dtype=float)
    
    col = 0
    for b in range(num_blocks):
        if _time_remaining() < 5.0:
            break
        v_map = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for t in range(7):
            u_coeffs = strassen_terms[t, 0:4]
            v_coeffs = strassen_terms[t, 4:8]
            w_coeffs = strassen_terms[t, 8:12]
            
            for i in range(4):
                if u_coeffs[i] != 0:
                    U[i, col] = u_coeffs[i]
                if v_coeffs[i] != 0:
                    V[v_map[i], col] = v_coeffs[i]
                if w_coeffs[i] != 0:
                    W[w_map[i], col] = w_coeffs[i]
            
            col += 1
            
    # Convert to lists of ints if possible
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    for i in range(len(U_out)):
        for j in range(len(U_out[i])):
            if U_out[i][j] == int(U_out[i][j]):
                U_out[i][j] = int(U_out[i][j])
                
    for i in range(len(V_out)):
        for j in range(len(V_out[i])):
            if V_out[i][j] == int(V_out[i][j]):
                V_out[i][j] = int(V_out[i][j])
                
    for i in range(len(W_out)):
        for j in range(len(W_out[i])):
            if W_out[i][j] == int(W_out[i][j]):
                W_out[i][j] = int(W_out[i][j])

    return {"U": U_out, "V": V_out, "W": W_out}
