import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 4, 4, 7
    
    # Strassen decomposition for 4x4x4
    terms = []
    
    def lf(coeff, r, c):
        return [(coeff, r, c)]
        
    X1 = [[lf(1, 0, 0) + lf(1, 2, 2), lf(1, 0, 1) + lf(1, 2, 3)],
          [lf(1, 1, 0) + lf(1, 3, 2), lf(1, 1, 1) + lf(1, 3, 3)]]
    Y1 = [[lf(1, 0, 0) + lf(1, 2, 2), lf(1, 0, 1) + lf(1, 2, 3)],
          [lf(1, 1, 0) + lf(1, 3, 2), lf(1, 1, 1) + lf(1, 3, 3)]]
    terms.append((X1, Y1))
    
    X2 = [[lf(1, 2, 0) + lf(1, 3, 2), lf(1, 2, 1) + lf(1, 3, 3)],
          [lf(1, 3, 0) + lf(1, 3, 2), lf(1, 3, 1) + lf(1, 3, 3)]]
    Y2 = [[lf(1, 0, 0), lf(1, 0, 1)],
          [lf(1, 1, 0), lf(1, 1, 1)]]
    terms.append((X2, Y2))
    
    X3 = [[lf(1, 0, 0), lf(1, 0, 1)],
          [lf(1, 1, 0), lf(1, 1, 1)]]
    Y3 = [[lf(1, 0, 2) - lf(1, 2, 2), lf(1, 0, 3) - lf(1, 2, 3)],
          [lf(1, 1, 2) - lf(1, 3, 2), lf(1, 1, 3) - lf(1, 3, 3)]]
    terms.append((X3, Y3))
    
    X4 = [[lf(1, 2, 2), lf(1, 2, 3)],
          [lf(1, 3, 2), lf(1, 3, 3)]]
    Y4 = [[lf(1, 2, 0) - lf(1, 0, 0), lf(1, 2, 1) - lf(1, 0, 1)],
          [lf(1, 3, 0) - lf(1, 1, 0), lf(1, 3, 1) - lf(1, 1, 1)]]
    terms.append((X4, Y4))
    
    X5 = [[lf(1, 0, 0) + lf(1, 0, 2), lf(1, 0, 1) + lf(1, 0, 3)],
          [lf(1, 1, 0) + lf(1, 1, 2), lf(1, 1, 1) + lf(1, 1, 3)]]
    Y5 = [[lf(1, 2, 2), lf(1, 2, 3)],
          [lf(1, 3, 2), lf(1, 3, 3)]]
    terms.append((X5, Y5))
    
    X6 = [[lf(1, 2, 0) - lf(1, 0, 0), lf(1, 2, 1) - lf(1, 0, 1)],
          [lf(1, 3, 0) - lf(1, 1, 0), lf(1, 3, 1) - lf(1, 1, 1)]]
    Y6 = [[lf(1, 0, 0) + lf(1, 0, 2), lf(1, 0, 1) + lf(1, 0, 3)],
          [lf(1, 1, 0) + lf(1, 1, 2), lf(1, 1, 1) + lf(1, 1, 3)]]
    terms.append((X6, Y6))
    
    X7 = [[lf(1, 0, 2) - lf(1, 2, 2), lf(1, 0, 3) - lf(1, 2, 3)],
          [lf(1, 1, 2) - lf(1, 3, 2), lf(1, 1, 3) - lf(1, 3, 3)]]
    Y7 = [[lf(1, 2, 0) + lf(1, 2, 2), lf(1, 2, 1) + lf(1, 2, 3)],
          [lf(1, 3, 0) + lf(1, 3, 2), lf(1, 3, 1) + lf(1, 3, 3)]]
    terms.append((X7, Y7))
    
    U_list = []
    V_list = []
    W_list = []
    
    for m_idx in range(7):
        X, Y = terms[m_idx]
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    U = [0] * 16
                    V = [0] * 16
                    W = [0] * 28
                    
                    for coeff, r, col in X[a][b]:
                        U[r * 4 + col] += coeff
                    for coeff, r, col in Y[b][c]:
                        V[r * 4 + col] += coeff
                    
                    if m_idx == 0:
                        W[c * 4 + a] += 1
                        W[(c+2) * 4 + (a+2)] += 1
                    elif m_idx == 1:
                        W[c * 4 + (a+2)] += 1
                        W[(c+2) * 4 + (a+2)] -= 1
                    elif m_idx == 2:
                        W[(c+2) * 4 + a] += 1
                        W[(c+2) * 4 + (a+2)] += 1
                    elif m_idx == 3:
                        W[c * 4 + a] += 1
                        W[c * 4 + (a+2)] += 1
                    elif m_idx == 4:
                        W[c * 4 + a] -= 1
                        W[(c+2) * 4 + a] += 1
                    elif m_idx == 5:
                        W[(c+2) * 4 + (a+2)] += 1
                    elif m_idx == 6:
                        W[c * 4 + a] += 1
                        
                    U_list.append(U)
                    V_list.append(V)
                    W_list.append(W)
                    
    rank = 56
    U_final = [[0] * rank for _ in range(16)]
    V_final = [[0] * rank for _ in range(16)]
    W_final = [[0] * rank for _ in range(28)]
    
    for i in range(28):
        for r in range(16):
            U_final[r][i] = U_list[i][r]
            V_final[r][i] = V_list[i][r]
            U_final[r][i+28] = U_list[i][r]
            V_final[r][i+28] = V_list[i][r]
        for r in range(28):
            W_final[r][i] = W_list[i][r]
        for k_old in range(3):
            for i_row in range(4):
                W_final[(k_old+4)*4 + i_row][i+28] = W_list[i][k_old*4 + i_row]
                
    return {"U": U_final, "V": V_final, "W": W_final}
