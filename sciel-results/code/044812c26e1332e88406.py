import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # j_offset: starting j index (0 or 1? No, j is 0,1 for the 2x2 part)
    # k_offset: starting k index (0, 2, or 4)
    # i is always 0,1
    def add_strassen_222(j_off, k_off):
        nonlocal col
        # M1
        U[0*3 + 0, col] += 1; U[1*3 + 1, col] += 1
        V[0*5 + 0 + k_off, col] += 1; V[1*5 + 1 + k_off, col] += 1
        W[0*2 + 0, col] += 1; W[1*2 + 1, col] += 1
        col += 1
        
        # M2
        U[1*3 + 0, col] += 1; U[1*3 + 1, col] += 1
        V[0*5 + 0 + k_off, col] += 1
        W[1*2 + 0, col] += 1
        col += 1
        
        # M3
        U[0*3 + 0, col] += 1
        V[0*5 + 1 + k_off, col] += 1; V[1*5 + 1 + k_off, col] -= 1
        W[0*2 + 1, col] += 1
        col += 1
        
        # M4
        U[1*3 + 1, col] += 1
        V[1*5 + 0 + k_off, col] += 1; V[0*5 + 0 + k_off, col] -= 1
        W[1*2 + 0, col] += 1; W[0*2 + 1, col] += 1
        col += 1
        
        # M5
        U[0*3 + 0, col] += 1; U[0*3 + 1, col] += 1
        V[1*5 + 1 + k_off, col] += 1
        W[0*2 + 0, col] -= 1; W[0*2 + 1, col] += 1
        col += 1
        
        # M6
        U[1*3 + 0, col] += 1; U[0*3 + 0, col] -= 1
        V[0*5 + 0 + k_off, col] += 1; V[0*5 + 1 + k_off, col] += 1
        W[1*2 + 1, col] += 1
        col += 1
        
        # M7
        U[0*3 + 1, col] += 1; U[1*3 + 1, col] -= 1
        V[1*5 + 0 + k_off, col] += 1; V[1*5 + 1 + k_off, col] += 1
        W[0*2 + 0, col] += 1; W[1*2 + 1, col] -= 1
        col += 1

    # Helper to add a Schoolbook 2x2x1 block
    def add_schoolbook_221(j_off, k_off):
        nonlocal col
        for i in range(2):
            for j in range(2):
                U[i*3 + j + j_off, col] = 1
                V[(j + j_off)*5 + k_off, col] = 1
                W[k_off*2 + i, col] = 1
                col += 1

    # Helper to add a Schoolbook 2x1x5 block
    def add_schoolbook_215(j_off):
        nonlocal col
        for i in range(2):
            for k in range(5):
                U[i*3 + j_off, col] = 1
                V[j_off*5 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1

    # Part 1: j in {0, 1}
    # k in {0, 1}
    add_strassen_222(0, 0)
    # k in {2, 3}
    add_strassen_222(0, 2)
    # k in {4}
    add_schoolbook_221(0, 4)
    
    # Part 2: j in {2}
    add_schoolbook_215(2)
    
    return {'U': U, 'V': V, 'W': W}

if __name__ == '__main__':
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    # Verify
    T_calc = np.einsum('ac,bc,dc->abd', U, V, W)
    T = np.zeros((6, 15, 10))
    for i in range(2):
        for j in range(3):
            for k in range(5):
                T[i*3+j, j*5+k, k*2+i] = 1.0
    diff = np.linalg.norm(T_calc - T)
    print(f"Norm diff: {diff}")
    print(f"Rank: {U.shape[1]}")
