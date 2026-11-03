import numpy as np

def construct():
    n, m, p = 5, 5, 5
    rank = 124
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
    def add_strassen_2x2(i_start, j_start, k_start):
        nonlocal col
        # Local indices 0,1 map to global i_start+i, j_start+j, k_start+k
        # U indices: (i_start+i)*m + (j_start+j)
        # V indices: (j_start+j)*p + (k_start+k)
        # W indices: (k_start+k)*n + (i_start+i)
        
        # Term 1
        U[i_start*m + j_start, col] = 1.0
        U[(i_start+1)*m + (j_start+1), col] = 1.0
        V[j_start*p + k_start, col] = 1.0
        V[(j_start+1)*p + (k_start+1), col] = 1.0
        W[k_start*n + i_start, col] = 1.0
        W[(k_start+1)*n + (i_start+1), col] = 1.0
        col += 1
        
        # Term 2
        U[(i_start+1)*m + j_start, col] = 1.0
        U[(i_start+1)*m + (j_start+1), col] = 1.0
        V[j_start*p + k_start, col] = 1.0
        W[(k_start+1)*n + (i_start+1), col] = -1.0
        W[k_start*n + (i_start+1), col] = 1.0
        col += 1
        
        # Term 3
        U[i_start*m + j_start, col] = 1.0
        V[(j_start+1)*p + k_start, col] = 1.0
        V[(j_start+1)*p + (k_start+1), col] = -1.0
        W[(k_start+1)*n + i_start, col] = 1.0
        W[(k_start+1)*n + (i_start+1), col] = 1.0
        col += 1
        
        # Term 4
        U[(i_start+1)*m + (j_start+1), col] = 1.0
        V[(j_start+1)*p + k_start, col] = 1.0
        V[j_start*p + k_start, col] = -1.0
        W[k_start*n + i_start, col] = 1.0
        W[k_start*n + (i_start+1), col] = 1.0
        col += 1
        
        # Term 5
        U[i_start*m + j_start, col] = 1.0
        U[i_start*m + (j_start+1), col] = 1.0
        V[(j_start+1)*p + (k_start+1), col] = 1.0
        W[k_start*n + i_start, col] = -1.0
        W[(k_start+1)*n + i_start, col] = 1.0
        col += 1
        
        # Term 6
        U[i_start*m + j_start, col] = -1.0
        U[(i_start+1)*m + j_start, col] = 1.0
        V[j_start*p + k_start, col] = 1.0
        V[(j_start+1)*p + k_start, col] = 1.0
        W[(k_start+1)*n + (i_start+1), col] = 1.0
        col += 1
        
        # Term 7
        U[i_start*m + (j_start+1), col] = 1.0
        U[(i_start+1)*m + (j_start+1), col] = -1.0
        V[(j_start+1)*p + k_start, col] = 1.0
        V[(j_start+1)*p + (k_start+1), col] = 1.0
        W[k_start*n + i_start, col] = 1.0
        col += 1

    # Block 1: A00 B00 (2x2, 2x2 -> 2x2)
    # i in [0,2), j in [0,2), k in [0,2)
    add_strassen_2x2(0, 0, 0)
    
    # Block 2: A01 B10 (2x3, 3x2 -> 2x2)
    # i in [0,2), j in [2,5), k in [0,2)
    add_schoolbook(0, 2, 2, 5, 0, 2)
    
    # Block 3: A00 B01 (2x2, 2x3 -> 2x3)
    # i in [0,2), j in [0,2), k in [2,5)
    add_schoolbook(0, 2, 0, 2, 2, 5)
    
    # Block 4: A01 B11 (2x3, 3x3 -> 2x3)
    # i in [0,2), j in [2,5), k in [2,5)
    add_schoolbook(0, 2, 2, 5, 2, 5)
    
    # Block 5: A10 B00 (3x2, 2x2 -> 3x2)
    # i in [2,5), j in [0,2), k in [0,2)
    add_schoolbook(2, 5, 0, 2, 0, 2)
    
    # Block 6: A11 B10 (3x3, 3x2 -> 3x2)
    # i in [2,5), j in [2,5), k in [0,2)
    add_schoolbook(2, 5, 2, 5, 0, 2)
    
    # Block 7: A10 B01 (3x2, 2x3 -> 3x3)
    # i in [2,5), j in [0,2), k in [2,5)
    add_schoolbook(2, 5, 0, 2, 2, 5)
    
    # Block 8: A11 B11 (3x3, 3x3 -> 3x3)
    # i in [2,5), j in [2,5), k in [2,5)
    add_schoolbook(2, 5, 2, 5, 2, 5)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
