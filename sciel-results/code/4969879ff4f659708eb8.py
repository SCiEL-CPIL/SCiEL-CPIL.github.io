import time
import numpy as np

def construct():
    """
    Rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    3 Strassen blocks (rank 7 each) + schoolbook for last column (rank 4).
    """
    t_start = time.time()
    deadline = t_start + 120.0
    
    n, m, p = 2, 2, 7
    rank = 25
    
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[0 + k_start, col_start+0] += 1
        V[7 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        U[0, col_start+2] += 1
        V[0 + (k_start+1), col_start+2] += 1
        V[7 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        U[3, col_start+3] += 1
        V[7 + k_start, col_start+3] += 1
        V[0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[0 + k_start, col_start+5] += 1
        V[0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7 + k_start, col_start+6] += 1
        V[7 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    col_start = 21
    k = 6
    U[0, col_start+0] += 1; V[0 + k, col_start+0] += 1; W[2*k + 0, col_start+0] += 1
    U[1, col_start+1] += 1; V[7 + k, col_start+1] += 1; W[2*k + 0, col_start+1] += 1
    U[2, col_start+2] += 1; V[0 + k, col_start+2] += 1; W[2*k + 1, col_start+2] += 1
    U[3, col_start+3] += 1; V[7 + k, col_start+3] += 1; W[2*k + 1, col_start+3] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
