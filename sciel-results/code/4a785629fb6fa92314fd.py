import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Returns a rank-56 exact tensor decomposition for multiplying a 2x4 matrix 
    by a 4x8 matrix.
    
    The decomposition uses Strassen's algorithm (rank 7) for each of the 8 
    independent 2x2 * 2x2 sub-problems formed by blocking:
    - A (2x4) into two 2x2 blocks: A[:, 0:2] and A[:, 2:4]
    - B (4x8) into eight 2x2 blocks: B[2b:2b+2, 2c:2c+2] for b in 0..1, c in 0..3
    
    The output C (2x8) is formed by summing contributions:
    C[:, 2c:2c+2] = A[:, 0:2] * B[0:2, 2c:2c+2] + A[:, 2:4] * B[2:4, 2c:2c+2]
    
    Total rank = 8 * 7 = 56.
    """
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=float)  # (8, 56)
    V = np.zeros((m * p, rank), dtype=float)  # (32, 56)
    W = np.zeros((p * n, rank), dtype=float)  # (16, 56)
    
    col = 0
    for b in range(2):  # A block index (0 or 1)
        for c in range(4):  # B column block index (0, 1, 2, or 3)
            if time.time() > deadline:
                break
            
            # Local indices for A (2x2 block A_b)
            # A is indexed as A[row, col] -> flat index row*m + col
            idx_a = 0 * m + 2 * b       # A[0, 2b]
            idx_b = 0 * m + 2 * b + 1   # A[0, 2b+1]
            idx_c_a = 1 * m + 2 * b     # A[1, 2b]
            idx_d = 1 * m + 2 * b + 1   # A[1, 2b+1]
            
            # Local indices for B (2x2 block B_bc)
            # B is indexed as B[row, col] -> flat index row*p + col
            idx_e = 2 * b * p + 2 * c       # B[2b, 2c]
            idx_f = 2 * b * p + 2 * c + 1   # B[2b, 2c+1]
            idx_g = (2 * b + 1) * p + 2 * c     # B[2b+1, 2c]
            idx_h = (2 * b + 1) * p + 2 * c + 1 # B[2b+1, 2c+1]
            
            # Local indices for C (2x2 output block)
            # W follows convention W[k*n+i, r] where k is col block, i is row
            # C is 2x8, indexed as C[row, col] -> flat index col*n + row
            idx_c11 = 2 * c * n + 0       # C[0, 2c]
            idx_c12 = (2 * c + 1) * n + 0 # C[0, 2c+1]
            idx_c21 = 2 * c * n + 1       # C[1, 2c]
            idx_c22 = (2 * c + 1) * n + 1 # C[1, 2c+1]
            
            # M1 = (a+d)(e+h)
            U[idx_a, col] += 1
            U[idx_d, col] += 1
            V[idx_e, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[idx_c_a, col] += 1
            U[idx_d, col] += 1
            V[idx_e, col] += 1
            W[idx_c21, col] += 1
            W[idx_c22, col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[idx_a, col] += 1
            V[idx_f, col] += 1
            V[idx_h, col] -= 1
            W[idx_c12, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[idx_d, col] += 1
            V[idx_g, col] += 1
            V[idx_e, col] -= 1
            W[idx_c11, col] += 1
            W[idx_c21, col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[idx_a, col] += 1
            U[idx_b, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] -= 1
            W[idx_c12, col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[idx_c_a, col] += 1
            U[idx_a, col] -= 1
            V[idx_e, col] += 1
            V[idx_f, col] += 1
            W[idx_c22, col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[idx_b, col] += 1
            U[idx_d, col] -= 1
            V[idx_g, col] += 1
            V[idx_h, col] += 1
            W[idx_c11, col] += 1
            col += 1
        
        if time.time() > deadline:
            break

    # Convert to lists of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Shapes:", len(res["U"]), len(res["U"][0]), len(res["V"]), len(res["V"][0]), len(res["W"]), len(res["W"][0]))
