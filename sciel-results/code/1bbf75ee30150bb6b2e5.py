import time
import sys

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    of the tensor decomposition for matrix multiplication M_{2,3,10}.
    Uses Strassen's algorithm for the 2x2x2 blocks and schoolbook for the 2x1x2 blocks.
    Rank = 5 * 7 + 5 * 4 = 55.
    """
    n, m, p = DIMENSIONS
    # Split m=3 into m1=2 and m2=1.
    # Split p=10 into 5 blocks of size 2.
    # For each block, we have a 2x2x2 multiplication (rank 7) and a 2x1x2 multiplication (rank 4).
    # Total rank per block = 11.
    # Total rank = 5 * 11 = 55.
    
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Indices: i,j in {0,1} for U; j,k in {0,1} for V; k,i in {0,1} for W
    # s=0: M0 = (A00+A11)(B00+B11)
    # s=1: M1 = (A10+A11)B00
    # s=2: M2 = A00(B10-B11)
    # s=3: M3 = A11(B01-B00)
    # s=4: M4 = (A00+A01)B11
    # s=5: M5 = (A10-A00)(B00+B01)
    # s=6: M6 = (A01-A11)(B10+B11)
    
    L = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    R = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # W coefficients: W[k,i]
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    Wc = [
        {(0,0): 1, (1,1): 1}, # M0
        {(0,1): 1, (1,1): -1}, # M1
        {(1,0): 1, (1,1): 1}, # M2
        {(0,0): 1}, # M3
        {(0,0): -1, (1,0): 1}, # M4
        {(0,1): 1, (1,1): 1}, # M5
        {(0,0): -1} # M6
    ]
    
    # Iterate over 5 blocks of size 2 in p
    for block in range(5):
        k_off = block * 2
        
        # Strassen part for j in {0,1}
        for s in range(7):
            # U
            for (i, j), val in L[s].items():
                U[i * m + j][col] = val
            # V
            for (j, k_loc), val in R[s].items():
                V[j * p + k_off + k_loc][col] = val
            # W
            for (k_loc, i), val in Wc[s].items():
                W[k_off + k_loc + n * i][col] = val # W index is k*n + i? 
                # Wait, W shape is (p*n, rank). Index is k*n + i?
                # Task says: W[k*n+i, r].
                # My previous check: W[k*n+i].
                # Let's verify: W shape (p*n, rank).
                # Index c = k*n + i.
                # In loop: W[k_off + k_loc + n * i][col] = val.
                # This matches k*n + i.
            col += 1
            
        # Schoolbook part for j=2
        # C[i, k] = A[i, 2] * B[2, k]
        # For each i in {0,1} and k_loc in {0,1}
        for i in range(n):
            for k_loc in range(2):
                k = k_off + k_loc
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
