import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 7-term decomposition for 2x2 * 2x2
    # Terms are defined by indices (i, j, k) relative to the 2x2 block
    # U indices: i (0-1), j (0-1) -> i*2 + j
    # V indices: j (0-1), k (0-1) -> j*2 + k
    # W indices: k (0-1), i (0-1) -> k*2 + i
    
    # 1. (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11 (indices 0, 3 in 2x2) -> 0, 3? 
    # Wait, U shape is (n*m, rank) = (8, 56).
    # For a block at j_block, k_block:
    # A indices: (i, j_block + j_rel)
    # B indices: (j_block + j_rel, k_block + k_rel)
    # C indices: (k_block + k_rel, i)
    
    def add_strassen_block(j_off, k_off, base_col):
        # 1. (A00+A11)(B00+B11) -> C00, C11
        U[0*2 + j_off + 0, base_col + 0] += 1
        U[1*2 + j_off + 1, base_col + 0] += 1
        V[(j_off + 0)*4 + k_off + 0, base_col + 0] += 1
        V[(j_off + 1)*4 + k_off + 1, base_col + 0] += 1
        W[(k_off + 0)*2 + 0, base_col + 0] += 1
        W[(k_off + 1)*2 + 1, base_col + 0] += 1
        
        # 2. (A10+A11)(B00) -> C10, C11(-1)
        U[1*2 + j_off + 0, base_col + 1] += 1
        U[1*2 + j_off + 1, base_col + 1] += 1
        V[(j_off + 0)*4 + k_off + 0, base_col + 1] += 1
        W[(k_off + 0)*2 + 1, base_col + 1] += 1
        W[(k_off + 1)*2 + 1, base_col + 1] -= 1
        
        # 3. A00(B01-B11) -> C01, C11
        U[0*2 + j_off + 0, base_col + 2] += 1
        V[(j_off + 0)*4 + k_off + 1, base_col + 2] += 1
        V[(j_off + 1)*4 + k_off + 1, base_col + 2] -= 1
        W[(k_off + 1)*2 + 0, base_col + 2] += 1
        W[(k_off + 1)*2 + 1, base_col + 2] += 1
        
        # 4. A11(B10-B00) -> C10, C00
        U[1*2 + j_off + 1, base_col + 3] += 1
        V[(j_off + 1)*4 + k_off + 0, base_col + 3] += 1
        V[(j_off + 0)*4 + k_off + 0, base_col + 3] -= 1
        W[(k_off + 0)*2 + 0, base_col + 3] += 1
        W[(k_off + 0)*2 + 1, base_col + 3] += 1
        
        # 5. (A00+A01)B11 -> C00(-1), C11
        U[0*2 + j_off + 0, base_col + 4] += 1
        U[0*2 + j_off + 1, base_col + 4] += 1
        V[(j_off + 1)*4 + k_off + 1, base_col + 4] += 1
        W[(k_off + 0)*2 + 0, base_col + 4] -= 1
        W[(k_off + 1)*2 + 0, base_col + 4] += 1
        
        # 6. (A10-A00)(B00+B01) -> C11
        U[1*2 + j_off + 0, base_col + 5] += 1
        U[0*2 + j_off + 0, base_col + 5] -= 1
        V[(j_off + 0)*4 + k_off + 0, base_col + 5] += 1
        V[(j_off + 0)*4 + k_off + 1, base_col + 5] += 1
        W[(k_off + 1)*2 + 1, base_col + 5] += 1
        
        # 7. (A01-A11)(B10+B11) -> C00
        U[0*2 + j_off + 1, base_col + 6] += 1
        U[1*2 + j_off + 1, base_col + 6] -= 1
        V[(j_off + 1)*4 + k_off + 0, base_col + 6] += 1
        V[(j_off + 1)*4 + k_off + 1, base_col + 6] += 1
        W[(k_off + 0)*2 + 0, base_col + 6] += 1

    # We have 2 blocks in j (0, 1) and 4 blocks in k (0, 1, 2, 3)
    # j_off is 0 or 2 (since m=4, j is 0-3, blocks of size 2)
    # k_off is 0, 2, 4, 6 (since p=8, k is 0-7, blocks of size 2)
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 2, 4, 6]:
            add_strassen_block(j_off, k_off, col)
            col += 7
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
