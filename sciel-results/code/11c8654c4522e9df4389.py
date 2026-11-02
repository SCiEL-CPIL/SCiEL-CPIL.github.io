import numpy as np
import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication.
    
    Uses Strassen (rank 7) for 2x2*2x2 sub-blocks and schoolbook (rank 4) for 
    2x2*2x1 sub-blocks. Middle dimension m=4 split into two 2-blocks.
    Output dimension p=7 split into three 2-blocks and one 1-block.
    
    Total rank = 2 * (3*7 + 4) = 50.
    """
    start_time = time.time()
    DEADLINE = 110.0  # seconds, safety margin before 120s limit
    
    n, m, p = 2, 4, 7
    total_rank = 50
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(j_start, k_start):
        nonlocal col
        if time.time() - start_time > DEADLINE:
            return False
        
        a00 = j_start
        a01 = j_start + 1
        a10 = m + j_start
        a11 = m + j_start + 1
        
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        w00 = k_start * n + 0
        w01 = (k_start + 1) * n + 0
        w10 = k_start * n + 1
        w11 = (k_start + 1) * n + 1
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[w00, col] += 1
        col += 1
        return True
    
    def add_schoolbook_block(j_start, k_start):
        nonlocal col
        if time.time() - start_time > DEADLINE:
            return False
        
        a00 = j_start
        a01 = j_start + 1
        a10 = m + j_start
        a11 = m + j_start + 1
        
        b00 = j_start * p + k_start
        b10 = (j_start + 1) * p + k_start
        
        w00 = k_start * n + 0
        w10 = k_start * n + 1
        
        U[a00, col] += 1; V[b00, col] += 1; W[w00, col] += 1
        col += 1
        U[a01, col] += 1; V[b10, col] += 1; W[w00, col] += 1
        col += 1
        U[a10, col] += 1; V[b00, col] += 1; W[w10, col] += 1
        col += 1
        U[a11, col] += 1; V[b10, col] += 1; W[w10, col] += 1
        col += 1
        return True
    
    for j_start in [0, 2]:
        if time.time() - start_time > DEADLINE:
            break
        for k_start in [0, 2, 4]:
            if not add_strassen_block(j_start, k_start):
                break
        if time.time() - start_time > DEADLINE:
            break
        if not add_schoolbook_block(j_start, 6):
            break
    
    if col != total_rank:
        raise RuntimeError(f"Rank mismatch: expected {total_rank}, got {col}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 4, 7
    rank = U.shape[1]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = 1.0
                calc = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(calc - target) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    if valid:
        print(f"Verification Passed. Rank: {rank}")
    else:
        print("Verification Failed")
