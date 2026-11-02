import time
import numpy as np

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1 (Block 1)
    - Use Strassen's rank-7 algorithm for columns 2,3 (Block 2)
    - Use Strassen's rank-7 algorithm for columns 4,5 (Block 3)
    - Use schoolbook (rank 4) for column 6 (Block 4)
    Total rank: 7 + 7 + 7 + 4 = 25
    
    The tensor identity is:
    sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] == 1 for all valid i, j, k.
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    # Initialize factor matrices
    # U: (n*m, rank) -> (4, 25)
    # V: (m*p, rank) -> (14, 25)
    # W: (p*n, rank) -> (14, 25)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    def add_strassen_block(col_start, k_start):
        """
        Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
        col_start: starting column index in U, V, W for this block's terms.
        k_start: starting column index in B (and C).
        """
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # Contributes to C0,ks and C1,ks+1
        U[0, col_start+0] += 1  # A00
        U[3, col_start+0] += 1  # A11
        V[7*0 + k_start, col_start+0] += 1       # B0,ks
        V[7*1 + (k_start+1), col_start+0] += 1   # B1,ks+1
        W[2*k_start + 0, col_start+0] += 1       # C0,ks
        W[2*(k_start+1) + 1, col_start+0] += 1   # C1,ks+1

        # M2 = (A10 + A11) * B0,ks
        # Contributes to C1,ks and C1,ks+1 (with negative sign for C1,ks+1)
        U[2, col_start+1] += 1  # A10
        U[3, col_start+1] += 1  # A11
        V[7*0 + k_start, col_start+1] += 1       # B0,ks
        W[2*k_start + 1, col_start+1] += 1       # C1,ks
        W[2*(k_start+1) + 1, col_start+1] -= 1   # C1,ks+1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # Contributes to C0,ks+1 and C1,ks+1
        U[0, col_start+2] += 1  # A00
        V[7*0 + (k_start+1), col_start+2] += 1   # B0,ks+1
        V[7*1 + (k_start+1), col_start+2] -= 1   # B1,ks+1
        W[2*(k_start+1) + 0, col_start+2] += 1   # C0,ks+1
        W[2*(k_start+1) + 1, col_start+2] += 1   # C1,ks+1

        # M4 = A11 * (B1,ks - B0,ks)
        # Contributes to C0,ks and C1,ks
        U[3, col_start+3] += 1  # A11
        V[7*1 + k_start, col_start+3] += 1       # B1,ks
        V[7*0 + k_start, col_start+3] -= 1       # B0,ks
        W[2*k_start + 0, col_start+3] += 1       # C0,ks
        W[2*k_start + 1, col_start+3] += 1       # C1,ks

        # M5 = (A00 + A01) * B1,ks+1
        # Contributes to C0,ks (negative) and C0,ks+1
        U[0, col_start+4] += 1  # A00
        U[1, col_start+4] += 1  # A01
        V[7*1 + (k_start+1), col_start+4] += 1   # B1,ks+1
        W[2*k_start + 0, col_start+4] -= 1       # C0,ks
        W[2*(k_start+1) + 0, col_start+4] += 1   # C0,ks+1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # Contributes to C1,ks+1
        U[2, col_start+5] += 1  # A10
        U[0, col_start+5] -= 1  # A00
        V[7*0 + k_start, col_start+5] += 1       # B0,ks
        V[7*0 + (k_start+1), col_start+5] += 1   # B0,ks+1
        W[2*(k_start+1) + 1, col_start+5] += 1   # C1,ks+1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # Contributes to C0,ks
        U[1, col_start+6] += 1  # A01
        U[3, col_start+6] -= 1  # A11
        V[7*1 + k_start, col_start+6] += 1       # B1,ks
        V[7*1 + (k_start+1), col_start+6] += 1   # B1,ks+1
        W[2*k_start + 0, col_start+6] += 1       # C0,ks

    # Apply Strassen to pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    add_strassen_block(14, 4)
    
    # Apply schoolbook to column 6
    col_start = 21
    k = 6
    
    # C0,6 = A00*B0,6 + A01*B1,6
    # Term 1: A00 * B0,6
    U[0, col_start+0] += 1
    V[7*0 + k, col_start+0] += 1
    W[2*k + 0, col_start+0] += 1
    
    # Term 2: A01 * B1,6
    U[1, col_start+1] += 1
    V[7*1 + k, col_start+1] += 1
    W[2*k + 0, col_start+1] += 1
    
    # C1,6 = A10*B0,6 + A11*B1,6
    # Term 3: A10 * B0,6
    U[2, col_start+2] += 1
    V[7*0 + k, col_start+2] += 1
    W[2*k + 1, col_start+2] += 1
    
    # Term 4: A11 * B1,6
    U[3, col_start+3] += 1
    V[7*1 + k, col_start+3] += 1
    W[2*k + 1, col_start+3] += 1

    # Verification
    # Check tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                s = 0
                for r in range(rank):
                    s += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                
                if s != 1:
                    valid = False
                    # print(f"Mismatch at {i},{j},{k}: sum={s}")
    
    if not valid:
        # Fallback to schoolbook rank 28 if verification fails (should not happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
