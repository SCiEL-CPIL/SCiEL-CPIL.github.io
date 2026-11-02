"""Rank-84 exact tensor decomposition for 4x4 * 4x6 matrix multiplication using Kronecker product of Strassen and schoolbook."""

import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: Kronecker product of Strassen (rank 7 for 2x2x2) and schoolbook (rank 12 for 2x2x3).
    This gives total rank 7 * 12 = 84, beating pure schoolbook rank 96.
    
    The 4x4 * 4x6 multiplication is decomposed as:
    - Split A into 2x2 blocks (4 blocks of 2x2)
    - Split B into 2x3 blocks (4 blocks of 2x3)  
    - Split C into 2x3 blocks (4 blocks of 2x3)
    
    Strassen reduces the 2x2 * 2x2 block multiplication to 7 products.
    Schoolbook for 2x2 * 2x3 has rank 12 (2*2*3).
    Kronecker product: 7 * 12 = 84 total rank.
    """
    start_time = time.time()
    DEADLINE = 150.0  # seconds
    
    n, m, p = 4, 4, 6
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    # Strassen's 7 intermediate products for 2x2 * 2x2 -> 2x2
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    strassen_steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
        # M2 = (A10 + A11)B00
        {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},
        # M3 = A00(B01 - B11)
        {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},
        # M4 = A11(B10 - B00)
        {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
        # M5 = (A00 + A01)B11
        {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},
        # M6 = (A10 - A00)(B00 + B01)
        {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},
        # M7 = (A01 - A11)(B10 + B11)
        {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]}
    ]
    
    # For each Strassen step, expand with schoolbook for 2x2 * 2x3 -> 2x3
    # Schoolbook rank for 2x2 * 2x3 is 2*2*3 = 12
    # Each term: C[r,k] += A[r,s] * B[s,k] for r in {0,1}, s in {0,1}, k in {0,1,2}
    
    for i, step in enumerate(strassen_steps):
        for r in range(2):       # row of output block (0 or 1)
            for s in range(2):   # sum index (0 or 1)
                for k in range(3):  # column of output block (0, 1, or 2)
                    # Local index within the 2x3 output block
                    c_in = r * 3 + s * 3 + k  # This needs to be: for each (r,s,k) triple
                    # Actually, the 12 schoolbook terms are indexed by (r, s, k) where
                    # r in {0,1}, s in {0,1}, k in {0,1,2}
                    # So c_in = r * 6 + s * 3 + k gives 0..11
                    c_in = r * 6 + s * 3 + k
                    c_global = i * 12 + c_in
                    
                    # Fill U: for each A block term
                    for block_id, coeff in step['A']:
                        if block_id == 0: I, J = r, s       # A00: row r, col s
                        elif block_id == 1: I, J = r, s+2    # A01: row r, col s+2
                        elif block_id == 2: I, J = r+2, s    # A10: row r+2, col s
                        elif block_id == 3: I, J = r+2, s+2  # A11: row r+2, col s+2
                        else: continue
                        u_idx = I * 4 + J
                        U[u_idx][c_global] += coeff
                    
                    # Fill V: for each B block term
                    for block_id, coeff in step['B']:
                        if block_id == 0: J, K = s, k       # B00: row s, col k
                        elif block_id == 1: J, K = s, k+3    # B01: row s, col k+3
                        elif block_id == 2: J, K = s+2, k    # B10: row s+2, col k
                        elif block_id == 3: J, K = s+2, k+3  # B11: row s+2, col k+3
                        else: continue
                        v_idx = J * 6 + K
                        V[v_idx][c_global] += coeff
                    
                    # Fill W: for each C block term
                    for block_id, coeff in step['C']:
                        if block_id == 0: I, K = r, k       # C00: row r, col k
                        elif block_id == 1: I, K = r, k+3    # C01: row r, col k+3
                        elif block_id == 2: I, K = r+2, k    # C10: row r+2, col k
                        elif block_id == 3: I, K = r+2, k+3  # C11: row r+2, col k+3
                        else: continue
                        w_idx = K * 4 + I  # W[k*n+i, r] convention: W[K*4+I]
                        W[w_idx][c_global] += coeff
                    
                    # Time guard
                    if time.time() - start_time > DEADLINE - 10:
                        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
