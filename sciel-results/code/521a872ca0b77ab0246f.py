import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, use full time safely
    
    n, m, p = 2, 4, 8
    rank = 56
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen coefficients for 2x2 * 2x2
    # Term t: u_t = sum x_coeffs[i,j] * A[i,j]
    #          v_t = sum y_coeffs[j,k] * B[j,k]
    #          C contribution = u_t * v_t * z_coeffs[i,k]
    strassen_terms = [
        # t=0: (x00+x11)(y00+y11) -> C00 += M, C11 += M
        {'x': {(0, 0): 1, (1, 1): 1}, 'y': {(0, 0): 1, (1, 1): 1}, 'z': {(0, 0): 1, (1, 1): 1}},
        # t=1: (x10+x11)y00 -> C10 += M, C11 -= M
        {'x': {(1, 0): 1, (1, 1): 1}, 'y': {(0, 0): 1}, 'z': {(1, 0): 1, (1, 1): -1}},
        # t=2: x00(y01-y11) -> C01 += M, C11 += M
        {'x': {(0, 0): 1}, 'y': {(0, 1): 1, (1, 1): -1}, 'z': {(0, 1): 1, (1, 1): 1}},
        # t=3: x11(y10-y00) -> C00 += M, C10 += M
        {'x': {(1, 1): 1}, 'y': {(1, 0): 1, (0, 0): -1}, 'z': {(0, 0): 1, (1, 0): 1}},
        # t=4: (x00+x01)y11 -> C00 -= M, C01 += M
        {'x': {(0, 0): 1, (0, 1): 1}, 'y': {(1, 1): 1}, 'z': {(0, 0): -1, (0, 1): 1}},
        # t=5: (x10-x00)(y00+y01) -> C11 += M
        {'x': {(1, 0): 1, (0, 0): -1}, 'y': {(0, 0): 1, (0, 1): 1}, 'z': {(1, 1): 1}},
        # t=6: (x01-x11)(y10+y11) -> C00 += M
        {'x': {(0, 1): 1, (1, 1): -1}, 'y': {(1, 0): 1, (1, 1): 1}, 'z': {(0, 0): 1}},
    ]
    
    col_idx = 0
    # A is 2x4: split columns into 2 blocks of 2 (a_block = 0,1)
    # B is 4x8: for each a_block, B rows 2*a_block..2*a_block+1 form a 2x8 matrix
    #           split B columns into 4 blocks of 2 (b_col_block = 0,1,2,3)
    # C is 2x8: output for each (a_block, b_col_block) contributes to C[:, 2*b_col_block:2*b_col_block+2]
    
    for a_block in range(2):  # A column block: cols 2*a_block, 2*a_block+1
        for b_col_block in range(4):  # B column block: cols 2*b_col_block, 2*b_col_block+1
            for t in range(7):
                term = strassen_terms[t]
                
                # X part: A[i', 2*a_block + j']
                for (i_prime, j_prime), coeff in term['x'].items():
                    u_idx = i_prime * m + (2 * a_block + j_prime)
                    U[u_idx][col_idx] += coeff
                
                # Y part: B[2*a_block + j', 2*b_col_block + k']
                for (j_prime, k_prime), coeff in term['y'].items():
                    b_row = 2 * a_block + j_prime
                    b_col = 2 * b_col_block + k_prime
                    v_idx = b_row * p + b_col
                    V[v_idx][col_idx] += coeff
                
                # Z part: C[i', 2*b_col_block + k'] -> W[2*b_col_block + k', i']
                for (i_prime, k_prime), coeff in term['z'].items():
                    c_col = 2 * b_col_block + k_prime
                    c_row = i_prime
                    w_idx = c_col * n + c_row
                    W[w_idx][col_idx] += coeff
                
                col_idx += 1
                
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
