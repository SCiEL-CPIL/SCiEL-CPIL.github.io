import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125.0  # safety margin before 150s kill
    
    # Helper to verify decomposition
    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, (n, m)).astype(float)
        B_test = np.random.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test
        
        rank = U_arr.shape[1]
        C_computed = np.zeros((n, p))
        
        # Efficient verification using matrix operations
        # C[i, j] = sum_t W[j*n+i, t] * (sum_k U[i*m+k, t] * V[k*p+j, t])
        # Let T = U @ V.T -> shape (n*m, m*p) ... no
        # Let's just loop, it's small enough (36x36x133)
        
        for i in range(n):
            for j in range(p):
                total = 0.0
                for t in range(rank):
                    s_val = 0.0
                    for k in range(m):
                        s_val += U_arr[i*m+k, t] * V_arr[k*p+j, t]
                    total += W_arr[j*n+i, t] * s_val
                C_computed[i, j] = total
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Laderman 3x3 terms
    def get_laderman_3x3():
        terms = []
        # P1
        terms.append(({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}))
        # P2
        terms.append(({0:1, 3:-1}, {0:1, 3:-1}, {0:1, 3:1}))
        # P3
        terms.append(({4:1, 7:1}, {3:1, 6:1}, {3:1, 6:1}))
        # P4
        terms.append(({4:1, 7:-1}, {3:1, 6:-1}, {3:1, 6:1}))
        # P5
        terms.append(({1:1, 2:1}, {7:1, 8:1}, {1:1, 2:1}))
        # P6
        terms.append(({1:1, 2:-1}, {7:1, 8:-1}, {1:1, 2:1}))
        # P7
        terms.append(({6:1, 8:1}, {1:1, 2:1}, {7:1, 8:1}))
        # P8
        terms.append(({6:1, 8:-1}, {1:1, 2:-1}, {7:1, 8:1}))
        # P9
        terms.append(({0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}, {0:1, 4:1, 8:1}))
        # P10
        terms.append(({3:1, 8:1}, {6:1, 1:1}, {4:-1, 8:-1}))
        # P11
        terms.append(({6:1, 4:1}, {3:1, 2:1}, {7:-1, 2:-1}))
        # P12
        terms.append(({1:1, 8:1}, {7:1, 0:1}, {0:-1, 7:-1}))
        # P13
        terms.append(({2:1, 4:1}, {5:1, 0:1}, {2:-1, 4:-1}))
        # P14
        terms.append(({5:1, 1:1}, {1:1, 4:1}, {5:-1, 1:-1}))
        # P15
        terms.append(({7:1, 2:1}, {2:1, 4:1}, {8:-1, 2:-1}))
        # P16
        terms.append(({8:1, 3:1}, {3:1, 2:1}, {8:-1, 3:-1}))
        # P17
        terms.append(({0:1, 8:1}, {0:1, 8:1}, {0:1, 8:1}))
        # P18
        terms.append(({1:1, 5:1}, {0:1, 4:1}, {1:1, 5:1}))
        # P19
        terms.append(({2:1, 6:1}, {0:1, 4:1}, {2:1, 6:1}))
        return terms

    laderman_terms = get_laderman_3x3()
    
    # Try 133-rank decomposition
    rank_133 = 7 * 19
    U_133 = np.zeros((n * m, rank_133), dtype=int)
    V_133 = np.zeros((m * p, rank_133), dtype=int)
    W_133 = np.zeros((p * n, rank_133), dtype=int)
    
    # Strassen terms for 2x2 blocks of 3x3 matrices
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    strassen_defs = [
        # (A_blocks, B_blocks)
        # Each block is (br, bc, coeff)
        # A00 is (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1)
        [([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
         ([(1,0,1), (1,1,1)], [(0,0,1)]),
         ([(0,0,1)], [(0,1,1), (1,1,-1)]),
         ([(1,1,1)], [(1,0,1), (0,0,-1)]),
         ([(0,0,1), (0,1,1)], [(1,1,1)]),
         ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
         ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])]
    ]
    
    # Strassen reconstruction coefficients for C blocks
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C blocks: C00(0,0), C01(0,1), C10(1,0), C11(1,1)
    strassen_c_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): -1},      # M2
        {(0,1): 1, (1,1): 1},       # M3
        {(0,0): 1, (1,0): 1},       # M4
        {(0,0): -1, (0,1): 1},      # M5
        {(1,1): 1},                 # M6
        {(0,0): 1}                  # M7
    ]
    
    for k in range(7):
        a_blocks = strassen_defs[0][k][0]
        b_blocks = strassen_defs[0][k][1]
        
        for l in range(19):
            t = k * 19 + l
            a_laderman, b_laderman, c_laderman = laderman_terms[l]
            
            # Process U
            for a_flat, a_coeff in a_laderman.items():
                a_row = a_flat // 3
                a_col = a_flat % 3
                for (br, bc, block_coeff) in a_blocks:
                    global_row = br * 3 + a_row
                    global_col = bc * 3 + a_col
                    global_idx = global_row * 6 + global_col
                    U_133[global_idx, t] += a_coeff * block_coeff
            
            # Process V
            for b_flat, b_coeff in b_laderman.items():
                b_row = b_flat // 3
                b_col = b_flat % 3
                for (br, bc, block_coeff) in b_blocks:
                    global_row = br * 3 + b_row
                    global_col = bc * 3 + b_col
                    global_idx = global_row * 6 + global_col
                    V_133[global_idx, t] += b_coeff * block_coeff
            
            # Process W
            for c_flat, c_coeff in c_laderman.items():
                c_row = c_flat // 3
                c_col = c_flat % 3
                for (c_br, c_bc, c_block_coeff) in strassen_c_coeffs[k]:
                    global_row = c_br * 3 + c_row
                    global_col = c_bc * 3 + c_col
                    # W index is j*n + i, where i is row, j is col
                    global_idx = global_col * 6 + global_row
                    W_133[global_idx, t] += c_coeff * c_block_coeff
    
    if verify(U_133.tolist(), V_133.tolist(), W_133.tolist()):
        return {"U": U_133.tolist(), "V": V_133.tolist(), "W": W_133.tolist()}
    
    # Fallback to 189-rank decomposition
    rank_189 = 27 * 7
    U_189 = np.zeros((n * m, rank_189), dtype=int)
    V_189 = np.zeros((m * p, rank_189), dtype=int)
    W_189 = np.zeros((p * n, rank_189), dtype=int)
    
    idx = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                for s in range(7):
                    a00 = (2*i) * 6 + 2*k
                    a01 = (2*i) * 6 + 2*k + 1
                    a10 = (2*i+1) * 6 + 2*k
                    a11 = (2*i+1) * 6 + 2*k + 1
                    
                    b00 = (2*k) * 6 + 2*j
                    b01 = (2*k) * 6 + 2*j + 1
                    b10 = (2*k+1) * 6 + 2*j
                    b11 = (2*k+1) * 6 + 2*j + 1
                    
                    c00 = (2*j) * 6 + 2*i
                    c01 = (2*j+1) * 6 + 2*i
                    c10 = (2*j) * 6 + (2*i+1)
                    c11 = (2*j+1) * 6 + (2*i+1)
                    
                    if s == 0:
                        U_189[a00, idx] = 1
                        U_189[a11, idx] = 1
                        V_189[b00, idx] = 1
                        V_189[b11, idx] = 1
                        W_189[c00, idx] = 1
                        W_189[c11, idx] = 1
                    elif s == 1:
                        U_189[a10, idx] = 1
                        U_189[a11, idx] = 1
                        V_189[b00, idx] = 1
                        W_189[c10, idx] = 1
                        W_189[c11, idx] = -1
                    elif s == 2:
                        U_189[a00, idx] = 1
                        V_189[b01, idx] = 1
                        V_189[b11, idx] = -1
                        W_189[c01, idx] = 1
                        W_189[c11, idx] = 1
                    elif s == 3:
                        U_189[a11, idx] = 1
                        V_189[b10, idx] = 1
                        V_189[b00, idx] = -1
                        W_189[c00, idx] = 1
                        W_189[c10, idx] = 1
                    elif s == 4:
                        U_189[a00, idx] = 1
                        U_189[a01, idx] = 1
                        V_189[b11, idx] = 1
                        W_189[c00, idx] = -1
                        W_189[c01, idx] = 1
                    elif s == 5:
                        U_189[a10, idx] = 1
                        U_189[a00, idx] = -1
                        V_189[b00, idx] = 1
                        V_189[b01, idx] = 1
                        W_189[c11, idx] = 1
                    elif s == 6:
                        U_189[a01, idx] = 1
                        U_189[a11, idx] = -1
                        V_189[b10, idx] = 1
                        V_189[b11, idx] = 1
                        W_189[c00, idx] = 1
                    
                    idx += 1
    
    return {"U": U_189.tolist(), "V": V_189.tolist(), "W": W_189.tolist()}
# EVOLVE-BLOCK-END
