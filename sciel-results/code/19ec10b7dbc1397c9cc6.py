import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, U.shape[1]):
        return False
    if V.shape != (m*p, V.shape[1]):
        return False
    if W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                true_val = 1.0
                dec_val = 0.0
                for r in range(rank):
                    u_val = U[i*m + j, r]
                    v_val = V[j*p + k, r]
                    w_val = W[k*n + i, r]
                    dec_val += u_val * v_val * w_val
                if abs(dec_val - true_val) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 13 per 2x3x3 block, 2 blocks for width 6
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 13 terms for a 2x3x3 block
    # A is 2x3, B is 3x3, C is 2x3
    # Indices: A(i,j), B(j,k), C(i,k)
    # Strassen on A(:,0:2) * B(0:2, :)
    # A_L = A[:, 0:2] (2x2), B_T = B[0:2, :] (2x3)
    # B_T = [B_22, B_21] where B_22 is 2x2, B_21 is 2x1
    # Strassen on A_L * B_22
    # Scalar on A_L * B_21
    # Scalar on A_R * B_B (A_R is 2x1, B_B is 1x3)
    
    terms = []
    
    # Strassen terms for A_L (2x2) * B_22 (2x2)
    # A_L indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (in 2x3 A, col 0,1)
    # B_22 indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (in 3x3 B, row 0,1 col 0,1)
    # C_22 indices: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4 (in 2x3 C, col 0,1)
    
    # M1 = (A00+A11)(B00+B11) -> C00+C01+C10+C11
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 4: 1}, 'w': {0: 1, 1: 1, 3: 1, 4: 1}})
    # M2 = (A10+A11)B00 -> C10+C11
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {3: 1, 4: 1}})
    # M3 = A00(B11-B01) -> C01-C00
    terms.append({'u': {0: 1}, 'v': {4: 1, 1: -1}, 'w': {1: 1, 0: -1}})
    # M4 = A11(B10-B00) -> C11-C10
    terms.append({'u': {4: 1}, 'v': {3: 1, 0: -1}, 'w': {4: 1, 3: -1}})
    # M5 = (A00+A01)B11 -> C01+C11
    terms.append({'u': {0: 1, 1: 1}, 'v': {4: 1}, 'w': {1: 1, 4: 1}})
    # M6 = (A10-A00)(B00+B01) -> C10+C11-C00-C01
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1, 4: 1, 0: -1, 1: -1}})
    # M7 = (A01-A11)(B10+B11) -> C00+C01-C10-C11
    terms.append({'u': {1: 1, 4: -1}, 'v': {3: 1, 4: 1}, 'w': {0: 1, 1: 1, 3: -1, 4: -1}})
    
    # Scalar terms for A_L * B_21 (A[:,0:2] * B[0:2, 2])
    # A00 * B02 -> C02
    terms.append({'u': {0: 1}, 'v': {2: 1}, 'w': {2: 1}})
    # A01 * B12 -> C02
    terms.append({'u': {1: 1}, 'v': {5: 1}, 'w': {2: 1}})
    # A10 * B02 -> C12
    terms.append({'u': {3: 1}, 'v': {2: 1}, 'w': {5: 1}})
    # A11 * B12 -> C12
    terms.append({'u': {4: 1}, 'v': {5: 1}, 'w': {5: 1}})
    
    # Scalar terms for A_R * B_B (A[:,2] * B[2, 0:3])
    # A02 * B20 -> C00
    terms.append({'u': {2: 1}, 'v': {6: 1}, 'w': {0: 1}})
    # A02 * B21 -> C01
    terms.append({'u': {2: 1}, 'v': {7: 1}, 'w': {1: 1}})
    # A02 * B22 -> C02
    terms.append({'u': {2: 1}, 'v': {8: 1}, 'w': {2: 1}})
    # A12 * B20 -> C10
    terms.append({'u': {5: 1}, 'v': {6: 1}, 'w': {3: 1}})
    # A12 * B21 -> C11
    terms.append({'u': {5: 1}, 'v': {7: 1}, 'w': {4: 1}})
    # A12 * B22 -> C12
    terms.append({'u': {5: 1}, 'v': {8: 1}, 'w': {5: 1}})
    
    # Total 13 terms
    
    for block in range(2):
        for t_idx, term in enumerate(terms):
            col = block * 13 + t_idx
            for idx, val in term['u'].items():
                U[idx][col] = val
            for l_idx, val in term['v'].items():
                # l_idx is 0..8 for 3x3 B
                j_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = j_local * p + (block * 3 + k_local)
                V[g_idx][col] = val
            for l_idx, val in term['w'].items():
                # l_idx is 0..5 for 2x3 C
                i_local = l_idx // 3
                k_local = l_idx % 3
                g_idx = (block * 3 + k_local) * n + i_local
                W[g_idx][col] = val
    
    result = {"U": U, "V": V, "W": W}
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        if _verify(U, V, W):
            return result
    return result
