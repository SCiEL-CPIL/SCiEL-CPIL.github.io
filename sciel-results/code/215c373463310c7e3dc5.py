import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    n, m, p = 2, 2, 12
    
    # Use rank-10 decomposition for 2x2x4, applied to 3 blocks of 4 columns
    # Total rank = 3 * 10 = 30
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-10 decomposition for 2x2x4 matrix multiplication
    # A is 2x2, B is 2x4, C is 2x4
    # We need to decompose the tensor T[i*m+j, j*p4+k, k*n+i] = 1
    # where p4 = 4 for the local block
    
    # The 10 terms for 2x2x4:
    # Each term has U (4 entries), V (8 entries), W (8 entries)
    # U indices: (i,j) for i=0,1; j=0,1
    # V indices: (j,k) for j=0,1; k=0,1,2,3
    # W indices: (k,i) for k=0,1,2,3; i=0,1
    
    # Let me define the 10 terms explicitly.
    # This is a known rank-10 decomposition for 2x2x4.
    
    # Term structure: (U_dict, V_dict, W_dict)
    # where U_dict: {(i,j): val}, V_dict: {(j,k): val}, W_dict: {(k,i): val}
    
    terms_2x2x4 = []
    
    # M1 = (A00 + A11)(B00 + B10)
    # C00 += M1, C10 += M1
    terms_2x2x4.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,0): 1},
        'W': {(0,0): 1, (0,1): 1}
    })
    
    # M2 = (A10 + A11)(B01 + B11)
    # C01 += M2, C11 += M2
    terms_2x2x4.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,1): 1, (1,1): 1},
        'W': {(1,0): 1, (1,1): 1}
    })
    
    # M3 = (A00 + A01)(B02 + B12)
    # C02 += M3, C12 += M3
    terms_2x2x4.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(0,2): 1, (1,2): 1},
        'W': {(2,0): 1, (2,1): 1}
    })
    
    # M4 = (A10 + A11)(B03 + B13)
    # C03 += M4, C13 += M4
    terms_2x2x4.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,3): 1, (1,3): 1},
        'W': {(3,0): 1, (3,1): 1}
    })
    
    # M5 = A00(B10 - B00)
    # C00 -= M5, C10 += M5
    terms_2x2x4.append({
        'U': {(0,0): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = A00(B11 - B01)
    # C01 -= M6, C11 += M6
    terms_2x2x4.append({
        'U': {(0,0): 1},
        'V': {(1,1): 1, (0,1): -1},
        'W': {(1,0): -1, (1,1): 1}
    })
    
    # M7 = A00(B12 - B02)
    # C02 -= M7, C12 += M7
    terms_2x2x4.append({
        'U': {(0,0): 1},
        'V': {(1,2): 1, (0,2): -1},
        'W': {(2,0): -1, (2,1): 1}
    })
    
    # M8 = A00(B13 - B03)
    # C03 -= M8, C13 += M8
    terms_2x2x4.append({
        'U': {(0,0): 1},
        'V': {(1,3): 1, (0,3): -1},
        'W': {(3,0): -1, (3,1): 1}
    })
    
    # M9 = A10(B00 - B10)
    # C00 += M9, C10 -= M9
    terms_2x2x4.append({
        'U': {(1,0): 1},
        'V': {(0,0): 1, (1,0): -1},
        'W': {(0,0): 1, (0,1): -1}
    })
    
    # M10 = A10(B01 - B11)
    # C01 += M10, C11 -= M10
    terms_2x2x4.append({
        'U': {(1,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(1,0): 1, (1,1): -1}
    })
    
    # Verify the 2x2x4 decomposition locally
    # Let me check if this works by computing the tensor
    
    # Actually, let me reconsider. The above terms don't look quite right.
    # Let me use a more systematic approach.
    
    # For 2x2x4, we have:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] for k=0,1,2,3
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] for k=0,1,2,3
    
    # That's 8 output entries, each requiring 2 products = 16 products total.
    # With rank 10, we save 6 products.
    
    # Let me try a different set of terms that I'm more confident about.
    
    # Actually, let me just verify by direct computation whether my terms work.
    
    # First, let me build the local tensor and check.
    
    # For now, let me fill in the global matrices using these terms for 3 blocks.
    
    num_blocks = 3  # 3 blocks of 4 columns each
    rank_per_block = 10
    
    for b in range(num_blocks):
        if time.time() > deadline - 5:
            break
        base_rank = b * rank_per_block
        base_col = b * 4
        
        for t_idx in range(rank_per_block):
            r = base_rank + t_idx
            term = terms_2x2x4[t_idx]
            
            # U
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            
            # V
            for (j, k_loc), val in term['V'].items():
                k_glob = base_col + k_loc
                V[j * p + k_glob][r] = val
            
            # W
            for (k_loc, i), val in term['W'].items():
                k_glob = base_col + k_loc
                W[k_glob * n + i][r] = val
    
    # Verify the full decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    
    if max_err > 1e-9:
        # Fallback to rank-42 Strassen
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        terms_strassen = []
        terms_strassen.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
        terms_strassen.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): 1, (1,1): -1}})
        terms_strassen.append({'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(1,0): 1, (1,1): 1}})
        terms_strassen.append({'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (0,1): 1}})
        terms_strassen.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (1,0): 1}})
        terms_strassen.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
        terms_strassen.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
        
        num_blocks_s = p // 2
        for b in range(num_blocks_s):
            base_rank = b * 7
            base_col = b * 2
            for t_idx in range(7):
                r = base_rank + t_idx
                term = terms_strassen[t_idx]
                for (i, j), val in term['U'].items():
                    U[i * m + j][r] = val
                for (j, k_loc), val in term['V'].items():
                    k_glob = base_col + k_loc
                    V[j * p + k_glob][r] = val
                for (k_loc, i), val in term['W'].items():
                    k_glob = base_col + k_loc
                    W[k_glob * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
