"""Direct construction of rank-98 tensor for 4x4 * 4x8 using Strassen rank-49 for 4x4x4."""
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Rank-7 decomposition for 2x2 * 2x2.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),
        ({2:1, 3:1}, {0:1},   {0:0, 1:0, 2:1, 3:-1}),
        ({0:1}, {1:-1, 3:1},  {0:0, 1:1, 2:1, 3:0}),
        ({3:1}, {0:-1, 2:1},  {0:-1, 1:0, 2:1, 3:0}),
        ({0:1, 2:1}, {3:1},   {0:1, 1:-1, 2:0, 3:0}),
        ({0:-1, 2:1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),
        ({1:-1, 3:1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),
    ]
    for t, (u_c, v_c, w_c) in enumerate(terms):
        for idx, val in u_c.items():
            U[idx][t] = val
        for idx, val in v_c.items():
            V[idx][t] = val
        for idx, val in w_c.items():
            W[idx][t] = val
    return U, V, W

def _strassen_4x4_4x4():
    """Rank-49 decomposition for 4x4 * 4x4.
    U: 16x49, V: 16x49, W: 16x49
    Indices: U[i*4+j], V[j*4+k], W[k*4+i]
    """
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Strassen 7 terms at block level
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # C00 = M1 - M4 + M5 + M6
    # C01 = M3 + M1 - M5 + M7
    # C10 = M2 + M4 - M1 + M3
    # C11 = M1 - M2 + M6 + M7
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11 for U
    #                0=B00, 1=B01, 2=B10, 3=B11 for V
    #                0=C00, 1=C01, 2=C10, 3=C11 for W
    
    block_terms = [
        # M1: A=(0,3), B=(0,3), C: C00+=1, C01+=1, C10-=1, C11+=1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 1:1, 2:-1, 3:1}),
        # M2: A=(2,3), B=(0), C: C10+=1, C11=-1
        ({2:1, 3:1}, {0:1}, {0:0, 1:0, 2:1, 3:-1}),
        # M3: A=(0), B=(1,-3), C: C01+=1, C10+=1
        ({0:1}, {1:1, 3:-1}, {0:0, 1:1, 2:1, 3:0}),
        # M4: A=(3), B=(2,-0), C: C00=-1, C10+=1
        ({3:1}, {2:1, 0:-1}, {0:-1, 1:0, 2:1, 3:0}),
        # M5: A=(0,2), B=(3), C: C00+=1, C01=-1
        ({0:1, 2:1}, {3:1}, {0:1, 1:-1, 2:0, 3:0}),
        # M6: A=(2,-0), B=(0,2), C: C00+=1, C11+=1
        ({2:1, 0:-1}, {0:1, 2:1}, {0:1, 1:0, 2:0, 3:1}),
        # M7: A=(1,-3), B=(1,3), C: C01+=1, C11+=1
        ({1:1, 3:-1}, {1:1, 3:1}, {0:0, 1:1, 2:0, 3:1}),
    ]
    
    u2, v2, w2 = _strassen_2x2()
    
    def block_to_global(block_idx, local_idx):
        """Map block index and local 2x2 index to global 4x4 index.
        block_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        local_idx: 0..3 for 2x2 (row*2+col)
        """
        r = local_idx // 2
        c = local_idx % 2
        if block_idx == 0:
            return r * 4 + c
        elif block_idx == 1:
            return r * 4 + (2 + c)
        elif block_idx == 2:
            return (2 + r) * 4 + c
        elif block_idx == 3:
            return (2 + r) * 4 + (2 + c)
    
    rank_idx = 0
    for bt, (ub, vb, wb) in enumerate(block_terms):
        for st in range(7):
            for b_u, coeff_u in ub.items():
                for l in range(4):
                    coeff_scalar = u2[l][st]
                    if coeff_scalar != 0:
                        g_idx = block_to_global(b_u, l)
                        U[g_idx][rank_idx] += coeff_u * coeff_scalar
            
            for b_v, coeff_v in vb.items():
                for l in range(4):
                    coeff_scalar = v2[l][st]
                    if coeff_scalar != 0:
                        g_idx = block_to_global(b_v, l)
                        V[g_idx][rank_idx] += coeff_v * coeff_scalar
            
            for b_w, coeff_w in wb.items():
                for l in range(4):
                    coeff_scalar = w2[l][st]
                    if coeff_scalar != 0:
                        g_idx = block_to_global(b_w, l)
                        W[g_idx][rank_idx] += coeff_w * coeff_scalar
            
            rank_idx += 1
            
    return U, V, W

def _verify_4x4_4x4(U, V, W, rank):
    """Verify that the decomposition is correct."""
    import numpy as np
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U_arr[i*4+j, r] * V_arr[j*4+k, r] * W_arr[k*4+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    deadline = time.time() + 100
    
    # Get rank-49 decomposition for 4x4x4
    U49, V49, W49 = _strassen_4x4_4x4()
    
    # Verify 4x4x4
    if not _verify_4x4_4x4(U49, V49, W49, 49):
        # Fallback to schoolbook
        return _schoolbook(n, m, p)
        
    # Build 4x4x8 result using two halves of B
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First half: B columns 0-3
    for i in range(16):
        for r in range(49):
            U[i][r] = U49[i][r]
    
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 8 + k][r] = V49[j * 4 + k][r]
    
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][r] = W49[k * 4 + i][r]
                
    # Second half: B columns 4-7
    for i in range(16):
        for r in range(49):
            U[i][49 + r] = U49[i][r]
            
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 8 + (k + 4)][49 + r] = V49[j * 4 + k][r]
                
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(k + 4) * 4 + i][49 + r] = W49[k * 4 + i][r]
                
    # Verify final result
    if time.time() < deadline - 10:
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(4):
            for j in range(4):
                for k in range(8):
                    val = sum(U_arr[i*4+j, r] * V_arr[j*8+k, r] * W_arr[k*4+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        return _schoolbook(n, m, p)
                        
    return {"U": U, "V": V, "W": W}
