import time
import sys

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V rows: 16*j + k
    # W rows: 2*k + i
    
    # Pre-define the patterns for each of the 7 multiplications
    # Pattern is a list of (index, value) for U, V, W
    
    patterns = []
    
    # M1: (A00+A11)(B00+B11)
    # U: A00(1), A11(1) -> idx 0, 3
    # V: B00(1), B11(1) -> idx 16*0+0, 16*1+1 (relative to block)
    # W: C00(1), C11(1) -> idx 2*0+0, 2*1+1 (relative to block)
    patterns.append({
        'U': [(0, 1), (3, 1)],
        'V': [(0, 1), (17, 1)], # 16*0+0, 16*1+1
        'W': [(0, 1), (3, 1)]   # 2*0+0, 2*1+1
    })
    
    # M2: (A10+A11)B00
    # U: A10(1), A11(1) -> idx 2, 3
    # V: B00(1) -> idx 16*0+0
    # W: C10(1), C11(-1) -> idx 2*0+1, 2*1+1
    patterns.append({
        'U': [(2, 1), (3, 1)],
        'V': [(0, 1)],
        'W': [(1, 1), (3, -1)]
    })
    
    # M3: A00(B11-B01)
    # U: A00(1) -> idx 0
    # V: B11(1), B01(-1) -> idx 16*1+1, 16*0+1
    # W: C01(1), C11(1) -> idx 2*1+0, 2*1+1
    patterns.append({
        'U': [(0, 1)],
        'V': [(17, 1), (1, -1)],
        'W': [(2, 1), (3, 1)]
    })
    
    # M4: A11(B01-B00)
    # U: A11(1) -> idx 3
    # V: B01(1), B00(-1) -> idx 16*0+1, 16*0+0
    # W: C00(1), C10(1) -> idx 2*0+0, 2*0+1
    patterns.append({
        'U': [(3, 1)],
        'V': [(1, 1), (0, -1)],
        'W': [(0, 1), (1, 1)]
    })
    
    # M5: (A00+A01)B11
    # U: A00(1), A01(1) -> idx 0, 1
    # V: B11(1) -> idx 16*1+1
    # W: C00(-1), C01(1) -> idx 2*0+0, 2*1+0
    patterns.append({
        'U': [(0, 1), (1, 1)],
        'V': [(17, 1)],
        'W': [(0, -1), (2, 1)]
    })
    
    # M6: (A10-A00)(B00+B11)
    # U: A10(1), A00(-1) -> idx 2, 0
    # V: B00(1), B11(1) -> idx 16*0+0, 16*1+1
    # W: C11(1) -> idx 2*1+1
    patterns.append({
        'U': [(2, 1), (0, -1)],
        'V': [(0, 1), (17, 1)],
        'W': [(3, 1)]
    })
    
    # M7: (A01-A11)(B01+B10)
    # U: A01(1), A11(-1) -> idx 1, 3
    # V: B01(1), B10(1) -> idx 16*0+1, 16*1+0
    # W: C00(1) -> idx 2*0+0
    patterns.append({
        'U': [(1, 1), (3, -1)],
        'V': [(1, 1), (16, 1)],
        'W': [(0, 1)]
    })
    
    for l in range(8):
        # Offset for V and W indices
        # V index offset: j*p + k.
        # In block l, k is 2l or 2l+1.
        # The relative indices in patterns are 0..17.
        # We need to add 2*l to the k-part of the index.
        # V index = 16*j + k.
        # Relative index r_v = 16*j + k_rel.
        # Actual k = k_rel + 2*l.
        # So Actual V index = 16*j + k_rel + 2*l.
        
        # W index offset: k*n + i.
        # Relative index r_w = 2*k_rel + i.
        # Actual k = k_rel + 2*l.
        # So Actual W index = 2*(k_rel + 2*l) + i = 2*k_rel + i + 4*l.
        
        v_offset = 2 * l
        w_offset = 4 * l
        
        for s in range(7):
            r = 7 * l + s
            pat = patterns[s]
            
            for idx, val in pat['U']:
                U[idx][r] = val
                
            for idx, val in pat['V']:
                V[idx + v_offset][r] = val
                
            for idx, val in pat['W']:
                W[idx + w_offset][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
