import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    margin = 5.0
    
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen 2x2x2 decomposition
    # a_off: global row offset for A (i is 0..1, so a_off is 0)
    # ja_off: global col offset for A (j is 0..1 local)
    # jb_off: global row offset for B (j is 0..1 local)
    # k_off: global col offset for B (k is 0..1 local)
    # rank_start: starting column index
    def add_strassen(a_off, ja_off, jb_off, k_off, rank_start):
        # Strassen 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A10)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A (u_ij)
        # Coefficients for B (v_jk)
        # Coefficients for C (w_ki)
        
        # Note: i, j, k are local indices 0,1
        # Global i = a_off + i
        # Global j_A = ja_off + j
        # Global j_B = jb_off + j
        # Global k = k_off + k
        
        # We define the 7 terms
        # Each term has u, v, w dicts
        # u: (i, j) -> val
        # v: (j, k) -> val
        # w: (k, i) -> val
        
        terms = []
        
        # M1
        u1 = {(0,0):1, (1,1):1}
        v1 = {(0,0):1, (1,1):1}
        w1 = {(0,0):1, (0,1):0, (1,0):0, (1,1):1} # C00, C11
        # Wait, C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to distribute the w coefficients to the 7 rank terms.
        # For M1:
        # w1[0,0] += 1
        # w1[1,1] += 1
        
        # Let's build the 7 terms explicitly.
        
        # Term 0: M1
        t0_u = {(0,0):1, (1,1):1}
        t0_v = {(0,0):1, (1,1):1}
        t0_w = {(0,0):1, (1,1):1}
        terms.append((t0_u, t0_v, t0_w))
        
        # Term 1: M2
        t1_u = {(1,0):1, (1,1):1}
        t1_v = {(0,0):1}
        t1_w = {(0,1):-1, (1,0):1} # C10 += M2, C11 -= M2
        terms.append((t1_u, t1_v, t1_w))
        
        # Term 2: M3
        t2_u = {(0,0):1}
        t2_v = {(1,0):1, (1,1):-1}
        t2_w = {(0,1):1, (1,1):1} # C01 += M3, C11 += M3
        terms.append((t2_u, t2_v, t2_w))
        
        # Term 3: M4
        t3_u = {(1,1):1}
        t3_v = {(0,1):1, (0,0):-1}
        t3_w = {(0,0):1, (1,0):1} # C00 += M4, C10 += M4
        terms.append((t3_u, t3_v, t3_w))
        
        # Term 4: M5
        t4_u = {(0,0):1, (1,0):1}
        t4_v = {(1,1):1}
        t4_w = {(0,0):-1, (0,1):1} # C00 -= M5, C01 += M5
        terms.append((t4_u, t4_v, t4_w))
        
        # Term 5: M6
        t5_u = {(1,0):1, (0,0):-1}
        t5_v = {(0,0):1, (0,1):1}
        t5_w = {(1,1):1} # C11 += M6
        terms.append((t5_u, t5_v, t5_w))
        
        # Term 6: M7
        t6_u = {(0,1):1, (1,1):-1}
        t6_v = {(1,0):1, (1,1):1}
        t6_w = {(0,0):1} # C00 += M7
        terms.append((t6_u, t6_v, t6_w))
        
        for idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = rank_start + idx
            
            # Fill U
            for (i, j), val in u_dict.items():
                gi = a_off + i
                gj = ja_off + j
                U[gi * m + gj][col] += val
            
            # Fill V
            for (j, k), val in v_dict.items():
                gj = jb_off + j
                gk = k_off + k
                V[gj * p + gk][col] += val
            
            # Fill W
            for (k, i), val in w_dict.items():
                gk = k_off + k
                gi = a_off + i
                W[gk * n + gi][col] += val

    # Apply to A1 B1 (4 blocks)
    # A1: i=0,1; j=0,1. B1: j=0,1; k=0..7.
    # Blocks l=0..3 correspond to k=2l, 2l+1.
    for l in range(4):
        if time.time() > deadline - margin:
            break
        add_strassen(0, 0, 0, 2*l, l*7)
        
    # Apply to A2 B2 (4 blocks)
    # A2: i=0,1; j=2,3. B2: j=2,3; k=0..7.
    # Blocks l=0..3 correspond to k=2l, 2l+1.
    for l in range(4):
        if time.time() > deadline - margin:
            break
        add_strassen(0, 2, 2, 2*l, 28 + l*7)
        
    return {"U": U, "V": V, "W": W}
