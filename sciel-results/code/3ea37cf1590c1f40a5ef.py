import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen for 2x2 x 2x2
    # A blocks indexed 0-3: (0,0),(0,1),(1,0),(1,1)
    # B blocks indexed 0-3: (0,0),(0,1),(1,0),(1,1)
    alpha = [
        {0: 1, 3: 1},       # s0: A00+A11
        {2: 1, 3: 1},       # s1: A10+A11
        {0: 1},             # s2: A00
        {3: 1},             # s3: A11
        {0: 1, 1: 1},       # s4: A00+A01
        {2: 1, 0: -1},      # s5: A10-A00
        {1: 1, 3: -1}       # s6: A01-A11
    ]
    beta = [
        {0: 1, 3: 1},       # s0: B00+B11
        {0: 1},             # s1: B00
        {1: 1, 3: -1},      # s2: B01-B11
        {2: 1, 0: -1},      # s3: B10-B00
        {3: 1},             # s4: B11
        {0: 1, 1: 1},       # s5: B00+B01
        {2: 1, 3: 1}        # s6: B10+B11
    ]
    # recon[s] = list of (a_block, b_block, coeff) for C reconstruction
    # C[a,b] = sum over s of coeff * (X_s * Y_s)[a,b]
    recon = [
        [(0,0,1), (1,1,1)],      # C00 = s0 - s4? Let me recheck
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)]
    ]
    
    # Actually, let me use the standard Strassen reconstruction:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0 - M3 + M5 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 + M4 - M5
    
    recon = [
        [(0,0,1), (1,1,-1), (0,1,1), (1,0,-1)],  # C00
        [(0,1,1), (1,1,1)],                        # C01
        [(1,0,1), (1,1,1)],                        # C10
        [(0,0,1), (0,1,-1), (1,1,1), (1,0,-1)]    # C11
    ]
    # recon[c_idx] where c_idx maps to (a_block, b_block) in C
    # C has 4 blocks: (0,0),(0,1),(1,0),(1,1) indexed 0,1,2,3
    
    col = 0
    for s in range(7):
        # X_s is 2x2, Y_s is 2x3
        # Compute X_s * Y_s:
        # C_out[:,0:2] via inner Strassen (7 terms)
        # C_out[:,2] via schoolbook (4 terms)
        
        # Inner Strassen for columns 0,1 of Y_s
        for si in range(7):
            if time.time() > deadline - 5:
                break
            
            # U: entries of X_s weighted by alpha[si]
            for a in range(4):
                ia = alpha[si].get(a, 0)
                if ia != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        sa = alpha[s].get(r, 0)
                        if sa != 0:
                            ar, br = r // 2, r % 2
                            # A index: row = ar*2+x_row, col = br*2+x_col
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += ia * sa
            
            # V: entries of Y_s[:,0:2] weighted by beta[si]
            # Y_s has 2 rows, 3 cols. For inner Strassen, use cols 0,1
            # B blocks for Y: (0,0),(0,1),(1,0),(1,1) where first idx is row, second is col within 2x2
            for b in range(4):
                ib = beta[si].get(b, 0)
                if ib != 0:
                    y_row, y_col = b // 2, b % 2
                    for c in range(4):
                        sb = beta[s].get(c, 0)
                        if sb != 0:
                            ac, bc = c // 2, c % 2
                            # B index: row = ac*2+y_row, col = bc*3+y_col
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_col)
                            V[v_idx, col] += ib * sb
            
            # W: output C_out[a,b] for a in {0,1}, b in {0,1}
            # From inner Strassen: C_out[a,b] = sum over si of recon_inner[si] contributions
            # But we need to combine top-level recon with inner recon
            # C_out[a,b] = sum_{si} gamma_si(a,b) * M_si
            # And each M_si contributes to C_out[a,b]
            # Then top-level: C[ar*2+a, bc*3+b] gets contribution from recon[s]
            
            # Inner recon: for each output position (a,b) in 2x2, which si contribute with what coeff
            # C00: M0 - M3 + M5 - M6
            # C01: M2 + M4
            # C10: M1 + M3
            # C11: M0 - M2 + M4 - M5
            inner_recon = {
                (0,0): [(0,1),(3,-1),(5,1),(6,-1)],
                (0,1): [(2,1),(4,1)],
                (1,0): [(1,1),(3,1)],
                (1,1): [(0,1),(2,-1),(4,1),(5,-1)]
            }
            
            for (a, b), coeff in inner_recon[(si, 0)] if False else []:
                pass
            
            # For term si, it contributes to inner output positions:
            for (oa, ob), coeff in inner_recon.get((si,), []) if False else []:
                pass
            
            # Actually, M_si contributes to C_out[a,b] with coefficient from inner_recon
            # Let me redo: for each output (a,b), sum over si of coeff(si,a,b) * M_si
            # So for a specific si, it contributes to multiple (a,b) positions
            
            # For term si, the inner output contribution:
            for (oa, ob) in [(0,0),(0,1),(1,0),(1,1)]:
                coeff_inner = 0
                for (si2, c2) in inner_recon[(oa,ob)]:
                    if si2 == si:
                        coeff_inner = c2
                        break
                if coeff_inner != 0:
                    for a_top, b_top, gamma_top in recon[s]:
                        # C[a_top*2+oa, b_top*3+ob]
                        w_idx = (b_top * 3 + ob) * 4 + (a_top * 2 + oa)
                        W[w_idx, col] += coeff_inner * gamma_top
            
            col += 1
        
        # Schoolbook for column 2 of Y_s
        for u in range(2):
            for v in range(2):
                if time.time() > deadline - 5:
                    break
                
                # Term: X_s[u,v] * Y_s[v,2]
                
                # U part
                for r in range(4):
                    sa = alpha[s].get(r, 0)
                    if sa != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += sa
                
                # V part
                for c in range(4):
                    sb = beta[s].get(c, 0)
                    if sb != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += sb
                
                # W part: C_out[u, 2]
                for a_top, b_top, gamma_top in recon[s]:
                    w_idx = (b_top * 3 + 2) * 4 + (a_top * 2 + u)
                    W[w_idx, col] += gamma_top
                
                col += 1
    
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                r = round(x)
                if abs(x - r) < 1e-9:
                    new_row.append(int(r))
                else:
                    r2 = round(2*x) / 2
                    if abs(x - r2) < 1e-9:
                        new_row.append(r2)
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
