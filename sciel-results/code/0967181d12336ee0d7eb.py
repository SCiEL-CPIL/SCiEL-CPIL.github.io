import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # Rank is 3 * (1 + 7) = 24
    rank = 24
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (a11+a12)(b21+b22)
    # M2: (a21+a22)b12
    # M3: a11b11
    # M4: a22(b21-b11)
    # M5: (a11+a22)(b11+b22)
    # M6: (a12-a22)(b21+b22)
    # M7: (a21-a11)(b11+b12)
    
    # Local indices for 2x2x2:
    # a: i in {0,1}, j in {0,1} (local j)
    # b: j in {0,1}, l in {0,1}
    # c: i in {0,1}, l in {0,1}
    
    # Strassen terms (u, v, w) where u is for A, v for B, w for C
    # We need to map local to global.
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: [a00, a01, a10, a11] ? No, u is vector of size 4 for A?
        # In the tensor decomposition, each term is u (size n*m) x v (size m*p) x w (size p*n).
        # Here we are building the global matrices.
        
        # Let's define the 7 terms for the 2x2x2 block.
        # Local A indices: i in 0..1, j_local in 0..1.
        # Global A indices: i in 0..1, j_global = 1 + j_local (since j=0 is used by rank 1 part).
        # Local B indices: j_local in 0..1, l_local in 0..1.
        # Global B indices: j_global = 1 + j_local, l_global = 2*q + l_local.
        # Local C indices: i in 0..1, l_local in 0..1.
        # Global C indices: l_global = 2*q + l_local, i in 0..1.
        
        # Term 1: (a00+a10) * (b10+b20) -> C00?
        # Strassen M1 = (a11+a12)(b21+b22) in 1-based.
        # 0-based: M1 = (a[0,0]+a[1,0]) * (b[1,0]+b[2,0])?
        # Let's stick to the standard Strassen definitions carefully mapped.
        
        # Standard Strassen (0-based a, b, c):
        # M1 = (a[0,0] + a[1,0]) * (b[1,0] + b[2,0]) ? No.
        # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]].
        # M1 = (a00 + a10) * (b10 + b11)
        # M2 = (a10 + a11) * b01
        # M3 = a00 * b00
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a11) * (b00 + b11)
        # M6 = (a01 - a11) * (b10 + b11)
        # M7 = (a10 - a00) * (b00 + b01)
        
        # C00 = M3 + M4 + M5 - M6
        # C01 = M1 - M2 - M5 + M7
        # C10 = M2 + M4 - M7
        # C11 = M1 - M3 + M5
    ]
    
    # We will define the 7 terms explicitly as (u_vec, v_vec, w_vec)
    # u_vec is size 6 (n*m). Non-zero at i*m + j.
    # v_vec is size 18 (m*p). Non-zero at j*p + l.
    # w_vec is size 12 (p*n). Non-zero at l*n + i.
    
    for q in range(3):
        # Block q: l_global in {2q, 2q+1}
        
        # Rank 1 term for j=0
        # U: i in {0,1}, j=0.
        for i in range(2):
            U[i * 3 + 0, col] = 1.0
        # V: j=0, l in {2q, 2q+1}
        V[0 * 6 + 2*q, col] = 1.0
        V[0 * 6 + 2*q + 1, col] = 1.0
        # W: l in {2q, 2q+1}, i in {0,1}
        for i in range(2):
            W[(2*q) * 2 + i, col] = 1.0
            W[(2*q + 1) * 2 + i, col] = 1.0
        col += 1
        
        # 7 Strassen terms for j=1,2
        # Local j indices 0,1 map to global j 1,2.
        # Local l indices 0,1 map to global l 2q, 2q+1.
        
        # Helper to add a term
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: dict or list for A indices (i, j_local)
            # v_coeffs: dict or list for B indices (j_local, l_local)
            # w_coeffs: dict or list for C indices (l_local, i)
            # We'll just hardcode the 7 terms.
            pass
            
        # Term M1: (a00+a10)(b10+b11)
        # A: a00 (i=0, jL=0), a10 (i=1, jL=0)
        # B: b10 (jL=1, lL=0), b11 (jL=1, lL=1)
        # C: c00, c01, c10, c11?
        # M1 contributes to C01 and C11.
        # C01 = M1 - M2 - M5 + M7
        # C11 = M1 - M3 + M5
        # So W for M1: c01 (lL=1, i=0) +1, c11 (lL=1, i=1) +1.
        
        # Let's list the 7 terms (u, v, w)
        # u: coefficients for A_{i, jL}
        # v: coefficients for B_{jL, lL}
        # w: coefficients for C_{lL, i} (Note W index is l*n+i, so lL maps to l_global)
        
        terms = []
        
        # M1
        u = { (0,0): 1, (1,0): 1 }
        v = { (1,0): 1, (1,1): 1 }
        w = { (1,0): 1, (1,1): 1 } # lL=1, i=0 and i=1
        terms.append((u,v,w))
        
        # M2: (a10+a11)b01
        u = { (1,0): 1, (1,1): 1 }
        v = { (0,1): 1 }
        w = { (1,0): -1, (0,1): 1 } # C01=-M2, C10=M2
        terms.append((u,v,w))
        
        # M3: a00b00
        u = { (0,0): 1 }
        v = { (0,0): 1 }
        w = { (0,0): 1, (1,1): -1 } # C00=M3, C11=-M3
        terms.append((u,v,w))
        
        # M4: a11(b10-b00)
        u = { (1,1): 1 }
        v = { (1,0): 1, (0,0): -1 }
        w = { (0,0): 1, (0,1): 1 } # C00=M4, C10=M4
        terms.append((u,v,w))
        
        # M5: (a00+a11)(b00+b11)
        u = { (0,0): 1, (1,1): 1 }
        v = { (0,0): 1, (1,1): 1 }
        w = { (0,0): 1, (1,0): -1, (1,1): 1 } # C00=M5, C01=-M5, C11=M5
        terms.append((u,v,w))
        
        # M6: (a01-a11)(b10+b11)
        u = { (0,1): 1, (1,1): -1 }
        v = { (1,0): 1, (1,1): 1 }
        w = { (0,0): -1 } # C00=-M6
        terms.append((u,v,w))
        
        # M7: (a10-a00)(b00+b01)
        u = { (1,0): 1, (0,0): -1 }
        v = { (0,0): 1, (0,1): 1 }
        w = { (1,0): 1, (0,1): -1 } # C10=-M7, C01=M7
        terms.append((u,v,w))
        
        for u, v, w in terms:
            # Fill U
            for (i, jL), val in u.items():
                j_global = 1 + jL
                U[i * 3 + j_global, col] += val
            # Fill V
            for (jL, lL), val in v.items():
                j_global = 1 + jL
                l_global = 2*q + lL
                V[j_global * 6 + l_global, col] += val
            # Fill W
            for (lL, i), val in w.items():
                l_global = 2*q + lL
                W[l_global * 2 + i, col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
